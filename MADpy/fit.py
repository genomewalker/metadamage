import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.random import PRNGKey as Key
from jax.scipy.special import logsumexp

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer import log_likelihood

from joblib import Parallel, delayed
from multiprocessing import current_process
import warnings
import os

# import utils
from MADpy import utils
from MADpy import fileloader

numpyro.enable_x64()


#%%


def model_PMD(z, N, y=None, phi_prior=1 / 1000):
    z = jnp.abs(z)

    p = numpyro.sample("p", dist.Beta(2, 3))  # mean = 0.4, shape = 5
    A = numpyro.sample("A", dist.Beta(2, 3))  # mean = 0.4, shape = 5
    c = numpyro.sample("c", dist.Beta(1, 9))  # mean = 0.1, shape = 10
    Dz = numpyro.deterministic("Dz", A * (1 - p) ** (z - 1) + c)
    D_max = numpyro.deterministic("D_max", A + c)

    phi = numpyro.sample("phi", dist.Exponential(phi_prior))
    theta = numpyro.deterministic("theta", phi + 2)

    alpha = numpyro.deterministic("alpha", Dz * theta)
    beta = numpyro.deterministic("beta", (1 - Dz) * theta)

    numpyro.sample("obs", dist.BetaBinomial(alpha, beta, N), obs=y)


def model_null(z, N, y=None, phi_prior=1 / 1000):
    p = numpyro.sample("p", dist.Beta(2, 3))  # mean = 0.4, shape = 5
    D_max = numpyro.deterministic("D_max", p)
    phi = numpyro.sample("phi", dist.Exponential(phi_prior))
    theta = numpyro.deterministic("theta", phi + 2)
    numpyro.sample("obs", dist.BetaBinomial(p * theta, (1 - p) * theta, N), obs=y)


#%%


def filter_out_y(data):
    return {key: value for key, value in data.items() if key != "y"}


def is_model_PMD(model):
    name = model.__name__.lower()
    if "pmd" in name:
        return True
    elif "null" in name:
        return False
    raise AssertionError(f"Model should be PMD or null, got {model}")


#%%


@jax.jit
def _get_posterior_PMD(rng_key, samples, *args, **kwargs):
    return Predictive(model_PMD, samples)(rng_key, *args, **kwargs)["obs"]


@jax.jit
def _get_posterior_null(rng_key, samples, *args, **kwargs):
    return Predictive(model_null, samples)(rng_key, *args, **kwargs)["obs"]


def get_posterior_predictive(mcmc, data):
    posterior_samples = mcmc.get_samples()
    rng_key = Key(0)
    data_no_y = filter_out_y(data)
    if is_model_PMD(mcmc.sampler.model):
        return _get_posterior_PMD(rng_key, posterior_samples, **data_no_y)
    else:
        return _get_posterior_null(rng_key, posterior_samples, **data_no_y)


#%%


def get_y_mean_and_hpdi(mcmc, data):
    posterior_predictive = get_posterior_predictive(mcmc, data)
    predictions_fraction = posterior_predictive / data["N"]
    y_mean = jnp.mean(predictions_fraction, axis=0)
    y_hpdi = numpyro.diagnostics.hpdi(predictions_fraction, prob=0.68)
    return y_mean, y_hpdi


#%%


@jax.jit
def _compute_log_likelihood_PMD(posterior_samples, data):
    return log_likelihood(model_PMD, posterior_samples, **data)["obs"]


@jax.jit
def _compute_log_likelihood_null(posterior_samples, data):
    return log_likelihood(model_null, posterior_samples, **data)["obs"]


def compute_log_likelihood(mcmc, data):
    posterior_samples = mcmc.get_samples()
    if is_model_PMD(mcmc.sampler.model):
        return _compute_log_likelihood_PMD(posterior_samples, data)
    else:
        return _compute_log_likelihood_null(posterior_samples, data)


#%%


def get_lppd_and_waic(mcmc, data):
    d_results = {}
    # get the log likehood for each (point, num_samples)
    logprob = compute_log_likelihood(mcmc, data)
    # lppd for each observation
    lppd_i = logsumexp(logprob, 0) - jnp.log(logprob.shape[0])
    d_results["lppd_i"] = lppd_i
    # lppd
    lppd = lppd_i.sum()
    d_results["lppd"] = lppd
    # waic penalty for each observation
    pWAIC_i = jnp.var(logprob, 0)
    d_results["pWAIC_i"] = pWAIC_i
    # waic penalty # the effective number of parameters penalty
    pWAIC = pWAIC_i.sum()
    d_results["pWAIC"] = pWAIC
    # waic  for each observation
    waic_i = -2 * (lppd_i - pWAIC_i)
    d_results["waic_i"] = waic_i
    # waic # prediction of  out-of-sample deviance
    waic = waic_i.sum()
    d_results["waic"] = waic
    # standard error of waic
    # waic_vec = -2 * (lppd_i - pWAIC_i)
    # waic_uncertainty = jnp.sqrt(logprob.shape[1] * jnp.var(waic_vec))
    return d_results


def SDOM(x, axis=0):
    return np.std(x, axis=axis) / np.sqrt(x.shape[axis])


def get_mean_of_variable(mcmc, variable, axis=0):
    return np.mean(mcmc.get_samples()[variable], axis=axis)


def get_SDOM_of_variable(mcmc, variable, axis=0):
    return SDOM(mcmc.get_samples()[variable], axis=axis)


def compute_waic_weight(d_results_PMD, d_results_null):
    waic = np.asarray([d_results_PMD["waic"], d_results_null["waic"]])
    delta = waic - waic.min()
    waic_weights = np.exp(-0.5 * delta) / np.exp(-0.5 * delta).sum()
    return waic_weights


def compute_n_sigma(d_results_PMD, d_results_null):
    n = len(d_results_PMD["waic_i"])
    waic_i_PMD = d_results_PMD["waic_i"]
    waic_i_null = d_results_null["waic_i"]
    dse = jnp.sqrt(n * jnp.var(waic_i_PMD - waic_i_null))
    d_waic = d_results_null["waic"] - d_results_PMD["waic"]
    n_sigma = d_waic / dse
    return n_sigma


def compare_models(mcmc_PMD, mcmc_null, data):
    d_results_PMD = get_lppd_and_waic(mcmc_PMD, data)
    d_results_null = get_lppd_and_waic(mcmc_null, data)

    d_results = {}
    d_results["D_max"] = get_mean_of_variable(mcmc_PMD, "D_max")

    n_sigma = compute_n_sigma(d_results_PMD, d_results_null)
    d_results["n_sigma"] = n_sigma.item()

    waic_weights = compute_waic_weight(d_results_PMD, d_results_null)
    d_results["waic_weight"] = waic_weights[0]
    d_results["waic_weight_null"] = waic_weights[1]

    # probability of a single succes
    d_results["p_succes"] = get_mean_of_variable(mcmc_PMD, "p")
    d_results["dispersion"] = get_mean_of_variable(mcmc_PMD, "theta")

    return d_results


#%%


def init_mcmc(model, **kwargs):
    return MCMC(NUTS(model), jit_model_args=True, **kwargs)


def fit_mcmc(mcmc, data, seed=0):
    mcmc.run(Key(seed), **data)


def use_last_state_as_warmup_state(mcmc):
    # https://github.com/pyro-ppl/numpyro/issues/539
    mcmc._warmup_state = mcmc._last_state


#%%


def group_to_numpyro_data(group):

    z = np.array(group.iloc[:15]["pos"].values)

    y_forward = np.array(group.iloc[:15]["CT"].values)
    N_forward = np.array(group.iloc[:15]["C"].values)

    y_reverse = np.array(group.iloc[-15:]["GA"].values)
    N_reverse = np.array(group.iloc[-15:]["G"].values)

    data = {
        "z": np.concatenate([z, -z]),
        "y": np.concatenate([y_forward, y_reverse]),
        "N": np.concatenate([N_forward, N_reverse]),
    }

    return data


def get_d_out(df_results, d_y_means, d_y_hpdis):
    return {"df_results": df_results, "d_y_means": d_y_means, "d_y_hpdis": d_y_hpdis}


def match_taxid_order_in_df_results(df_results, df):
    return df_results.loc[pd.unique(df.taxid)]


def fit_chunk(df, mcmc_kwargs, do_tqdm=True):

    # print(f"fit_chunk: {current_process()=} \n", flush=True)

    mcmc_PMD = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null = init_mcmc(model_null, **mcmc_kwargs)

    it = df.groupby("taxid", sort=False, observed=True)
    if do_tqdm:
        it = tqdm(it, desc="MCMC fits")

    d_results = {}
    d_y_means = {}
    d_y_hpdis = {}

    for taxid, group in it:

        data = group_to_numpyro_data(group)

        fit_mcmc(mcmc_PMD, data)
        fit_mcmc(mcmc_null, data)

        y_mean_PMD, y_hpdi_PMD = get_y_mean_and_hpdi(mcmc_PMD, data)
        y_mean_null, y_hpdi_null = get_y_mean_and_hpdi(mcmc_null, data)
        d_y_means[taxid] = {"PMD": y_mean_PMD, "null": y_mean_null}
        d_y_hpdis[taxid] = {"PMD": y_hpdi_PMD, "null": y_hpdi_null}
        d_results[taxid] = compare_models(mcmc_PMD, mcmc_null, data)

        # if False:
        #     posterior_predictive_PMD = get_posterior_predictive(mcmc_PMD, data)
        #     posterior_predictive_null = get_posterior_predictive(mcmc_null, data)
        #     use_last_state_as_warmup_state(mcmc_PMD)
        #     use_last_state_as_warmup_state(mcmc_null)

    df_results = pd.DataFrame.from_dict(d_results, orient="index")
    df_results = match_taxid_order_in_df_results(df_results, df)
    d_fits = get_d_out(df_results, d_y_means, d_y_hpdis)
    return d_fits


def compute_fits(df, mcmc_kwargs, num_cores=1, do_tqdm=True):

    N_taxids = len(pd.unique(df.taxid))
    print(f"Fitting {N_taxids} taxids using {num_cores} cores, please wait.", flush=True)

    if num_cores == 1:
        return fit_chunk(df, mcmc_kwargs, do_tqdm=do_tqdm)

    N_chunks = num_cores  # for now
    taxid_chunks = np.array_split(df.taxid.unique(), N_chunks)
    chunks = [df.loc[df.taxid.isin(chunk)] for chunk in taxid_chunks]

    if do_tqdm:
        dos = [i == 0 for i in range(N_chunks)]
    else:
        dos = [False for i in range(N_chunks)]

    utils.avoid_fontconfig_warning()

    it = zip(chunks, dos)
    generator = (delayed(fit_chunk)(chunk, mcmc_kwargs, do_tqdm=do) for chunk, do in it)
    results = Parallel(n_jobs=num_cores)(generator)

    df_results = pd.concat([res["df_results"] for res in results])
    df_results = match_taxid_order_in_df_results(df_results, df)

    d_y_means = {}
    d_y_hpdis = {}
    for result in results:
        d_y_means = {**d_y_means, **result["d_y_means"]}
        d_y_hpdis = {**d_y_hpdis, **result["d_y_hpdis"]}

    d_fits = get_d_out(df_results, d_y_means, d_y_hpdis)
    return d_fits


def save_df_results(df_results, filename):
    utils.init_parent_folder(filename)
    df_results.to_csv(filename)


def get_fits(df, cfg):

    name = utils.extract_name(cfg.filename)
    filename_df_results = "./data/fits/" + name + f"__N_taxids__{cfg.N_taxids}.csv"
    filename_d_fits = filename_df_results.replace(".csv", ".dill")

    # if file d_fits exists, use this
    if utils.file_exists(filename_d_fits, cfg.force_fits):
        if cfg.verbose:
            print(f"Loading fits from pregenerated file, {filename_d_fits}.", flush=True)
        d_fits = utils.load_dill(filename_d_fits)
        df_results = d_fits["df_results"]
        if not utils.file_exists(filename_df_results):
            save_df_results(df_results, filename_df_results)
        return d_fits, df_results

    if cfg.verbose:
        print(f"Generating fits and saving to file: {filename_d_fits}.")

    df_top_N = fileloader.get_top_N_taxids(df, cfg.N_taxids)

    num_cores = utils.get_num_cores(cfg)

    mcmc_kwargs = dict(
        progress_bar=False,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        chain_method="sequential",
    )

    d_fits = compute_fits(df, mcmc_kwargs, num_cores=num_cores)

    df_results = d_fits["df_results"]
    save_df_results(df_results, filename_df_results)
    utils.save_dill(filename_d_fits, d_fits)

    return d_fits, df_results


#%%