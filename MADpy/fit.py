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


def get_y_average_and_hpdi(mcmc, data, func=jnp.median, return_hpdi=True):
    """ func = central tendency function, e.g. jnp.mean or jnp.median
    """
    posterior_predictive = get_posterior_predictive(mcmc, data)
    predictions_fraction = posterior_predictive / data["N"]
    y_average = func(predictions_fraction, axis=0)
    if not return_hpdi:
        return y_average
    y_hpdi = numpyro.diagnostics.hpdi(predictions_fraction, prob=0.68)
    return y_average, y_hpdi


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
    return np.mean(mcmc.get_samples()[variable], axis=axis).item()


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


def compute_fit_results(mcmc_PMD, mcmc_null, data):
    d_results_PMD = get_lppd_and_waic(mcmc_PMD, data)
    d_results_null = get_lppd_and_waic(mcmc_null, data)

    d_results = {}
    # y_frac = y/N, on the first position
    y_frac_median, y_frac_hpdi = get_y_average_and_hpdi(mcmc_PMD, data, func=jnp.median)
    d_results["D_max"] = y_frac_median[0].item()

    n_sigma = compute_n_sigma(d_results_PMD, d_results_null)
    d_results["n_sigma"] = n_sigma.item()

    waic_weights = compute_waic_weight(d_results_PMD, d_results_null)
    d_results["waic_weight"] = waic_weights[0]
    d_results["waic_weight_null"] = waic_weights[1]

    # y_frac = y/N, on the first position, HPDIs
    d_results["D_max_lower_hpdi"] = y_frac_hpdi[0, 0].item()
    d_results["D_max_upper_hpdi"] = y_frac_hpdi[1, 0].item()

    # marginalized values:

    # probability of a single succes
    d_results["p_succes_mean"] = get_mean_of_variable(mcmc_PMD, "p")
    # dispersion or shape of beta/bino
    d_results["dispersion_mean"] = get_mean_of_variable(mcmc_PMD, "theta")
    # marginalized D_max
    d_results["D_max_marginalized_mean"] = get_mean_of_variable(mcmc_PMD, "D_max")
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


def match_taxid_order_in_df_fit_results(df_fit_results, df):
    return df_fit_results.loc[pd.unique(df.taxid)]


def fit_chunk(df, mcmc_kwargs, do_tqdm=True):
    # print(f"fit_chunk: {current_process()=} \n", flush=True)

    mcmc_PMD = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null = init_mcmc(model_null, **mcmc_kwargs)

    it = df.groupby("taxid", sort=False, observed=True)
    if do_tqdm:
        it = tqdm(it, desc="MCMC fits")

    d_fits = {}

    for taxid, group in it:

        data = group_to_numpyro_data(group)

        fit_mcmc(mcmc_PMD, data)
        fit_mcmc(mcmc_null, data)

        y_median_PMD, y_hpdi_PMD = get_y_average_and_hpdi(mcmc_PMD, data, func=jnp.median)
        fit_result = compute_fit_results(mcmc_PMD, mcmc_null, data)
        fit_result["N_alignments"] = group.N_alignments.iloc[0]

        d_fits[taxid] = {"median": y_median_PMD, "hpdi": y_hpdi_PMD, "fit_result": fit_result}

        # if False:
        #     posterior_predictive_PMD = get_posterior_predictive(mcmc_PMD, data)
        #     posterior_predictive_null = get_posterior_predictive(mcmc_null, data)
        #     use_last_state_as_warmup_state(mcmc_PMD)
        #     use_last_state_as_warmup_state(mcmc_null)

    fit_results = {taxid: d["fit_result"] for taxid, d in d_fits.items()}

    df_fit_results = pd.DataFrame.from_dict(fit_results, orient="index")
    df_fit_results = match_taxid_order_in_df_fit_results(df_fit_results, df)
    return d_fits, df_fit_results


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

    df_fit_results = pd.concat([res[1] for res in results])
    df_fit_results = match_taxid_order_in_df_fit_results(df_fit_results, df)

    d_fits = {}
    for result in results:
        d_fits = {**d_fits, **result[0]}
    return d_fits, df_fit_results


def save_df_fit_results(df_fit_results, filename):
    utils.init_parent_folder(filename)
    df_fit_results.to_csv(filename)


def _get_fit_filenames(cfg):
    d_filename = {}
    d_filename["df_fit_results"] = f"./data/fits/{cfg.name}__N_taxids__{cfg.N_taxids}.csv"
    d_filename["d_fits"] = d_filename["df_fit_results"].replace(".csv", ".dill")
    return d_filename


def _load_fits(cfg):

    d_filename = _get_fit_filenames(cfg)

    if cfg.verbose:
        print(f"Loading fits from pregenerated file, {d_filename['d_fits']}.", flush=True)
    d_fits, df_fit_results = utils.load_dill(d_filename["d_fits"])

    if not utils.file_exists(d_filename["df_fit_results"]):
        save_df_fit_results(df_fit_results, d_filename["df_fit_results"])

    return d_fits, df_fit_results


def get_fits(df, cfg):

    d_filename = _get_fit_filenames(cfg)

    # if file d_fits exists, use this
    if utils.file_exists(d_filename["d_fits"], cfg.force_fits):
        return _load_fits(cfg)

    if cfg.verbose:
        print(f"Generating fits and saving to file: {d_filename['d_fits']}.")

    df_top_N = fileloader.get_top_N_taxids(df, cfg.N_taxids)

    num_cores = utils.get_num_cores(cfg)

    mcmc_kwargs = dict(
        progress_bar=False,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        chain_method="sequential",
    )

    d_fits, df_fit_results = compute_fits(df_top_N, mcmc_kwargs, num_cores=num_cores)

    save_df_fit_results(df_fit_results, d_filename["df_fit_results"])
    utils.save_dill(d_filename["d_fits"], [d_fits, df_fit_results])

    return d_fits, df_fit_results
