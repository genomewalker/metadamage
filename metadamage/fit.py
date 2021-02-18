# Scientific Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from jax.scipy.special import logsumexp
from scipy.special import logsumexp

# Standard Library
import logging
from multiprocessing import current_process, Manager, Pool, Process, Queue
import os

# Third Party
import jax
import jax.numpy as jnp
from jax.random import PRNGKey as Key
from joblib import delayed, Parallel
import numpyro
from numpyro import distributions as dist
from numpyro.infer import log_likelihood, MCMC, NUTS, Predictive
from tqdm.auto import tqdm

# First Party
from metadamage import fileloader, utils
from metadamage.progressbar import console, progress


numpyro.enable_x64()
logger = logging.getLogger(__name__)

#%%


#%%


def model_PMD(z, N, y=None, phi_prior=1 / 1000):
    z = jnp.abs(z)

    q = numpyro.sample("q", dist.Beta(2, 3))  # mean = 0.4, shape = 5
    A = numpyro.sample("A", dist.Beta(2, 3))  # mean = 0.4, shape = 5
    c = numpyro.sample("c", dist.Beta(1, 9))  # mean = 0.1, shape = 10
    Dz = numpyro.deterministic("Dz", A * (1 - q) ** (z - 1) + c)
    D_max = numpyro.deterministic("D_max", A + c)

    delta = numpyro.sample("delta", dist.Exponential(phi_prior))
    phi = numpyro.deterministic("phi", delta + 2)

    alpha = numpyro.deterministic("alpha", Dz * phi)
    beta = numpyro.deterministic("beta", (1 - Dz) * phi)

    numpyro.sample("obs", dist.BetaBinomial(alpha, beta, N), obs=y)


def model_null(z, N, y=None, phi_prior=1 / 1000):
    q = numpyro.sample("q", dist.Beta(2, 3))  # mean = 0.4, shape = 5
    D_max = numpyro.deterministic("D_max", q)
    delta = numpyro.sample("delta", dist.Exponential(phi_prior))
    phi = numpyro.deterministic("phi", delta + 2)
    numpyro.sample("obs", dist.BetaBinomial(q * phi, (1 - q) * phi, N), obs=y)


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


def get_y_average_and_hpdi(mcmc, data, func=np.median, return_hpdi=True):
    """func = central tendency function, e.g. np.mean or np.median"""
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
    logprob = np.asarray(compute_log_likelihood(mcmc, data))
    # lppd for each observation
    lppd_i = logsumexp(logprob, 0) - np.log(logprob.shape[0])
    d_results["lppd_i"] = lppd_i
    # lppd
    lppd = lppd_i.sum()
    d_results["lppd"] = lppd
    # waic penalty for each observation
    pWAIC_i = np.var(logprob, 0)
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
    dse = np.sqrt(n * np.var(waic_i_PMD - waic_i_null))
    d_waic = d_results_null["waic"] - d_results_PMD["waic"]
    n_sigma = d_waic / dse
    return n_sigma


def compute_assymmetry_combined_vs_forwardreverse(
    d_results_PMD,
    d_results_PMD_forward,
    d_results_PMD_reverse,
):

    n = len(d_results_PMD["waic_i"])

    waic_i_combined = d_results_PMD["waic_i"]
    waic_i_forwardreverse = np.concatenate(
        [
            d_results_PMD_forward["waic_i"],
            d_results_PMD_reverse["waic_i"],
        ]
    )

    dse = np.sqrt(n * np.var(waic_i_combined - waic_i_forwardreverse))

    waic_combined = d_results_PMD["waic"]
    waic_forwardreverse = d_results_PMD_forward["waic"] + d_results_PMD_reverse["waic"]

    d_waic = waic_forwardreverse - waic_combined
    n_sigma = d_waic / dse
    return n_sigma


def compute_fit_results(
    mcmc_PMD,
    mcmc_null,
    mcmc_PMD_forward_reverse,
    mcmc_null_forward_reverse,
    data,
    group,
):

    d_results_PMD = get_lppd_and_waic(mcmc_PMD, data)
    d_results_null = get_lppd_and_waic(mcmc_null, data)

    fit_result = {}
    # y_frac = y/N, on the first position
    y_frac_median, y_frac_hpdi = get_y_average_and_hpdi(mcmc_PMD, data, func=np.median)
    fit_result["D_max"] = y_frac_median[0].item()

    n_sigma = compute_n_sigma(d_results_PMD, d_results_null)
    fit_result["n_sigma"] = n_sigma.item()

    # waic_weights = compute_waic_weight(d_results_PMD, d_results_null)
    # fit_result["waic_weight"] = waic_weights[0]
    # fit_result["waic_weight_null"] = waic_weights[1]

    # y_frac = y/N, on the first position, HPDIs
    fit_result["D_max_lower_hpdi"] = y_frac_hpdi[0, 0].item()
    fit_result["D_max_upper_hpdi"] = y_frac_hpdi[1, 0].item()

    # marginalized values:

    # probability of a single succes
    fit_result["q_mean"] = get_mean_of_variable(mcmc_PMD, "q")
    # concentration or shape of beta/bino
    fit_result["concentration_mean"] = get_mean_of_variable(mcmc_PMD, "phi")
    # marginalized D_max
    fit_result["D_max_marginalized_mean"] = get_mean_of_variable(mcmc_PMD, "D_max")

    fit_result["N_alignments"] = group.N_alignments.iloc[0]
    fit_result["N_z1_forward"] = data["N"][0]
    fit_result["N_z1_reverse"] = data["N"][15]
    fit_result["N_sum_forward"] = data["N"][:15].sum()
    fit_result["N_sum_reverse"] = data["N"][15:].sum()
    fit_result["N_sum_total"] = data["N"].sum()

    fit_result["tax_name"] = group["name"].iloc[0]
    fit_result["tax_rank"] = group["rank"].iloc[0]

    add_assymetry_results_to_fit_results(
        mcmc_PMD_forward_reverse,
        mcmc_null_forward_reverse,
        data,
        fit_result,
        d_results_PMD,
    )

    add_noise_estimates(group, fit_result)

    return fit_result


def add_assymetry_results_to_fit_results(
    mcmc_PMD_forward_reverse,
    mcmc_null_forward_reverse,
    data,
    fit_result,
    d_results_PMD,
):
    """computes the assymetry between a fit to forward data and reverse data
    the assymmetry is here defined as the n_sigma (WAIC) between the two fits
    """

    # FORWARD

    data_forward = {key: val[data["z"] > 0] for key, val in data.items()}
    fit_mcmc(mcmc_PMD_forward_reverse, data_forward)
    fit_mcmc(mcmc_null_forward_reverse, data_forward)
    d_results_PMD_forward = get_lppd_and_waic(mcmc_PMD_forward_reverse, data_forward)
    d_results_null_forward = get_lppd_and_waic(mcmc_null_forward_reverse, data_forward)

    fit_result["n_sigma_forward"] = compute_n_sigma(
        d_results_PMD_forward,
        d_results_null_forward,
    )

    fit_result["D_max_forward"] = get_y_average_and_hpdi(
        mcmc_PMD_forward_reverse,
        data_forward,
        func=np.median,
        return_hpdi=False,
    )[0]

    fit_result["q_mean_forward"] = get_mean_of_variable(mcmc_PMD_forward_reverse, "q")

    # REVERSE

    data_reverse = {key: val[data["z"] < 0] for key, val in data.items()}
    fit_mcmc(mcmc_PMD_forward_reverse, data_reverse)
    fit_mcmc(mcmc_null_forward_reverse, data_reverse)
    d_results_PMD_reverse = get_lppd_and_waic(mcmc_PMD_forward_reverse, data_reverse)
    d_results_null_reverse = get_lppd_and_waic(mcmc_null_forward_reverse, data_reverse)

    fit_result["n_sigma_reverse"] = compute_n_sigma(
        d_results_PMD_reverse,
        d_results_null_reverse,
    )
    fit_result["D_max_reverse"] = get_y_average_and_hpdi(
        mcmc_PMD_forward_reverse,
        data_forward,
        func=np.median,
        return_hpdi=False,
    )[0]

    fit_result["q_mean_reverse"] = get_mean_of_variable(mcmc_PMD_forward_reverse, "q")

    fit_result["asymmetry"] = compute_assymmetry_combined_vs_forwardreverse(
        d_results_PMD,
        d_results_PMD_forward,
        d_results_PMD_reverse,
    )


def add_noise_estimates(group, fit_result):

    base_columns = cols = [
        col for col in group.columns if len(col) == 2 and col[0] != col[1]
    ]

    f_ij = group[base_columns].copy()

    f_ij.loc[f_ij.index[:15], "CT"] = np.nan
    f_ij.loc[f_ij.index[15:], "GA"] = np.nan

    f_mean = f_ij.mean(axis=0)
    noise_z = f_ij / f_mean

    fit_result["normalized_noise"] = np.nanstd(noise_z.values)
    fit_result["normalized_noise_forward"] = np.nanstd(noise_z.iloc[:15].values)
    fit_result["normalized_noise_reverse"] = np.nanstd(noise_z.iloc[15:].values)


#%%


def init_mcmc(model, **kwargs):
    return MCMC(NUTS(model), jit_model_args=True, **kwargs)


def fit_mcmc(mcmc, data, seed=0):
    mcmc.run(Key(seed), **data)


def use_last_state_as_warmup_state(mcmc):
    # https://github.com/pyro-ppl/numpyro/issues/539
    mcmc._warmup_state = mcmc._last_state


#%%


def group_to_numpyro_data(group, cfg):

    forward = cfg.substitution_bases_forward
    forward_ref = forward[0]
    reverse = cfg.substitution_bases_reverse
    reverse_ref = reverse[0]

    z = np.array(group.iloc[:15]["position"], dtype=np.int)

    y_forward = np.array(group.iloc[:15][forward], dtype=np.int)
    N_forward = np.array(group.iloc[:15][forward_ref], dtype=np.int)

    y_reverse = np.array(group.iloc[-15:][reverse], dtype=np.int)
    N_reverse = np.array(group.iloc[-15:][reverse_ref], dtype=np.int)

    data = {
        "z": np.concatenate([z, -z]),
        "y": np.concatenate([y_forward, y_reverse]),
        "N": np.concatenate([N_forward, N_reverse]),
    }

    return data


def match_taxid_order_in_df_fit_results(df_fit_results, df):
    return df_fit_results.loc[pd.unique(df.taxid)]


def fit_single_group(
    group,
    cfg,
    mcmc_PMD,
    mcmc_null,
    mcmc_PMD_forward_reverse,
    mcmc_null_forward_reverse,
):

    data = group_to_numpyro_data(group, cfg)
    fit_mcmc(mcmc_PMD, data)
    fit_mcmc(mcmc_null, data)
    # mcmc_PMD.print_summary(prob=0.68)

    y_median_PMD, y_hpdi_PMD = get_y_average_and_hpdi(
        mcmc_PMD,
        data,
        func=np.median,
    )

    fit_result = compute_fit_results(
        mcmc_PMD,
        mcmc_null,
        mcmc_PMD_forward_reverse,
        mcmc_null_forward_reverse,
        data,
        group,
    )

    d_fit = {
        "median": y_median_PMD,
        "hpdi": y_hpdi_PMD,
        "fit_result": fit_result,
    }

    # if False:
    #     posterior_predictive_PMD = get_posterior_predictive(mcmc_PMD, data)
    #     posterior_predictive_null = get_posterior_predictive(mcmc_null, data)
    #     use_last_state_as_warmup_state(mcmc_PMD)
    #     use_last_state_as_warmup_state(mcmc_null)

    return d_fit


def fit_not_parallel(df, mcmc_kwargs, cfg):

    mcmc_PMD = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null = init_mcmc(model_null, **mcmc_kwargs)
    mcmc_PMD_forward_reverse = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null_forward_reverse = init_mcmc(model_null, **mcmc_kwargs)

    groupby = df.groupby("taxid", sort=False, observed=True)
    d_fits = {}

    with progress:
        task_fit = progress.add_task(
            "task_status_fitting",
            progress_type="status",
            status="Fitting ",
            name="Fits: ",
            total=len(groupby),
        )

        for taxid, group in groupby:

            d_fit = fit_single_group(
                group,
                cfg,
                mcmc_PMD,
                mcmc_null,
                mcmc_PMD_forward_reverse,
                mcmc_null_forward_reverse,
            )

            d_fits[taxid] = d_fit

            progress.advance(task_fit)

    fit_results = {taxid: d["fit_result"] for taxid, d in d_fits.items()}
    df_fit_results = pd.DataFrame.from_dict(fit_results, orient="index")
    df_fit_results = match_taxid_order_in_df_fit_results(df_fit_results, df)
    return d_fits, df_fit_results


def worker(queue_in, queue_out, mcmc_kwargs, cfg):

    mcmc_PMD = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null = init_mcmc(model_null, **mcmc_kwargs)
    mcmc_PMD_forward_reverse = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null_forward_reverse = init_mcmc(model_null, **mcmc_kwargs)

    while True:
        # block=True means make a blocking call to wait for items in queue
        taxid_group = queue_in.get(block=True)
        if taxid_group is None:
            break
        taxid, group = taxid_group

        d_fit = fit_single_group(
            group,
            cfg,
            mcmc_PMD,
            mcmc_null,
            mcmc_PMD_forward_reverse,
            mcmc_null_forward_reverse,
        )

        queue_out.put((taxid, d_fit))


def compute_parallel_fits_with_progressbar(df, cfg, mcmc_kwargs):

    groupby = df.groupby("taxid", sort=False, observed=True)
    N_groupby = len(groupby)

    num_cores = cfg.num_cores if cfg.num_cores < N_groupby else N_groupby

    manager = Manager()
    queue_in = manager.Queue()
    queue_out = manager.Queue()
    the_pool = Pool(num_cores, worker, (queue_in, queue_out, mcmc_kwargs, cfg))

    d_fits = {}

    with progress:
        task_fit = progress.add_task(
            "task_status_fitting",
            progress_type="status",
            status="Fitting ",
            name="Fits: ",
            total=N_groupby,
        )

        logger.debug(f"\n")

        # for taxid, group in groupby:
        for i, (taxid, group) in enumerate(groupby):
            if i % (N_groupby // 3) == 0 or i == N_groupby - 1:
                logger.debug(f"queue_in.put() i {i}")
                s = f"queue_in.qsize() = {queue_in.qsize()} queue_in.empty() = {queue_in.empty()}"
                logger.debug(s)
            queue_in.put((taxid, group))
        logger.debug(f"Did put {i+1} elements into queue_in out of {N_groupby}")

        logger.debug(f"\n")

        # Get and print results
        # for _ in range(N_groupby):
        for i in range(N_groupby):
            if i % (N_groupby // 3) == 0:
                logger.debug(f"queue_out.get() i {i}")
                s = f"queue_in.qsize() = {queue_in.qsize()} queue_in.empty() = {queue_in.empty()}"
                logger.debug(s)
            taxid, d_fit = queue_out.get()
            d_fits[taxid] = d_fit
            progress.advance(task_fit)
        logger.debug(f"Received {i+1} elements from queue_out out of {N_groupby}")

    logger.debug(f"\n")

    # for _ in range(N_groupby):
    for i in range(num_cores):
        if i % (N_groupby // 3) == 0:
            logger.debug(f"queue_in.put(None) i {i}")
            s = f"queue_in.qsize() = {queue_in.qsize()} queue_in.empty() = {queue_in.empty()}"
            logger.debug(s)
        queue_in.put(None)

    # prevent adding anything more to the queue and wait for queue to empty
    # queue_in.close()
    # queue_in.join_thread()

    # # join the queue until we're finished processing results
    # queue_out.join()
    # # not closing the Queues caused me untold heartache and suffering
    # queue_in.close()
    # queue_out.close()

    logger.debug(f"\n")
    logger.debug(f"Done?")
    s = f"queue_in.qsize() = {queue_in.qsize()} queue_in.empty() = {queue_in.empty()}"
    logger.debug(s)

    # prevent adding anything more to the process pool and wait for all processes to finish
    the_pool.close()
    the_pool.join()

    logger.debug(f"\n")
    logger.debug(f"after close() and join()")
    s = f"queue_in.qsize() = {queue_in.qsize()} queue_in.empty() = {queue_in.empty()}"
    logger.debug(s)
    logger.debug(f"\n")

    return d_fits


def compute_fits(df, cfg, mcmc_kwargs):

    groupby = df.groupby("taxid", sort=False, observed=True)

    if cfg.num_cores == 1 or len(groupby) < 10:
        return fit_not_parallel(df, mcmc_kwargs, cfg)

    # utils.avoid_fontconfig_warning()
    d_fits = compute_parallel_fits_with_progressbar(df, cfg, mcmc_kwargs)
    fit_results = {taxid: d["fit_result"] for taxid, d in d_fits.items()}
    df_fit_results = pd.DataFrame.from_dict(fit_results, orient="index")
    df_fit_results = match_taxid_order_in_df_fit_results(df_fit_results, df)
    return d_fits, df_fit_results


#%%


def save_df_fit_results(df_fit_results, filename):
    utils.init_parent_folder(filename)
    df_fit_results.to_csv(filename)


def _get_fit_filenames(cfg):
    d_filename = {}
    d_filename["df_fit_results"] = cfg.filename_fit_results
    d_filename["d_fits"] = d_filename["df_fit_results"].replace(".csv", ".dill")
    return d_filename


def _load_fits(cfg):

    d_filename = _get_fit_filenames(cfg)
    d_fits, df_fit_results = utils.load_dill(d_filename["d_fits"])

    if not utils.file_exists(d_filename["df_fit_results"]):
        save_df_fit_results(df_fit_results, d_filename["df_fit_results"])

    return d_fits, df_fit_results


def get_fits(df, cfg):

    cfg.set_number_of_fits(df)

    d_filename = _get_fit_filenames(cfg)

    # if file d_fits exists, use this
    if utils.file_exists(d_filename["d_fits"], cfg.force_fits):
        logger.info(f"Loading fits from pregenerated file.")
        return _load_fits(cfg)

    logger.info(f"Generating fits and saving to file.")

    df_top_N = fileloader.get_top_max_fits(df, cfg.number_of_fits)

    mcmc_kwargs = dict(
        progress_bar=False,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        chain_method="sequential",
    )

    d_fits, df_fit_results = compute_fits(df_top_N, cfg, mcmc_kwargs)

    save_df_fit_results(df_fit_results, d_filename["df_fit_results"])
    utils.save_dill(d_filename["d_fits"], [d_fits, df_fit_results])

    return d_fits, df_fit_results


#%%


# import arviz as az

# # mcmc = mcmc_PMD
# # model = model_PMD

# data_no_y = filter_out_y(data)

# def get_InferenceData(mcmc, model):

#     posterior_samples = mcmc.get_samples()
#     posterior_predictive = Predictive(model, posterior_samples)(Key(1), **data_no_y)
#     prior = Predictive(model, num_samples=500)(Key(2), **data_no_y)

#     numpyro_data = az.from_numpyro(
#         mcmc,
#         prior=prior,
#         posterior_predictive=posterior_predictive,
#         # coords={"school": np.arange(eight_school_data["J"])},
#         # dims={"theta": ["school"]},
#     )

#     return numpyro_data

# data_PMD = get_InferenceData(mcmc_PMD, model_PMD)
# data_null = get_InferenceData(mcmc_null, model_null)

# # az.plot_bpv(data_PMD, kind="p_value")
# # az.plot_bpv(data_null, kind="p_value")

# var_names = ["A", "D_max", "q", "c", "phi"]
# # az.plot_density([data_PMD, data_null], var_names=var_names)

# az.plot_dist_comparison(data_PMD, var_names=var_names)

# # az.plot_elpd({"PMD model": data_PMD, "null model": data_null})

# # az.plot_ess(data_PMD, kind="local", var_names=var_names)

# # az.plot_forest(
# #     data_PMD,
# #     kind="ridgeplot",
# #     # var_names=var_names[:-1],
# #     combined=True,
# #     ridgeplot_overlap=3,
# #     colors="white",
# #     figsize=(9, 7),
# # )

# # az.plot_loo_pit(idata=data_PMD, y="obs")
# # az.plot_loo_pit(idata=data_PMD, y="obs", ecdf=True)

# az.plot_posterior(data_PMD, var_names=var_names)

# # az.plot_ppc(data_PMD)

# az.plot_trace(data_PMD, var_names=var_names)

# model_compare = az.compare({"PMD": data_PMD, "Null": data_null}, ic="waic", scale='deviance')

# model_compare[['rank', 'waic', 'd_waic', 'dse']]

# az.plot_compare(model_compare, insample_dev=False)
