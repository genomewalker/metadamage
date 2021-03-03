# Scientific Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Third Party
import jax
import jax.numpy as jnp
from jax.random import PRNGKey as Key
from joblib import delayed, Parallel
import numpyro
from numpyro import distributions as dist
from numpyro.infer import log_likelihood, MCMC, NUTS, Predictive

numpyro.enable_x64()


# fmt: off
data = {
    'z': np.array([
                    1,   2,   3,   4,   5,
                    6,   7,   8,   9,  10,
                    11,  12, 13, 14,  15,
                    -1,  -2,  -3,  -4,  -5,
                    -6,  -7,  -8,  -9, -10,
                    -11, -12, -13, -14, -15,
                    ]),
    'y': np.array([
                    1,  5,  2,  3,  1,
                    10,  3,  5,  0, 11,
                    8,  1,  2,  3,  0,
                    2,  4, 0,  5,  5,
                    2,  1,  3, 11,  3,
                    0,  1,  0,  2,  7,
                   ]),
    'N': np.array([
                    245, 138, 117, 107, 138,
                    134, 161, 137, 163, 141,
                    138, 129, 146, 142, 128,
                    295, 172, 147, 138, 124,
                    172, 157, 152, 135, 146,
                    174, 166, 146, 182, 148,
                    ])}
# fmt: on


mcmc_kwargs = dict(
    progress_bar=False,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    chain_method="sequential",
)

#%%


def model(z, N, y=None, phi_prior=1 / 1000):
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


#%%


def filter_out_y(data):
    return {key: value for key, value in data.items() if key != "y"}


#%%


@jax.jit
def _get_posterior(rng_key, samples, *args, **kwargs):
    return Predictive(model, samples)(rng_key, *args, **kwargs)["obs"]


def get_posterior_predictive(mcmc, data):
    posterior_samples = mcmc.get_samples()
    rng_key = Key(0)
    data_no_y = filter_out_y(data)
    return _get_posterior(rng_key, posterior_samples, **data_no_y)


def get_y_average_and_hpdi(mcmc, data, func=np.median, return_hpdi=True):
    """func = central tendency function, e.g. np.mean or np.median"""
    posterior_predictive = get_posterior_predictive(mcmc, data)
    predictions_fraction = posterior_predictive / data["N"]
    y_average = func(predictions_fraction, axis=0)
    if not return_hpdi:
        return y_average
    y_hpdi = numpyro.diagnostics.hpdi(predictions_fraction, prob=0.68)
    return y_average, y_hpdi


@jax.jit
def _compute_log_likelihood(posterior_samples, data):
    return log_likelihood(model, posterior_samples, **data)["obs"]


def compute_log_likelihood(mcmc, data):
    posterior_samples = mcmc.get_samples()
    return _compute_log_likelihood(posterior_samples, data)


#%%


def init_mcmc(model, **kwargs):
    return MCMC(NUTS(model), jit_model_args=True, **kwargs)


def fit_mcmc(mcmc, data, seed=0):
    mcmc.run(Key(seed), **data)


#%%

mcmc = init_mcmc(model, **mcmc_kwargs)
fit_mcmc(mcmc, data)

y_median, y_hpdi = get_y_average_and_hpdi(
    mcmc,
    data,
    func=np.median,
)

print(y_median)
print(y_hpdi)