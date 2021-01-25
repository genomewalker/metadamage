import multiprocessing
from multiprocessing import current_process, Manager, Pool, Process, Queue
import time
from random import randint
import os
from rich.progress import Progress
from tqdm import tqdm


def simulate_cpu_intensive_process():
    delay = randint(1, 2)
    time_start = time.perf_counter()
    x = 0
    while time.perf_counter() - time_start < delay:
        x += 1


NUM_PROCESSES = 4
NUM_QUEUE_ITEMS = 50


#%%


# def worker(queue_in, queue_out):
#     print("  worker", os.getpid(), current_process().name)
#     for func, args in iter(queue_in.get, "STOP"):
#         result = calculate(func, args)
#         queue_out.put(result)


# def calculate(func, args):
#     print("    calculate", os.getpid(), current_process().name)
#     result = func(*args)
#     return "%s says that %s%s = %s" % (
#         current_process().name,
#         func.__name__,
#         args,
#         result,
#     )


# #
# # Functions referenced by tasks
# #


# def mul(a, b):
#     time.sleep(0.5 * random.random())
#     return a * b


# def plus(a, b):
#     time.sleep(0.5 * random.random())
#     return a + b


# def test():
#     NUMBER_OF_PROCESSES = 4
#     TASKS1 = [(mul, (i, 7)) for i in range(20)]
#     TASKS2 = [(plus, (i, 8)) for i in range(10)]

#     # Create queues
#     queue_in = Queue()
#     queue_out = Queue()

#     # Submit tasks
#     for task in TASKS1:
#         queue_in.put(task)

#     # Start worker processes
#     for i in range(NUMBER_OF_PROCESSES):
#         Process(target=worker, args=(queue_in, queue_out)).start()

#     # Get and print results
#     print("Unordered results:")
#     for i in range(len(TASKS1)):
#         print("\t", queue_out.get())

#     # Add more tasks using `put()`
#     for task in TASKS2:
#         queue_in.put(task)

#     # Get and print some more results
#     for i in range(len(TASKS2)):
#         print("\t", queue_out.get())

#     # Tell child processes to stop
#     for i in range(NUMBER_OF_PROCESSES):
#         queue_in.put("STOP")


# if __name__ == "__main__":
#     test()

#     x = x


#%%


# def worker_main(queue_in, queue_out):
#     print("  worker", os.getpid(), current_process().name)
#     time.sleep(10)
#     i = 0
#     while True:
#         # block=True means make a blocking call to wait for items in queue
#         item = queue_in.get(block=True)
#         if item is None:
#             break
#         simulate_cpu_intensive_process()
#         queue_out.put(i)
#         i += 1
#     print("  stopping now", os.getpid(), current_process().name)


# if __name__ == "__main__" and True:

#     print("main", os.getpid(), current_process().name)

#     # queue_in = Queue(maxsize=4)
#     # Create queues
#     queue_in = Queue()
#     queue_out = Queue()

#     the_pool = Pool(NUM_PROCESSES, worker_main, (queue_in, queue_out))

#     with Progress() as progress:  # console=console
#         task_id = progress.add_task("task", total=NUM_QUEUE_ITEMS)

#         for i in range(NUM_QUEUE_ITEMS):
#             queue_in.put(f"hello {i}")
#         print("Finished adding to queue")

#         # Get and print results
#         results = []
#         for i in range(NUM_QUEUE_ITEMS):
#             # print("\t", queue_out.get())
#             results.append(queue_out.get())
#             progress.advance(task_id)
#         print("Finished getting results", results)

#     for i in range(NUM_PROCESSES):
#         queue_in.put(None)
#     print("No more queue")

#     # prevent adding anything more to the queue and wait for queue to empty
#     queue_in.close()
#     queue_in.join_thread()
#     print("close queue")

#     # prevent adding anything more to the process pool and wait for all processes to finish
#     the_pool.close()
#     the_pool.join()

#     print("Finished")

#     x = x

#%%


import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist

from jax import random
from numpyro.infer import MCMC, NUTS

# Third Party
import jax
import jax.numpy as jnp
from jax.random import PRNGKey as Key
from jax.scipy.special import logsumexp
from joblib import delayed, Parallel
import numpyro
from numpyro import distributions as dist
from numpyro.infer import log_likelihood, MCMC, NUTS, Predictive
from joblib import delayed, Parallel


mcmc_kwargs = dict(
    progress_bar=False,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    chain_method="sequential",
)


def init_mcmc(model, **kwargs):
    return MCMC(NUTS(model), jit_model_args=True, **kwargs)


def fit_mcmc(mcmc, data, seed=0):
    mcmc.run(Key(seed), **data)


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


def is_model_PMD(model):
    name = model.__name__.lower()
    if "pmd" in name:
        return True
    elif "null" in name:
        return False
    raise AssertionError(f"Model should be PMD or null, got {model}")


def filter_out_y(data):
    return {key: value for key, value in data.items() if key != "y"}


@jax.jit
def _get_posterior_PMD(rng_key, samples, *args, **kwargs):
    return Predictive(model_PMD, samples)(rng_key, *args, **kwargs)["obs"]


# @jax.jit
# def _get_posterior_null(rng_key, samples, *args, **kwargs):
#     return Predictive(model_null, samples)(rng_key, *args, **kwargs)["obs"]


def get_posterior_predictive(mcmc, data):
    posterior_samples = mcmc.get_samples()
    rng_key = Key(0)
    data_no_y = filter_out_y(data)
    if is_model_PMD(mcmc.sampler.model):
        return _get_posterior_PMD(rng_key, posterior_samples, **data_no_y)
    # else:
    # return _get_posterior_null(rng_key, posterior_samples, **data_no_y)


def get_y_average_and_hpdi(mcmc, data, func=jnp.median, return_hpdi=True):
    """func = central tendency function, e.g. jnp.mean or jnp.median"""
    posterior_predictive = get_posterior_predictive(mcmc, data)
    predictions_fraction = posterior_predictive / data["N"]
    y_average = func(predictions_fraction, axis=0)
    if not return_hpdi:
        return y_average
    y_hpdi = numpyro.diagnostics.hpdi(predictions_fraction, prob=0.68)
    return y_average, y_hpdi


#%%


def worker_joblib(df, do_tqdm=False):
    print("  worker_joblib", os.getpid(), current_process().name)
    mcmc_PMD = init_mcmc(model_PMD, **mcmc_kwargs)
    groupby = df.groupby("taxid", sort=False, observed=True)

    it = enumerate(groupby)
    if do_tqdm:
        it = tqdm(it, total=len(groupby))

    for i, (taxid, group) in it:
        # break
        data = group_to_numpyro_data(group)
        fit_mcmc(mcmc_PMD, data)
        y_median_PMD, y_hpdi_PMD = get_y_average_and_hpdi(
            mcmc_PMD, data, func=jnp.median
        )
    return i


def test_joblib(df):
    N_chunks = 7  # for now
    taxid_chunks = np.array_split(df.taxid.unique(), N_chunks)
    chunks = [df.loc[df.taxid.isin(chunk)] for chunk in taxid_chunks]
    dos = [i == 0 for i in range(N_chunks)]
    generator = (delayed(worker_joblib)(chunk, do) for chunk, do in zip(chunks, dos))
    results = Parallel(n_jobs=7)(generator)
    print(results)


#%%


def worker_new(queue_in, queue_out):
    print("  worker_new", os.getpid(), current_process().name)

    mcmc_PMD = init_mcmc(model_PMD, **mcmc_kwargs)

    i = 0

    while True:
        # block=True means make a blocking call to wait for items in queue
        taxid_group = queue_in.get(block=True)
        if taxid_group is None:
            break

        taxid, group = taxid_group
        data = group_to_numpyro_data(group)
        fit_mcmc(mcmc_PMD, data)
        y_median_PMD, y_hpdi_PMD = get_y_average_and_hpdi(
            mcmc_PMD, data, func=jnp.median
        )

        queue_out.put((taxid, i))
        i += 1


def test_new(df):

    print("main", os.getpid(), current_process().name)

    queue_in = Queue()
    queue_out = Queue()
    the_pool = Pool(7, worker_new, (queue_in, queue_out))

    groupby = df.groupby("taxid", sort=False, observed=True)

    with Progress() as progress:  # console=console
        task_id = progress.add_task("task", total=len(groupby))

        for i, (taxid, group) in enumerate(groupby):
            queue_in.put((taxid, group))
        print("Finished adding new stuff")

        # Get and print results
        results = []
        for _ in range(len(groupby)):
            results.append(queue_out.get())
            progress.advance(task_id)
        print("Finished getting results")

    for i, (taxid, group) in enumerate(groupby):
        queue_in.put(None)
    print("No more queue")

    # prevent adding anything more to the queue and wait for queue to empty
    queue_in.close()
    queue_in.join_thread()
    print("close queue")

    # prevent adding anything more to the process pool and wait for all processes to finish
    the_pool.close()
    the_pool.join()
    print("Finished")
    return results


if __name__ == "__main__":

    from about_time import about_time

    df = pd.read_pickle("df.pickle")

    print("\n\nRunning Joblib", flush=True)
    with about_time() as at1:
        results_joblib = test_joblib(df)
    print(results_joblib)
    print(at1.duration_human)

    print("\n\nRunning multiprocessing.Queue", flush=True)
    with about_time() as at2:
        results_new = test_new(df)
    print(at2.duration_human)


#%%


def worker(queue_in, queue_out):
    init = very_slow_initialisation()
    while True:
        filename = queue_in.get(block=True)
        if filename is None:
            break
        result = fast_function(filename, init)
        queue_out.put((filename, result))


def main(df):

    queue_in = Queue()
    queue_out = Queue()
    the_pool = Pool(NUM_CORES, worker, (queue_in, queue_out))

    filenames = get_filenames()
    N = len(filenames)

    with Progress() as progress:  # console=console
        task_id = progress.add_task("Task", total=N)

        for filename in filenames:
            queue_in.put(filename)

        # Get and print results
        results = []
        for _ in range(N):
            results.append(queue_out.get())
            progress.advance(task_id)

    for _ in range((N):
        queue_in.put(None)

    # prevent adding anything more to the queue and wait for queue to empty
    queue_in.close()
    queue_in.join_thread()

    # prevent adding anything more to the process pool and wait for all processes to finish
    the_pool.close()
    the_pool.join()

    return results
