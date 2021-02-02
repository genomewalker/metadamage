# Scientific Library
import numpy as np
import pandas as pd

from multiprocessing.managers import SharedMemoryManager
from multiprocessing import current_process
from multiprocessing.shared_memory import SharedMemory
import jax.numpy as jnp

from metadamage import fileloader, fit, plot, utils

# filename_parquet = cfg.filename_parquet
filename_parquet = "./data/parquet/KapK-198A-Ext-55-Lib-55-Index1.parquet"

df = pd.read_parquet(filename_parquet)
df = fileloader.get_top_max_fits(df, 1000)
# df.to_hdf("test.hdf5", key="df", format="table", mode="w")

mcmc_kwargs = dict(
    progress_bar=False,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    chain_method="sequential",
)


def group_to_numpyro_data(group):

    forward = "CT"
    forward_ref = forward[0]
    reverse = "GA"
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


def worker(queue_in, queue_out):

    print(f"JoinableQueue - Worker: {current_process()=}")

    mcmc_PMD = fit.init_mcmc(fit.model_PMD, **mcmc_kwargs)
    mcmc_null = fit.init_mcmc(fit.model_null, **mcmc_kwargs)

    while True:
        # block=True means make a blocking call to wait for items in queue
        taxid_group = queue_in.get(block=True)

        if taxid_group is None:
            break

        taxid, group = taxid_group
        # group = pd.read_hdf("test.hdf5", start=start, stop=stop + 1)
        data = group_to_numpyro_data(group)

        fit.fit_mcmc(mcmc_PMD, data)
        fit.fit_mcmc(mcmc_null, data)
        y_median_PMD, y_hpdi_PMD = fit.get_y_average_and_hpdi(
            mcmc_PMD, data, func=jnp.median
        )
        fit_result = fit.compute_fit_results(mcmc_PMD, mcmc_null, data)
        fit_result["N_alignments"] = group.N_alignments.iloc[0]
        d_fit = {"median": y_median_PMD, "hpdi": y_hpdi_PMD, "fit_result": fit_result}

        queue_out.put((taxid, d_fit))
        # queue_in.task_done()

        # print(f"JoinableQueue - Worker: put stuff in queue_out")


from multiprocessing import current_process, Manager, Pool, Process, Queue

#%%

if __name__ == "__main__" and False:

    num_cores = 6

    manager = Manager()
    queue_in = manager.Queue()
    queue_out = manager.Queue()
    the_pool = Pool(num_cores, worker, (queue_in, queue_out))

    groupby = df.groupby("taxid", sort=False, observed=True)
    N_groupby = len(groupby)
    print(f"{N_groupby} groups")

    for taxid, group in groupby:
        # pass
        # start = group.index[0]
        # stop = group.index[-1]
        # queue_in.put((taxid, start, stop))
        queue_in.put((taxid, group))

    d_fits = {}
    # Get and print results
    for i in range(N_groupby):
        taxid, d_fit = queue_out.get()
        d_fits[taxid] = d_fit

    # for _ in range(N_groupby):
    for i in range(num_cores):
        queue_in.put(None)

    # prevent adding anything more to the process pool and wait for all processes to finish
    the_pool.close()
    the_pool.join()

    fit_results = {taxid: d["fit_result"] for taxid, d in d_fits.items()}
    df_fit_results = pd.DataFrame.from_dict(fit_results, orient="index")

    print(df_fit_results)

#%%


# def writer(queue_out):
#     print(f"JoinableQueue - Writer: {current_process()=}")
#     d_fits = {}
#     counter = 0
#     while True:

#         taxid_group = queue_out.get(block=True)
#         if taxid_group is None:
#             # print(f"JoinableQueue - Writer: breaking, {counter}")
#             break

#         taxid, d_fit = taxid_group
#         d_fits[taxid] = d_fit
#         # print(f"JoinableQueue - Writer: got queue_out, {counter}")
#         counter += 1

#     fit_results = {taxid: d["fit_result"] for taxid, d in d_fits.items()}
#     df_fit_results = pd.DataFrame.from_dict(fit_results, orient="index")
#     print(df_fit_results)


# from multiprocessing import Process, Queue, JoinableQueue

# if __name__ == "__main__":

#     groupby = df.groupby("taxid", sort=False, observed=True)
#     N_groupby = len(groupby)
#     print(f"{N_groupby} groups")

#     # queue_in = JoinableQueue(10)  # allow no more than 100 items on queue
#     queue_in = JoinableQueue()
#     queue_out = Queue()

#     writer_proc = Process(target=writer, args=(queue_out,))

#     worker_procs = 6
#     for i in range(worker_procs):
#         p = Process(target=worker, args=(queue_in, queue_out))
#         p.daemon = True
#         p.start()

#     writer_proc.start()

#     for taxid, group in groupby:
#         queue_in.put((taxid, group))

#     for i in range(worker_procs):
#         queue_in.put(None)

#     queue_in.join()

#     # fit_results = {taxid: d["fit_result"] for taxid, d in d_fits.items()}
#     # df_fit_results = pd.DataFrame.from_dict(fit_results, orient="index")

#     # print(df_fit_results)

#%%

if __name__ == "__main__":

    groupby = df.groupby("taxid", sort=False, observed=True)
    N_groupby = len(groupby)
    print(f"{N_groupby} groups")

    mcmc_PMD = fit.init_mcmc(fit.model_PMD, **mcmc_kwargs)
    mcmc_null = fit.init_mcmc(fit.model_null, **mcmc_kwargs)

    d_fits = {}

    for taxid, group in groupby:

        data = group_to_numpyro_data(group)

        fit.fit_mcmc(mcmc_PMD, data)
        fit.fit_mcmc(mcmc_null, data)
        y_median_PMD, y_hpdi_PMD = fit.get_y_average_and_hpdi(
            mcmc_PMD, data, func=jnp.median
        )
        fit_result = fit.compute_fit_results(mcmc_PMD, mcmc_null, data)
        fit_result["N_alignments"] = group.N_alignments.iloc[0]
        d_fit = {"median": y_median_PMD, "hpdi": y_hpdi_PMD, "fit_result": fit_result}
        d_fits[taxid] = d_fit

    fit_results = {taxid: d["fit_result"] for taxid, d in d_fits.items()}
    df_fit_results = pd.DataFrame.from_dict(fit_results, orient="index")

    print(df_fit_results)
