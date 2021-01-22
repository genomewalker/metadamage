# Scientific Library
import numpy as np

# Standard Library
from collections import defaultdict
import copy
from dataclasses import dataclass, field
import multiprocessing
from multiprocessing import current_process, Manager, Pool, Process, Queue
import os
import random
import time
from time import perf_counter
from typing import List, Optional

# Third Party
from about_time.human import duration_human
from joblib import delayed, Parallel
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
    TransferSpeedColumn,
)
from rich.table import Table


#%%


@dataclass
class TableHelper:
    live: Optional[Live] = None
    rows: List = field(default_factory=list)
    total: int = 0
    has_live: bool = False
    time_start: float = perf_counter()
    times: List = field(default_factory=list)
    times_delta: List = field(default_factory=list)
    is_final_line: bool = False

    def __post_init__(self):
        if self.live is not None:
            self.has_live = True

    def _init_table(self):
        table = Table(show_header=True, header_style="bold green")
        table.add_column("Progress", justify="center")
        table.add_column("Name", style="dim")
        table.add_column("Description")
        table.add_column("Time pr. task")
        table.add_column("Total time")
        return table

    def _update_time(self):
        self.times.append(perf_counter() - self.time_start)
        times_with_zero = [0] + self.times
        self.times_delta.append(times_with_zero[-1] - times_with_zero[-2])

    def add_row(self, lst):
        if len(self.rows) > 0:
            self._update_time()
        self.rows.append(lst)
        if self.has_live:
            self.update(self.live)

    def update_current_row(self, lst):
        self.rows[-1] = lst
        if self.has_live:
            self.update(self.live)

    def update_current_description(self, s):
        self.rows[-1][-1] = s
        if self.has_live:
            self.update(self.live)

    def _get_overall_progress(self, i, add_one=True):
        if add_one:
            i += 1
        if self.total == 0:
            return f"#{i}"
        else:
            percent = i / self.total * 100
            return f"{i} • {percent:>3.0f}%"

    def _get_times(self, i):
        if i == len(self.rows) - 1 and not self.is_final_line:
            return "", ""
        else:
            return duration_human(self.times[i]), duration_human(self.times_delta[i])

    def get_table(self):
        table = self._init_table()
        for irow, row in enumerate(self.rows):
            overall_progress = self._get_overall_progress(irow)
            time, time_delta = self._get_times(irow)
            row_combined = [overall_progress] + row + [time_delta, time]
            table.add_row(*row_combined)
        return table

    def update(self, live):
        table = self.get_table()
        live.update(table)

    def finalize(self):
        self._update_time()
        self.is_final_line = True
        if self.has_live:
            self.update(self.live)


if False:

    with Live(refresh_per_second=4) as live:  # update 4 times a second to feel fluid
        table_helper = TableHelper(live, total=3)

        table_helper.add_row(["A", "42"])
        time.sleep(2)

        table_helper.add_row(["B", "43"])
        time.sleep(2)

        table_helper.add_row(["C", "44"])
        time.sleep(2)

        table_helper.update_current_description("44+1")
        time.sleep(2)

        table_helper.finalize()

#%%


progress_bar_outer = (
    "[bold green][progress.description]{task.description}:",
    BarColumn(bar_width=None, complete_style="green"),
    "[progress.percentage]{task.percentage:>3.0f}%",
    # TextColumn("[bold blue]\n{task.fields[name]}", justify="left"),
    "• Remaining:",
    TimeRemainingColumn(),
    "• Elapsed:",
    TimeElapsedColumn(),
)

progress_bar_inner = (
    TextColumn("[bold blue]{task.fields[name]}"),
    SpinnerColumn(),
    BarColumn(bar_width=None, complete_style="blue"),
)


class MyProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("progress_type") == "outer":
                self.columns = progress_bar_outer
            if task.fields.get("progress_type") == "inner":
                self.columns = progress_bar_inner
            yield self.make_tasks_table([task])


if False:

    with MyProgress() as progress:  # console=console
        task_outer = progress.add_task("task_outer", progress_type="outer", total=5)
        for job in range(5):
            task_inner = progress.add_task(
                "task_inner",
                progress_type="inner",
                total=1000,
                name=f"job={job}",
            )
            for _ in range(1000):
                time.sleep(0.001)
                progress.advance(task_inner)
            progress.advance(task_outer)

# if progress.transient:
#     progress.console.control(progress._live_render.restore_cursor())
# if progress.ipy_widget is not None and progress.transient:  # pragma: no cover
#     progress.ipy_widget.clear_output()
#     progress.ipy_widget.close()


#%%

# with Live(refresh_per_second=4) as live:  # update 4 times a second to feel fluid
#     for row in range(12):
#         time.sleep(0.4)  # arbitrary delay

#         table = Table()
#         table.add_column("Name")
#         table.add_column("Description")
#         table.add_column("Level")

#         table.add_row(f"{row}", f"description {row}", "[red]ERROR")

#         live.update(table)


# # table.columns
# # dir(table.rows[0])

# table.columns[1]._cells

if False:
    # Standard Library
    import random
    import time

    # Third Party
    from rich.live import Live
    from rich.table import Table

    def generate_table() -> Table:

        table = Table()
        table.add_column("ID")
        table.add_column("Value")
        table.add_column("Status")

        for row in range(random.randint(2, 6)):
            value = random.random() * 100
            table.add_row(
                f"{row}",
                f"{value:3.2f}",
                "[red]ERROR" if value < 50 else "[green]SUCCESS",
            )
        return table

    with Live(refresh_per_second=4) as live:
        for _ in range(40):
            time.sleep(0.4)
            live.update(generate_table())


# def worker_inner(job):
#     time.sleep(0.01)


# def run_inner_loop():
#     with Progress() as progress_inner:
#         task_inner = progress_inner.add_task("Inner", total=100)
#         for job_inner in range(100):
#             worker_inner(job_inner)
#             progress_inner.advance(task_inner)


# with Progress() as progress_outer:
#     task_outer = progress_outer.add_task("Outer", total=10)
#     for job_outer in range(10):
#         run_inner_loop()
#         progress_outer.advance(task_outer)


if False:

    # if True:

    # progress = Progress(transient=True)
    progress = Progress()

    with progress:

        task1 = progress.add_task("[red]Downloading...", total=100)
        # start=False # indeterminate progress bar
        task2 = progress.add_task("[green]Processing...", total=100)
        while not progress.finished:
            progress.update(task1, advance=0.5)
            progress.advance(task2)
            time.sleep(0.01)

    # if True:

    # Third Party
    from rich.panel import Panel
    from rich.progress import Progress

    class MyProgress(Progress):
        def get_renderables(self):
            yield Panel(self.make_tasks_table(self.tasks))

    with MyProgress() as progress:
        task = progress.add_task("twiddling thumbs", total=5)
        for job in range(5):
            progress.console.print(f"Working on job #{job}")
            time.sleep(1)
            progress.advance(task)

    # if True:

    # Standard Library
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    import os.path
    import sys
    from typing import Iterable
    from urllib.request import urlopen

    # Third Party
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    def copy_url(task_id: TaskID, url: str, path: str) -> None:
        """Copy data from a url to a local file."""
        response = urlopen(url)
        # This will break if the response doesn't contain content length
        progress.update(task_id, total=int(response.info()["Content-length"]))
        with open(path, "wb") as dest_file:
            progress.start_task(task_id)
            for data in iter(partial(response.read, 32768), b""):
                dest_file.write(data)
                progress.update(task_id, advance=len(data))

    def download(urls: Iterable[str], dest_dir: str):
        """Download multuple files to the given directory."""
        with progress:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for url in urls:
                    filename = url.split("/")[-1]
                    dest_path = os.path.join(dest_dir, filename)
                    task_id = progress.add_task(
                        "download", filename=filename, start=False
                    )
                    pool.submit(copy_url, task_id, url, dest_path)

    download(
        ["https://releases.ubuntu.com/20.04/ubuntu-20.04.1-desktop-amd64.iso"], "./"
    )

#%%

# def do_work(n):
#     time.sleep(1)
#     return n

# if __name__ == "__main__" and False:

#     with Progress() as progress:
#         task_id = progress.add_task("[cyan]Completed...", total=100)

#         with Pool(processes=8) as pool:
#             results = pool.imap(do_work, range(100))
#             for result in results:
#                 progress.print(result)
#                 progress.advance(task_id)


# def do_work_working(job_id, task_id):
#     print(f"child job_id: {job_id}, task_id: {task_id} (working)")
#     for _ in range(1000):
#         time.sleep(0.005)
#     return job_id


# def do_work_not_working(job_id, task_id, progress):
#     print(f"child job_id: {job_id}, task_id: {task_id} (not working)")
#     for _ in range(1000):
#         time.sleep(0.005)
#         if job_id == 0:
#             progress.advance(task_id)
#     return job_id


# if __name__ == "__main__":

#     with Progress() as progress:
#         task_id = progress.add_task("Downloading...", total=1000)
#         print(f"main task_id: {task_id}")

#         # working
#         generator = (delayed(do_work_working)(job_id, task_id) for job_id in range(4))
#         results = Parallel(n_jobs=4)(generator)
#         print("Finished the working version", results)
#         print("\n\n")

#         time.sleep(5)

#         # not working
#         generator = (delayed(do_work_not_working)(job_id, task_id, progress) for job_id in range(4))
#         results = Parallel(n_jobs=4)(generator)


def do_work(job_id, q):
    # print(f"child job_id: {job_id}")
    for i in range(100):
        time.sleep(0.05)
        if job_id == 0:
            q.status = i
    return job_id


# if __name__ == "__main__":
#     manager = Manager()
#     q = manager.Namespace()
#     q.status = -1

#     time.sleep(1)

#     with Progress() as progress:

#         task_id = progress.add_task("Downloading...", total=100)
#         generator = (delayed(do_work)(job_id, q) for job_id in range(4))
#         results = Parallel(n_jobs=4)(generator)
#         progress.advance(task_id)


# process_name = current_process().name
# print(f"Process Name: {process_name}")


# def save_to_file(q):
#     print(f"\nsave_to_file. PID = {os.getpid()} got q={q}, name={current_process().name}")
#     while True:
#         val = q.get()
#         print("save_to_file", val)
#         if val == "END":
#             break
#     print("returning 42")
#     return 42


def worker(i, ns):
    print(
        f"worker. PID = {os.getpid()} got i={i}, ns={ns}, name={current_process().name}"
    )
    time.sleep(5 + i)
    if i == 0:
        ns.status = 0
    # q.put(str(i))
    return os.getpid()


# from math import sqrt


def sqrt(i):
    print(
        f"worker. PID = {os.getpid()} got i={i}, ns={ns}, name={current_process().name}"
    )
    time.sleep(3)
    return i ** 2


# if __name__ == "__main__":
# manager = Manager()
# q = manager.Queue()

# print("Main PID", os.getpid(), {current_process().name})

# ns = manager.Namespace()
# ns.status = -1
# print(ns, "\n", flush=True)

# # p = Process(target=save_to_file, args=(q,))
# # p.start()
# # print("after p.start()\n", flush=True)

# with Parallel(n_jobs=4) as parallel:
#     accumulator = 0.0
#     n_iter = 0
#     while accumulator < 4:
#         results = parallel(delayed(sqrt)(accumulator + i ** 2) for i in range(4))
#         print(results, current_process().name, flush=True)
#         accumulator += sum(results)  # synchronization barrier
#         n_iter += 1

# # res = Parallel(n_jobs=4)(delayed(worker)(i, ns) for i in range(4))

# # print("\nafter Parallel\n", flush=True)

# # q.put("END")

# # print("after q.put(END)\n", flush=True)

# # p.join()

# print(ns, "\n")
# # print(res)

# # while ns.status == -1:
# #     print(ns.status)
# #     time.sleep(0.1, flush=True)


# from multiprocessing import Process, Queue


# def square(numbers, queue):
#     print(f"square, name={current_process().name}", flush=True)
#     for i in numbers:
#         time.sleep(1)
#         queue.put(i * i)
#     queue.put("END")


# if __name__ == "__main__":

#     numbers = [0, 1, 2, 3, 4]
#     print(f"MAIN, name={current_process().name}", flush=True)

#     queue = Queue()
#     square_process = Process(target=square, args=(numbers, queue))
#     square_process.start()
#     square_process.join()

#     while True:
#         val = queue.get()
#         print(val)
#         if val == "END":
#             break

# import multiprocessing


# def producer(ns, event):
#     ns.value = 42
#     event.set()


# def consumer(ns, event):
#     event.wait()
#     print("After event, consumer got:", ns.value)


# if __name__ == "__main__":
#     mgr = Manager()
#     namespace = mgr.Namespace()

#     event = multiprocessing.Event()
#     p = Process(target=producer, args=(namespace, event))
#     c = Process(target=consumer, args=(namespace, event))

#     c.start()
#     p.start()

#     c.join()
#     p.join()


#%%


class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        print(f"Consumer.run, name={current_process().name}", flush=True)
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print("%s: Exiting" % proc_name)
                self.task_queue.task_done()
                break
            print("%s: %s" % (proc_name, next_task))
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        wait = np.random.uniform(5, 10)
        time.sleep(wait)  # pretend to take some time to do the work
        return "%s * %s = %s" % (self.a, self.b, self.a * self.b)

    def __str__(self):
        return "%s * %s" % (self.a, self.b)


# if __name__ == "__main__":
#     # Establish communication queues
#     tasks = multiprocessing.JoinableQueue()
#     results = multiprocessing.Queue()

#     print(f"MAIN, name={current_process().name}", flush=True)

#     # Start consumers
#     num_consumers = 4  # multiprocessing.cpu_count() * 2
#     print("Creating %d consumers" % num_consumers)
#     consumers = [Consumer(tasks, results) for i in range(num_consumers)]
#     for w in consumers:
#         w.start()

#     # Enqueue jobs
#     num_jobs = 10
#     for i in range(num_jobs):
#         tasks.put(Task(i, i))

#     # Add a poison pill for each consumer
#     for i in range(num_consumers):
#         tasks.put(None)

#     # Wait for all of the tasks to finish
#     tasks.join()

#     # Start printing results
#     while num_jobs:
#         result = results.get()
#         print("Result:", result, current_process().name)
#         num_jobs -= 1


def writer(i, q):
    print(current_process().name)
    # wait = int(np.random.uniform(100_000_000, 200_000_000))

    if i == 0:
        wait = 150_000_000
    elif i == 1:
        wait = 100_000_000
    elif i == 2:
        wait = 200_000_000
    elif i == 3:
        wait = 175_000_000

    x = 0
    for x in range(wait):
        x += x
        if i == 0 and ((x % 100_000_000) == 0):
            q.put(f"extra: i = {i}, x={x}")
    message = f"i = {i}, x={x}"
    q.put(message)


# if __name__ == "__main__":
#     # Create multiprocessing queue
#     q = Queue()

#     # Create a group of parallel writers and start them
#     for i in range(4):
#         Process(target=writer, args=(i, q)).start()

#     # Read the queue sequentially

#     n_messages = 0

#     while n_messages < 4:
#         message = q.get()
#         print(message)
#         if not "extra" in message:
#             n_messages += 1


#%%


def writer(i, q):
    # Imitate CPU-bound work happening in writer
    delay = random.randint(1, 10)
    time.sleep(delay)

    # Put the result into the queue
    t = time.time()
    print(f"I am writer {i}: {t}, {current_process().name}")
    q.put(t)
    return f"writer {i}"


# def reader(i, q):
#     """
#     Queue reader worker
#     """

#     # Read the top message from the queue
#     message = q.get()

#     # Imitate CPU-bound work happening in reader
#     time.sleep(3)
#     print(f"I am reader {i}: {message}, {current_process().name}")
#     return f"reader {i}"


if __name__ == "__main__":
    # Create manager
    # m = Manager()
    # Create multiprocessing queue
    # q = m.Queue()
    q = Queue()
    # Create a group of parallel writers and start them
    for i in range(4):
        Process(target=writer, args=(i, q)).start()
    # Create multiprocessing pool

    while not q.empty():
        print(q.get())

    # p = Pool(4)
    # # Create a group of parallel readers and start them
    # # Number of readers is matching the number of writers
    # # However, the number of simultaneously running
    # # readers is constrained to the pool size
    # readers = []
    # for i in range(4):
    #     readers.append(p.apply_async(reader, (i, q)))
    # # Wait for the asynchrounous reader threads to finish
    # results = [r.get() for r in readers]
    # print(results)


#%%


def worker(filenames):

    # this is a very slow initialization
    init = very_slow_initialization()

    # loop over files
    for filename in filenames:

        # quite fast function once the initialization is done
        fast_function(filename, init)
