# Standard Library
from multiprocessing import Pool
import time

# Third Party
from rich.progress import Progress, track

#%%

import time

from rich.live import Live
from rich.table import Table


from dataclasses import dataclass, field
from collections import defaultdict
from typing import List
import copy
from typing import List, Optional
from time import perf_counter
from about_time.human import duration_human


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
    import random
    import time

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
                f"{row}", f"{value:3.2f}", "[red]ERROR" if value < 50 else "[green]SUCCESS"
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
                    task_id = progress.add_task("download", filename=filename, start=False)
                    pool.submit(copy_url, task_id, url, dest_path)

    download(["https://releases.ubuntu.com/20.04/ubuntu-20.04.1-desktop-amd64.iso"], "./")

#%%


def do_work(n):
    time.sleep(1)
    return n


if __name__ == "__main__" and False:

    with Progress() as progress:
        task_id = progress.add_task("[cyan]Completed...", total=100)

        with Pool(processes=8) as pool:
            results = pool.imap(do_work, range(100))
            for result in results:
                progress.print(result)
                progress.advance(task_id)
