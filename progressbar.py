# Standard Library
from multiprocessing import Pool
import time

# Third Party
from rich.progress import Progress, track


if False:

    progress = Progress(transient=True)

    with progress:

        task1 = progress.add_task("[red]Downloading...", total=100)
        # start=False # indeterminate progress bar
        task2 = progress.add_task("[green]Processing...", total=100)
        while not progress.finished:
            progress.update(task1, advance=0.5)
            progress.advance(task2)
            time.sleep(0.01)

if False:

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


if False:

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


if __name__ == "__main__":

    with Progress() as progress:
        task_id = progress.add_task("[cyan]Completed...", total=100)

        with Pool(processes=8) as pool:
            results = pool.imap(do_work, range(100))
            for result in results:
                # progress.print(result)
                progress.advance(task_id)
