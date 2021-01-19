# Standard Library
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

# Third Party
from click_help_colors import HelpColorsCommand, HelpColorsGroup
from rich.console import Console
import typer

# First Party
from metadamage import utils
from metadamage.__init__ import __version__
from metadamage.main import main


def version_callback(value: bool):
    if value:
        typer.echo(f"Metadamage CLI, version: {__version__}")
        raise typer.Exit()


class CustomHelpColorsCommand(HelpColorsCommand):
    """Colorful command line main help. Colors one of:
    "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white", "reset",
    "bright_black", "bright_red", "bright_green", "bright_yellow",
    "bright_blue", "bright_magenta", "bright_cyan", "bright_white"
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.help_headers_color = "yellow"
        self.help_options_color = "blue"


class CustomHelpColorsGroup(HelpColorsGroup):
    # colorfull command line for subcommands
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.help_headers_color = "yellow"
        self.help_options_color = "blue"


class ColorfulApp(typer.Typer):
    def __init__(self, *args, cls=CustomHelpColorsGroup, **kwargs) -> None:
        super().__init__(*args, cls=cls, **kwargs)

    def command(
        self, *args, cls=CustomHelpColorsCommand, **kwargs
    ) -> typer.Typer.command:
        return super().command(*args, cls=cls, **kwargs)


app = ColorfulApp()


@app.command()
def cli(
    # input arguments (filenames)
    filenames: List[Path] = typer.Argument(..., help="username"),
    # maximum values
    max_fits: Optional[int] = typer.Option(None, help="[default: None]"),
    max_plots: Optional[int] = typer.Option(None, help="[default: None]"),
    max_cores: int = 1,
    max_position: int = typer.Option(15),
    # boolean flags
    verbose: bool = typer.Option(False, "--verbose"),
    force_reload_files: bool = typer.Option(False, "--force-reload-files"),
    force_plots: bool = typer.Option(False, "--force-plots"),
    force_fits: bool = typer.Option(False, "--force-fits"),
    # version
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback
    ),
):
    """Metagenomics Ancient Damage: metadamage

    FILENAME is the name of the file(s) to fit (with the ancient-model)

    run as e.g.:

    \b
        $ metadamage --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt

    or by for two files:

    \b
        $ metadamage --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt

    """

    d_cfg = {
        "max_fits": max_fits,
        "max_plots": max_plots,
        "max_cores": max_cores,
        "max_position": max_position,
        #
        "verbose": verbose,
        #
        "force_reload_files": force_reload_files,
        "force_plots": force_plots,
        "force_fits": force_fits,
        "version": __version__,
    }

    cfg = utils.Config(**d_cfg)
    main(filenames, cfg)


# if __name__ == "__main__":
#     app()


def main_cli():
    app(prog_name="metadamage")
