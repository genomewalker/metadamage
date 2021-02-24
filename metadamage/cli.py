# Standard Library
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

# Third Party
from click_help_colors import HelpColorsCommand, HelpColorsGroup
from rich.console import Console
import typer

# First Party
from metadamage import utils
from metadamage.__version__ import __version__
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
    filenames: List[Path] = typer.Argument(...),
    # maximum values
    max_fits: Optional[int] = typer.Option(None, help="[default: None (All fits)]"),
    max_plots: Optional[int] = typer.Option(0, help="[default: 0 (No plots)]"),
    # max_plots: int = 0,
    max_cores: int = 1,
    max_position: int = typer.Option(15),
    # minimum fit values (used for deciding what to plot)
    min_damage: Optional[float] = typer.Option(None, help="[default: None]"),
    min_sigma: Optional[float] = typer.Option(None, help="[default: None]"),
    min_alignments: int = 10,
    #
    sort_by: utils.SortBy = typer.Option(utils.SortBy.alignments, case_sensitive=False),
    # sort_by: Literal["alignments", "damage", "sigma"] = "alignments",
    substitution_bases_forward: utils.SubstitutionBases = typer.Option(
        utils.SubstitutionBases.CT
    ),
    substitution_bases_reverse: utils.SubstitutionBases = typer.Option(
        utils.SubstitutionBases.GA
    ),
    # boolean flags
    force_reload_files: bool = typer.Option(False, "--force-reload-files"),
    force_fits: bool = typer.Option(False, "--force-fits"),
    force_plots: bool = typer.Option(False, "--force-plots"),
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
        "min_damage": min_damage,
        "min_sigma": min_sigma,
        "min_alignments": min_alignments,
        #
        # note: convert Enum to actual value
        "sort_by": sort_by.value,
        "substitution_bases_forward": substitution_bases_forward.value,
        "substitution_bases_reverse": substitution_bases_reverse.value,
        #
        "force_reload_files": force_reload_files,
        "force_fits": force_fits,
        "force_plots": force_plots,
        #
        "version": "0.0.0",
    }

    cfg = utils.Config(**d_cfg)
    main(filenames, cfg)


# if __name__ == "__main__":
#     app()


def main_cli():
    app(prog_name="metadamage")
