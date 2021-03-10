# Standard Library
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Union

# Third Party
import click
from click import Context
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


class OrderedCommands(click.Group):
    def list_commands(self, ctx: Context) -> Iterable[str]:
        return self.commands.keys()


app = ColorfulApp(cls=OrderedCommands)
# app = ColorfulApp(chain=True)


@app.callback()
def callback():
    """
    Metagenomics Ancient Damage: metadamage.

    First run it with the fit command:

    \b
        $ metadamage fit --help

    And subsequently visualize the results using the dashboard:

    \b
        $ metadamage dashboard --help

    """


@app.command("fit")
def cli_fit(
    # input arguments (filenames)
    filenames: List[Path] = typer.Argument(...),
    # Fit options
    max_fits: Optional[int] = typer.Option(None, help="[default: None (All fits)]"),
    max_cores: int = 1,
    max_position: int = typer.Option(15),
    # Filters
    min_alignments: int = 10,
    min_y_sum: int = 10,
    #
    # sort_by: utils.SortBy = typer.Option(utils.SortBy.alignments, case_sensitive=False),
    # sort_by: Literal["alignments", "damage", "sigma"] = "alignments",
    substitution_bases_forward: utils.SubstitutionBases = typer.Option(
        utils.SubstitutionBases.CT
    ),
    substitution_bases_reverse: utils.SubstitutionBases = typer.Option(
        utils.SubstitutionBases.GA
    ),
    # boolean flags
    force_fits: bool = typer.Option(False, "--force-fits"),
    # version
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback
    ),
):
    """Fitting Ancient Damage.

    FILENAME is the name of the file(s) to fit (with the ancient-model)

    run as e.g.:

    \b
        $ metadamage fit --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt

    or by for two files:

    \b
        $ metadamage fit --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt

    """

    d_cfg = {
        "max_fits": max_fits,
        "max_cores": max_cores,
        "max_position": max_position,
        #
        "min_alignments": min_alignments,
        "min_y_sum": min_y_sum,
        #
        # note: convert Enum to actual value
        "substitution_bases_forward": substitution_bases_forward.value,
        "substitution_bases_reverse": substitution_bases_reverse.value,
        #
        "force_fits": force_fits,

        #
        "version": "0.0.0",
    }

    cfg = utils.Config(**d_cfg)
    main(filenames, cfg)


@app.command("dashboard")
def cli_dashboard(string: str):
    """Dashboard: Visualizing Ancient Damage.

    FILENAME is the name of the file(s) to fit (with the ancient-model)

    run as e.g.:

    \b
        $ metadamage dashboard

    or by for two files:

    \b
        $ metadamage dashboard

    """

    typer.echo(f"Dashboard, string={string}.")


def cli_main():
    app(prog_name="metadamage")
