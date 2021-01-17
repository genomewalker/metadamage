import typer
from typing import Optional, List, Union
from pathlib import Path
from click_help_colors import HelpColorsGroup, HelpColorsCommand
from rich.console import Console
from dataclasses import dataclass, field

# from MADpy import fileloader
# from MADpy import fit
# from MADpy import plot
from MADpy import utils
from main import main


__version__ = "0.1.0"


def version_callback(value: bool):
    if value:
        typer.echo(f"Awesome CLI Version: {__version__}")
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

    def command(self, *args, cls=CustomHelpColorsCommand, **kwargs) -> typer.Typer.command:
        return super().command(*args, cls=cls, **kwargs)


app = ColorfulApp()


@app.command()
def cli(
    # input arguments (filenames)
    filenames: List[Path] = typer.Argument(..., help="username"),
    # maximum values
    max_fits: Optional[int] = typer.Option(None, help="[default: None]"),  # Optional[int] = None
    max_plots: Optional[int] = typer.Option(None, help="[default: None]"),  # Optional[int] = None
    max_cores: int = 1,
    max_position: Optional[int] = typer.Option(
        None, help="[default: None]"
    ),  # Optional[int] = None
    # boolean flags
    verbose: bool = typer.Option(False, "--verbose"),
    force_reload_files: bool = typer.Option(False, "--force-reload-files"),
    force_plots: bool = typer.Option(False, "--force-plots"),
    force_fits: bool = typer.Option(False, "--force-fits"),
    # version
    version: Optional[bool] = typer.Option(None, "--version", callback=version_callback),
):
    """Metagenomics Ancient Damage python: MADpy

    FILENAME is the name of the file(s) to fit (with the ancient-model)

    run as e.g.:

    \b
        $ MADpy --verbose --max-fits 10 --max_cores 2 ./data/input/data_ancient.txt

    or by for two files:

    \b
        $ MADpy --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt

    """

    d_cfg = {
        # "filenames": filenames,
        #
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


if __name__ == "__main__":
    app()


# d_help = {
#     "verbose": "Verbose. Flag. Default is True.",
#     "number_of_fits": "Number of fits to make. Default is 10. '-1' or '0' indicates to fit all TaxIDs.",
#     "number_of_plots": "Number of plots to make. Default is 10.",
#     "make_plots": "Make plots. Flag. Default is True.",
#     "make_fits": "Make plots. Flag. Default is True.",
#     "num_cores": "Number of cores to use. Default is 1.",
#     "force_plots": "Force plots. Flag. Default is False.",
# }


# def option(string, **kwargs):
#     string_no_dashes = string.replace("--", "")
#     try:
#         help = d_help[string_no_dashes]
#     except KeyError as e:
#         print(f"\n'd_help' does not contain {string_no_dashes} \n")
#         raise e
#     kwargs["help"] = help
#     return click.option(string, **kwargs)


# @click.command()
# @click.argument("filename", type=click.Path(exists=True), nargs=-1)
# @option("--verbose", is_flag=True, default=False)
# @option("--number_of_fits", default=10)
# @option("--number_of_plots", default=10)
# @option("--make_plots", default=True, is_flag=True)
# @option("--make_fits", default=True, is_flag=True)
# @option("--num_cores", default=1)
# @option("--force_plots", is_flag=True, default=False)
# @click.version_option()
# def cli(
#     filename,
#     verbose,
#     number_of_fits,
#     number_of_plots,
#     make_plots,
#     make_fits,
#     num_cores,
#     force_plots,
# ):
#     """Metagenomics Ancient Damage python: MADpy

#     FILENAME is the name of the file(s) to fit (with the ancient-model)

#     run as e.g.:

#     \b
#         $ MADpy --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt

#     or by running two files and then compare them:

#     \b
#         $ MADpy --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt

#     """

#     filenames = filename

#     # cfg = utils.DotDict(
#     #     {
#     #         "max_fits": number_of_fits,
#     #         "max_position": None,
#     #         "verbose": verbose,
#     #         "make_fits": make_fits,
#     #         "make_plots": make_plots,
#     #         "max_plots": number_of_plots,
#     #         "force_reload_files": False,
#     #         "force_plots": force_plots,
#     #         "force_fits": False,
#     #         "num_cores": num_cores,
#     #     }
#     # )

# cfg = {
#     "max_fits": number_of_fits,
#     "max_position": None,
#     "verbose": verbose,
#     "make_fits": make_fits,
#     "make_plots": make_plots,
#     "max_plots": number_of_plots,
#     "force_reload_files": False,
#     "force_plots": force_plots,
#     "force_fits": False,
#     "num_cores": num_cores,
# }

# if len(filenames) > 0:
#     main(filenames, cfg)
# else:
#     print("Running with any argument(s). Use '--help' for more help.")
