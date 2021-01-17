import typer
from typing import Optional, List
from pathlib import Path
from click_help_colors import HelpColorsGroup, HelpColorsCommand


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


def exists(file):
    if file.is_file():
        return True
    return False


@app.command()
def main(
    filenames: List[Path] = typer.Argument(..., help="username"),
    max_fits: Optional[int] = typer.Option(None, help="[default: None]"),  # Optional[int] = None
    max_fits: Optional[int] = typer.Option(None, help="[default: None]"),  # Optional[int] = None
    max_plots: Optional[int] = typer.Option(None, help="[default: None]"),  # Optional[int] = None
    max_cores: int = 1,
    verbose: bool = typer.Option(False, "--verbose"),
    force_reload_files: bool = typer.Option(False, "--force_reload_files"),
    force_plots: bool = typer.Option(False, "--force_plots"),
    force_fits: bool = typer.Option(False, "--force_fits"),
    version: Optional[bool] = typer.Option(None, "--version", callback=version_callback),
):
    """
    Say hi to NAME, optionally with a --lastname.

    If --formal is used, say hi very formally.
    """

    for filename in filenames:
        if not exists(filename):

            continue
        print(type(filename))

        if verbose:
            typer.echo(f"Good day Ms. {filename}.")
        else:
            typer.echo(f"Hello {filename}")


if __name__ == "__main__":
    app()
