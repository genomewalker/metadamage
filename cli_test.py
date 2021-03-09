# Third Party
import typer


# Init CLI
cli = typer.Typer()


# @cli.command()
@cli.command("b")
def cli_b():
    typer.echo("Chain 1")


# @cli.command()
@cli.command("a")
def cli_a():
    typer.echo("Chain 2")


if __name__ == "__main__":
    cli()
