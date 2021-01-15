import click
from dotmap import DotMap as DotDict

from MADpy import fileloader
from MADpy import fit
from MADpy import plot
from MADpy import utils

from main import main


d_help = {
    "verbose": "Verbose. Flag. Default is True.",
    "number_of_fits": "Number of fits to make. Default is 10. '-1' or '0' indicates to fit all TaxIDs.",
    "number_of_plots": "Number of plots to make. Default is 10.",
    "make_plots": "Make plots. Flag. Default is True.",
    "make_fits": "Make plots. Flag. Default is True.",
    "num_cores": "Number of cores to use. Default is 1.",
    "force_plots": "Force plots. Flag. Default is False.",
}


def option(string, **kwargs):
    string_no_dashes = string.replace("--", "")
    try:
        help = d_help[string_no_dashes]
    except KeyError as e:
        print(f"\n'd_help' does not contain {string_no_dashes} \n")
        raise e
    kwargs["help"] = help
    return click.option(string, **kwargs)


@click.command()
@click.argument("filename", type=click.Path(exists=True), nargs=-1)
@option("--verbose", is_flag=True, default=False)
@option("--number_of_fits", default=10)
@option("--number_of_plots", default=10)
@option("--make_plots", default=True, is_flag=True)
@option("--make_fits", default=True, is_flag=True)
@option("--num_cores", default=1)
@option("--force_plots", is_flag=True, default=False)
@click.version_option()
def cli(
    filename,
    verbose,
    number_of_fits,
    number_of_plots,
    make_plots,
    make_fits,
    num_cores,
    force_plots,
):
    """Metagenomics Ancient Damage python: MADpy

    FILENAME is the name of the file(s) to fit (with the ancient-model)

    run as e.g.:

    \b
        $ MADpy --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt

    or by running two files and then compare them:

    \b
        $ MADpy --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt

    """

    filenames = filename

    cfg = DotDict(
        {
            "N_taxids": number_of_fits,
            "max_pos": None,
            "verbose": verbose,
            "make_fits": make_fits,
            "make_plots": make_plots,
            "max_plots": number_of_plots,
            "force_reload": False,
            "force_plots": force_plots,
            "force_fits": False,
            "num_cores": num_cores,
        }
    )

    if len(filenames) > 0:
        main(filenames, cfg)
    else:
        print("Running with any argument(s). Use '--help' for more help.")