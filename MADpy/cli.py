import click
from dotmap import DotMap as DotDict

from MADpy import fileloader
from MADpy import fit
from MADpy import plot
from MADpy import utils


@click.command()
@click.argument("filename")
@click.option("--verbose", is_flag=True, default=True, help="Verbose. Flag. Default is True.")
@click.option("--number_of_fits", default=10, help="Number of fits to make. Default is 10.")
@click.option("--number_of_plots", default=10, help="Number of plots to make. Default is 10.")
@click.option("--make_plots", default=True, is_flag=True, help="Make plots. Flag. Default is True")
@click.option("--make_fits", default=True, is_flag=True, help="Make plots. Flag. Default is True")
@click.option("--num_cores", default=1, help="Number of cores to use. Default is 1.")
@click.option("--force_plots", is_flag=True, help="Force plots. Flag. Default is False")
def main(
    filename,
    verbose,
    number_of_fits,
    number_of_plots,
    make_plots,
    make_fits,
    num_cores,
    force_plots,
):
    """ filename: eg. ./data/input/data_ancient.txt"""

    cfg = DotDict(
        {
            "filename": filename,
            "name": filename,
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

    df = fileloader.load_dataframe(cfg)
    # df_top_N = fileloader.get_top_N_taxids(df, cfg.N_taxids)

    d_fits = None
    if cfg.make_fits:
        d_fits, df_results = fit.get_fits(df, cfg)

    if cfg.make_plots:

        plot.set_style()
        plot.plot_individual_error_rates(cfg, df, d_fits=d_fits)
