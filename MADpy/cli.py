import click
from dotmap import DotMap as DotDict

from MADpy import fileloader
from MADpy import fit
from MADpy import plot
from MADpy import utils


@click.command()
@click.argument("filename", type=click.Path(exists=True), nargs=-1)
@click.option("--verbose", is_flag=True, default=False, help="Verbose. Flag. Default is True.")
@click.option(
    "--number_of_fits",
    default=10,
    help="Number of fits to make. Default is 10. -1 or 0 indicates to fit all TaxIDs.",
)
@click.option("--number_of_plots", default=10, help="Number of plots to make. Default is 10.")
@click.option("--make_plots", default=True, is_flag=True, help="Make plots. Flag. Default is True")
@click.option("--make_fits", default=True, is_flag=True, help="Make plots. Flag. Default is True")
@click.option("--num_cores", default=1, help="Number of cores to use. Default is 1.")
@click.option(
    "--force_plots", is_flag=True, default=False, help="Force plots. Flag. Default is False"
)
@click.option(
    "--parallel_plots", is_flag=True, default=False, help="Plot in parallel. Default is False"
)
@click.version_option()  # __version__
def main(
    filename,
    verbose,
    number_of_fits,
    number_of_plots,
    make_plots,
    make_fits,
    num_cores,
    force_plots,
    parallel_plots,
):
    """Metagenomics Ancient Damage python: MADpy
    run as e.g.:

    $ MADpy --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt

    or by running first for two files and then compare them:

    MADpy --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt

    """

    filenames = filename
    all_fit_results = {}

    for filename in filenames:

        cfg = DotDict(
            {
                "filename": filename,
                "name": utils.extract_name(filename),
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
                "parallel_plots": True if num_cores > 1 and parallel_plots else False,
            }
        )
        click.echo(cfg)

        df = fileloader.load_dataframe(cfg)
        # df_top_N = fileloader.get_top_N_taxids(df, cfg.N_taxids)

        d_fits = None
        if cfg.make_fits:
            d_fits, df_results = fit.get_fits(df, cfg)
            all_fit_results[cfg.name] = df_results

        if cfg.make_plots:
            plot.set_style()
            plot.plot_error_rates(cfg, df, d_fits=d_fits)

    if len(all_fit_results) >= 1:
        plot.set_style()
        N_alignments_mins = [0, 10, 100, 1000, 10_000, 100_000]
        plot.plot_fit_results(all_fit_results, cfg, N_alignments_mins=N_alignments_mins)
