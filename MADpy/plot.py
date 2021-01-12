from jax.api import xla_computation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from pandas.core.groupby import groupby
from tqdm import tqdm

import datetime
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator

# from matplotlib.ticker import MaxNLocator
from pathlib import Path
from importlib import reload
import sigfig

from functools import partial
import subprocess

from PyPDF2 import PdfFileMerger, PdfFileReader

from MADpy import utils
from MADpy import fit
from MADpy import fileloader

import matplotlib.pyplot as plt
import matplotlib as mpl


def set_rc_params(fig_dpi=300):
    plt.rcParams["figure.figsize"] = (16, 10)
    plt.rcParams["figure.dpi"] = fig_dpi
    mpl.rc("axes", edgecolor="k", linewidth=2)


def set_style(style_path="./MADpy/style.mplstyle", fig_dpi=50):
    try:
        plt.style.use(style_path)
    except FileNotFoundError:
        print(f"Could not find Matplotlib style file. Aesthetics might not be optimal.")
    set_rc_params(fig_dpi=fig_dpi)


def str_round(x, **kwargs):
    try:
        return sigfig.round(str(x), **kwargs)
    except ValueError:
        return str(x)


def fit_results_to_string(fit_result):
    s = ""

    D_max = fit_result["D_max"]
    D_max_lower_hpdi = D_max - fit_result["D_max_lower_hpdi"]
    D_max_upper_hpdi = fit_result["D_max_upper_hpdi"] - D_max

    s += r"$D_\mathrm{max} = " + f"{D_max:.3f}" + r"_{-"
    s += f"{D_max_lower_hpdi:.3f}" + r"}^{+"
    s += f"{D_max_upper_hpdi:.3f}" + r"}$" + "\n"

    s += r"$n_\sigma = " + f"{fit_result['n_sigma']:.3f}" + r"$" + "\n"

    s += "\n"
    dispersion = utils.human_format(fit_result["dispersion_mean"])
    s += r"$\mathrm{dispersion} = " + f"{dispersion}" + r"$" + "\n"
    s += r"$p_\mathrm{succes} = " + f"{fit_result['p_succes_mean']:.3f}" + r"$" + "\n"

    s += "\n"
    N_alignments = utils.human_format(fit_result["N_alignments"])
    s += r"$N_\mathrm{alignments} = " + f"{N_alignments}" + r"$" + "\n"

    return s


from matplotlib.patches import Rectangle


def plot_single_group(group, cfg, d_fits=None, figsize=(18, 7)):

    taxid = group["taxid"].iloc[0]

    if d_fits and (taxid in d_fits):
        has_fits = True
    else:
        has_fits = False

    colors = ["C0", "C1", "C2", "C3", "C4"]

    if cfg.max_pos is None:
        max_pos = group.pos.max()
    else:
        max_pos = cfg.max_pos

    group_direction = {}
    group_direction["forward"] = utils.get_forward(group)
    group_direction["reverse"] = utils.get_reverse(group)

    fig, axes = plt.subplots(ncols=2, figsize=figsize, sharey=True, gridspec_kw={"wspace": 0.14},)

    for direction, ax in zip(["reverse", "forward"], axes):

        ax.plot(
            group_direction[direction]["pos"],
            group_direction[direction]["f_C2T"],
            ".",
            color=colors[0],
            label="C→T",
            alpha=1 if direction == "forward" else 0.3,
        )
        ax.plot(
            group_direction[direction]["pos"],
            group_direction[direction]["f_G2A"],
            ".",
            color=colors[1],
            label="G→A",
            alpha=1 if direction == "reverse" else 0.3,
        )

        ax.plot(
            group_direction[direction]["pos"],
            group_direction[direction]["f_other"],
            ".",
            color=colors[3],
            label="Other substitutions",
            alpha=0.3,
        )

    ax_reverse, ax_forward = axes

    zlim_forward = np.array(ax_forward.get_xlim())
    zlim_reverse = (-zlim_forward)[::-1]

    ax_forward.set_xlim(zlim_forward)
    ax_reverse.set_xlim(zlim_reverse)

    if has_fits:

        d_fit = d_fits[taxid]

        z = group.pos.values

        y_median = d_fit["median"]
        y_hpdi = d_fit["hpdi"]
        fit_result = d_fit["fit_result"]

        hpdi = y_hpdi.copy()
        hpdi[0, :] = y_median - hpdi[0, :]
        hpdi[1, :] = hpdi[1, :] - y_median

        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html
        kw = dict(fmt="none", color="C2", capsize=6, capthick=1.5)
        label = r"Fit (68\% HDPI)"
        ax_forward.errorbar(z[z > 0], y_median[z > 0], hpdi[:, z > 0], label=label, **kw)
        ax_reverse.errorbar(z[z < 0], y_median[z < 0], hpdi[:, z < 0], **kw)

        s = fit_results_to_string(fit_result)
        ax_reverse.text(
            0.05,
            0.90,
            s,
            transform=ax_reverse.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=30,
        )

    ax_forward.legend(loc="upper right", fontsize=30)

    ax_reverse.yaxis.tick_right()
    ax_reverse.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_forward.xaxis.set_minor_locator(MultipleLocator(1))
    ax_reverse.xaxis.set_minor_locator(MultipleLocator(1))

    # ax_reverse.grid(True, alpha=0.4)
    # ax_forward.grid(True, alpha=0.4)

    ymax_tail = 5 * group.query("abs(pos) > 7")[["f_C2T", "f_G2A"]].max().max()
    ymax_overall = max(ax_forward.get_ylim()[1], ax_reverse.get_ylim()[1])
    if (ymax_tail > ymax_overall) and (ymax_tail <= 1):
        ymax = ymax_tail
    else:
        ymax = ymax_overall

    ax_reverse.set(
        xlabel="Read Position", xticks=range(-max_pos, -1 + 1, 2), ylim=(0, ymax),
    )
    ax_reverse.set_title(" Reverse", loc="left", pad=10, fontdict=dict(fontsize=30))

    ax_forward.set(
        xlabel="Read Position", xticks=range(1, max_pos + 1, 2), ylim=(0, ymax),
    )
    ax_forward.set_title("Forward ", loc="right", pad=10, fontdict=dict(fontsize=30))

    fig.suptitle(f"Error rate frequency as a function of position.\nTaxID: {taxid}\n", fontsize=40)
    fig.subplots_adjust(top=0.75)

    return fig


#%%


def filename_to_tmp_file(filename):
    return filename.replace("figures/", "figures/tmp/").replace(".pdf", "")


# useful function for parallel saving of files
def _plot_and_save_single_group_worker(i, group, filename, cfg, d_fits):

    set_style()  # set style in child process

    try:
        fig = plot_single_group(group, cfg, d_fits)
        filename = filename_to_tmp_file(filename) + f"__{i:06d}.pdf"
        utils.init_parent_folder(filename)
        fig.savefig(filename)
        return None

    except Exception as e:
        raise e
        # taxid = group.taxid.iloc[0]
        # return taxid


from joblib import delayed


def parallel_saving_of_error_rates(cfg, df_top_N, filename, d_fits):

    groupby = df_top_N.groupby("taxid", sort=False, observed=True)

    kwargs = dict(filename=filename, cfg=cfg, d_fits=d_fits)
    generator = (
        delayed(_plot_and_save_single_group_worker)(i, group, **kwargs)
        for i, (name, group) in enumerate(groupby)
    )

    total = groupby.ngroups
    print(
        f"Plotting {utils.human_format(total)} TaxIDs in parallel using {cfg.num_cores} cores:",
        flush=True,
    )
    res = utils.ProgressParallel(use_tqdm=True, total=total, n_jobs=cfg.num_cores)(generator)
    errors = set([error for error in res if error])

    if len(errors) >= 1:
        print(f"Got errors at TaxIDs: {errors}")

    # Call the PdfFileMerger
    mergedObject = PdfFileMerger()

    # Loop through all of them and append their pages
    pdfs = sorted(Path(".").rglob(f"{filename_to_tmp_file(filename)}*.pdf"))
    for pdf in pdfs:
        mergedObject.append(PdfFileReader(str(pdf), "rb"))

    # Write all the files into a file which is named as shown below
    filename_tmp = filename.replace(".pdf", "_tmp.pdf")

    mergedObject.write(filename_tmp)

    # delete temporary files
    for pdf in pdfs:
        pdf.unlink()

    # make the combined pdf smaller by compression using the following command:
    # ps2pdf filename_big filename_small
    process = subprocess.run(["ps2pdf", filename_tmp, filename])
    Path(filename_tmp).unlink()

    Path("./figures/tmp").rmdir()


def seriel_saving_of_error_rates(cfg, df_top_N, filename, d_fits):

    groupby = df_top_N.groupby("taxid", sort=False, observed=True)
    desc = f"Plotting {utils.human_format(groupby.ngroups)} TaxIDs in seriel"

    utils.init_parent_folder(filename)
    with PdfPages(filename) as pdf:

        errors = {}
        with tqdm(groupby, desc=desc) as it:
            for taxid, group in it:
                it.set_postfix(taxid=taxid)
                # break
                try:
                    fig = plot_single_group(group, cfg, d_fits)
                    if fig:
                        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
                except Exception as e:
                    errors[taxid] = str(e)
                    continue

            d = pdf.infodict()
            d["Title"] = "Error Rate Distributions"
            d["Author"] = "Christian Michelsen"
            d["CreationDate"] = datetime.datetime.today()

            if len(errors) >= 1:
                print(f"Got errors at TaxIDs: {errors}")


#%%


def plot_error_rates(cfg, df, d_fits=None, max_plots=None):

    """
    freq_strings: list of frequencies encoded as string. First letter corresponds to reference and last letter to sequence. This way allows the following string:
    'AT' = 'A2T' = 'A.T' = 'A->T' ...
    """

    if not cfg.make_plots:
        if cfg.verbose:
            print("Not plotting any error rates since 'make_plots' is False")
        return None

    if max_plots is None:
        max_plots = cfg.max_plots

    # name = utils.extract_name(cfg.filename)
    filename = f"./figures/error_rates__{cfg.name}__N_taxids__{max_plots}.pdf"

    if utils.file_exists(filename, cfg.force_plots):
        if cfg.verbose:
            print(f"Error rates plot already exist, {filename}", flush=True)
        return None

    df_top_N = fileloader.get_top_N_taxids(df, max_plots)

    if cfg.parallel_plots:
        parallel_saving_of_error_rates(cfg, df_top_N, filename, d_fits)
    else:
        seriel_saving_of_error_rates(cfg, df_top_N, filename, d_fits)


#%%

xlim = (-3, 18)
ylim = (0, 1)
alpha_plot = 0.1

from matplotlib import colors


def transform(x_org, vmin=0, vmax=1, func=lambda x: x, xmin=None, xmax=None):
    x = func(x_org)
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    x_std = (x - xmin) / (xmax - xmin)
    x_scaled = x_std * (vmax - vmin) + vmin
    return x_scaled


#%%

from matplotlib.ticker import EngFormatter
import copy


class ExpFormatter(EngFormatter):
    def __call__(self, x, pos=None):
        s = "%s%s" % (self.format_eng(10 ** x), self.unit)
        # Remove the trailing separator when there is neither prefix nor unit
        if self.sep and s.endswith(self.sep):
            s = s[: -len(self.sep)]
        return self.fix_minus(s)


def get_custom_legend(zs, ax, vmin, vmax, func, kw_cols):

    log10 = np.log10(zs)
    log_max = int(log10.max()) + 1  # round up
    log_min = int(log10.min()) + 1  # round up

    z = 10 ** np.arange(log_min, log_max)
    s = transform(z, vmin=vmin, vmax=vmax, func=func)
    c = np.log10(z)

    # plot points outside canvas
    x = np.repeat(ax.get_xlim()[1] * 1.1, len(z))
    y = np.repeat(ax.get_ylim()[1] * 1.1, len(z))

    # make all grey scatterplot for N_alignments legend
    scatter = ax.scatter(x, y, s=s, c=c, alpha=0.5, cmap="Greys", vmin=0, vmax=1)

    kw = dict(prop="colors", fmt=ExpFormatter())
    handle, label = scatter.legend_elements(**kw)

    elements_sizes, _ = scatter.legend_elements(prop="sizes")
    for i in range(len(handle)):
        size = elements_sizes[i].get_markersize()
        handle[i].set_markersize(size)
        handle[i].set_markeredgewidth(0)

    # Get markers for names as dots in legend
    N = len(handle)
    legend_elements_colors = [
        [copy.copy(handle[i]) for i in [N // 2 - 1, N - 1]],
        [copy.copy(label[i]) for i in [N // 2 - 1, N - 1]],
    ]

    for i, (name, kw_cols) in enumerate(kw_cols.items()):
        cmap = matplotlib.cm.get_cmap(kw_cols["cmap"])
        legend_elements_colors[0][i].set_markersize(20)
        legend_elements_colors[0][i].set_markerfacecolor(cmap(0.75))
        legend_elements_colors[1][i] = name

    return (handle, label), legend_elements_colors


#%%

#%%


def minmax(x):
    return np.min(x), np.max(x)


def get_z_min_max(all_fit_results):
    z_minmax = np.array([minmax(df["N_alignments"]) for df in all_fit_results.values()])
    zmin = z_minmax[:, 0].min()
    zmax = z_minmax[:, 1].max()
    return zmin, zmax


def plot_fit_results_single_N_aligment(all_fit_results, cfg, N_alignments_min=0):

    fig, ax = plt.subplots(figsize=(10, 10))

    cmaps_list = ["Blues", "Reds", "Greens", "Purples"]
    cmaps = {name: cmap for name, cmap in zip(all_fit_results.keys(), cmaps_list)}

    zmin, zmax = get_z_min_max(all_fit_results)
    vmin, vmax, func = 10, 600, np.sqrt

    zs = np.array([], dtype=int)
    kw_cols = {}

    for name, df_res in all_fit_results.items():
        x = df_res["n_sigma"].values
        y = df_res["D_max"].values
        z = df_res["N_alignments"].values

        if N_alignments_min > 0:
            mask = z > N_alignments_min
            x = x[mask]
            y = y[mask]
            z = z[mask]
        zs = np.append(zs, z)

        s = transform(z, vmin=vmin, vmax=vmax, func=func, xmin=func(zmin), xmax=func(zmax))
        c = np.log10(z)

        kw_cols[name] = dict(cmap=cmaps[name], vmin=c.min() / 10, vmax=c.max() * 1.25, ec=None)
        ax.scatter(x, y, s=s, c=c, **kw_cols[name], alpha=0.5)

    xlabel = r"$n_\sigma$"
    ylabel = r"$D_\mathrm{max}$"
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    legend_N_alignments, legend_names = get_custom_legend(zs, ax, vmin, vmax, func, kw_cols)

    # Create a legend for the first line.
    ax.add_artist(
        plt.legend(
            *legend_N_alignments,
            title=r"$N_\mathrm{alignments}$",
            title_fontsize=40,
            fontsize=34,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
        )
    )

    # Create another legend for the second line.
    kw_leg_names = dict(loc="upper left", bbox_to_anchor=(-0.03, 0.999), fontsize=30)
    plt.legend(*legend_names, **kw_leg_names)

    title = f"Fit results "
    if N_alignments_min > 0:
        N_aligments = utils.human_format(N_alignments_min)
        title += f"with cut on " + r"$N_\mathrm{alignments}>$  " + f"{N_aligments}"
    else:
        title += "with no cut on " + r"$N_\mathrm{alignments}$"

    ax.text(
        -0.1,
        1.15,
        title,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=50,
    )

    return fig, ax


def plot_fit_results(all_fit_results, cfg, N_alignments_mins=[-1]):

    if not isinstance(N_alignments_mins, (list, tuple)):
        N_alignments_mins = [N_alignments_mins]

    if not 0 in N_alignments_mins:
        N_alignments_mins = [0] + N_alignments_mins

    filename = f"./figures/all_fit_results__N_taxids__{cfg.N_taxids}.pdf"
    utils.init_parent_folder(filename)
    with PdfPages(filename) as pdf:
        for N_alignments_min in N_alignments_mins:
            fig, ax = plot_fit_results_single_N_aligment(all_fit_results, cfg, N_alignments_min)
            # pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
            pdf.savefig(fig)

        d = pdf.infodict()
        d["Title"] = "Error Rate Distributions"
        d["Author"] = "Christian Michelsen"
        d["CreationDate"] = datetime.datetime.today()

    # else:
    # print("No fits to plot")


# %%
