# Scientific Library
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, FormatStrFormatter, MultipleLocator
import numpy as np

# Standard Library
from collections import defaultdict
import datetime
import itertools
import logging
from pathlib import Path
import re

# Third Party
from tqdm.auto import tqdm

# First Party
from metadamage import fileloader, fit, utils
from metadamage.progressbar import console, progress


# console = utils.console


logger = logging.getLogger(__name__)


def set_rc_params(fig_dpi=300):
    plt.rcParams["figure.figsize"] = (16, 10)
    plt.rcParams["figure.dpi"] = fig_dpi
    mpl.rc("axes", edgecolor="k", linewidth=2)


def set_style(style_path=None, fig_dpi=50):
    if style_path is None:
        style_path = utils.find_style_file()
    try:
        plt.style.use(style_path)
    except OSError:
        tqdm.write(
            f"Could not find Matplotlib style file. Aesthetics might not be optimal."
        )
    set_rc_params(fig_dpi=fig_dpi)


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
    concentration = utils.human_format(fit_result["concentration_mean"])
    s += r"$\mathrm{concentration} = " + f"{concentration}" + r"$" + "\n"
    s += r"$q = " + f"{fit_result['q_mean']:.3f}" + r"$" + "\n"

    s += "\n"
    N_alignments = utils.human_format(fit_result["N_alignments"])
    s += r"$N_\mathrm{alignments} = " + f"{N_alignments}" + r"$" + "\n"

    return s


def plot_single_group(group, cfg, d_fits=None, figsize=(18, 7)):
    taxid = group["taxid"].iloc[0]

    if d_fits and (taxid in d_fits):
        has_fits = True
    else:
        has_fits = False

    colors = ["C0", "C1", "C2", "C3", "C4"]

    if cfg.max_position is None:
        max_position = group.position.max()
    else:
        max_position = cfg.max_position

    group_direction = {}
    group_direction["forward"] = utils.get_forward(group)
    group_direction["reverse"] = utils.get_reverse(group)

    forward_fraction_name = f"f_{cfg.substitution_bases_forward}"
    forward_fraction_label = (
        f"{cfg.substitution_bases_forward[0]}→{cfg.substitution_bases_forward[1]}"
    )
    reverse_fraction_name = f"f_{cfg.substitution_bases_reverse}"
    reverse_fraction_label = (
        f"{cfg.substitution_bases_reverse[0]}→{cfg.substitution_bases_reverse[1]}"
    )

    fig, axes = plt.subplots(
        ncols=2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"wspace": 0.14},
    )

    for direction, ax in zip(["reverse", "forward"], axes):

        ax.plot(
            group_direction[direction]["position"],
            group_direction[direction][forward_fraction_name],
            ".",
            color=colors[0],
            label=forward_fraction_label,
            alpha=1 if direction == "forward" else 0.3,
        )
        ax.plot(
            group_direction[direction]["position"],
            group_direction[direction][reverse_fraction_name],
            ".",
            color=colors[1],
            label=reverse_fraction_label,
            alpha=1 if direction == "reverse" else 0.3,
        )

        # ax.plot(
        #     group_direction[direction]["pos"],
        #     group_direction[direction]["f_other"],
        #     ".",
        #     color=colors[3],
        #     label="Other substitutions",
        #     alpha=0.3,
        # )

    ax_reverse, ax_forward = axes

    zlim_forward = np.array(ax_forward.get_xlim())
    zlim_reverse = (-zlim_forward)[::-1]

    ax_forward.set_xlim(zlim_forward)
    ax_reverse.set_xlim(zlim_reverse)

    if has_fits:

        d_fit = d_fits[taxid]

        z = group.position.values

        y_median = d_fit["median"]
        y_hpdi = d_fit["hpdi"]
        fit_result = d_fit["fit_result"]

        hpdi = y_hpdi.copy()
        hpdi[0, :] = y_median - hpdi[0, :]
        hpdi[1, :] = hpdi[1, :] - y_median

        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html
        kw = dict(fmt="none", color="C2", capsize=6, capthick=1.5)
        label = r"Fit (68\% HDPI)"
        ax_forward.errorbar(
            z[z > 0], y_median[z > 0], hpdi[:, z > 0], label=label, **kw
        )
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

    ymax_tail = 5 * (
        group.query("abs(position) > 7")
        .loc[:, [forward_fraction_name, reverse_fraction_name]]
        .max()
        .max()
    )
    ymax_overall = max(ax_forward.get_ylim()[1], ax_reverse.get_ylim()[1])
    if (ymax_tail > ymax_overall) and (ymax_tail <= 1):
        ymax = ymax_tail
    else:
        ymax = ymax_overall

    ax_reverse.set(
        xlabel="Read Position",
        xticks=range(-max_position, -1 + 1, 2),
        ylim=(0, ymax),
    )
    ax_reverse.set_title(" Reverse", loc="left", pad=10, fontdict=dict(fontsize=30))

    ax_forward.set(
        xlabel="Read Position",
        xticks=range(1, max_position + 1, 2),
        ylim=(0, ymax),
    )
    ax_forward.set_title("Forward ", loc="right", pad=10, fontdict=dict(fontsize=30))

    name = group["name"].iloc[0]
    rank = group["rank"].iloc[0]
    title = f"Error rate frequency as a function of position.\nTax: {taxid}, {name}, {rank}\n"
    title = title.replace("root, no rank", "root")
    fig.suptitle(title, fontsize=40)
    fig.subplots_adjust(top=0.75)

    return fig


#%%


def seriel_saving_of_error_rates(cfg, df_plot_sorted, filename, d_fits):

    groupby = df_plot_sorted.groupby("taxid", sort=False, observed=True)

    task_id_status_plotting = progress.add_task(
        "task_status_plotting",
        progress_type="status",
        status="Plotting",
        name="Plots:",
        total=len(groupby),
    )

    utils.init_parent_folder(filename)
    with PdfPages(filename) as pdf, progress:
        for taxid, group in groupby:
            fig = plot_single_group(group, cfg, d_fits)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
            progress.advance(task_id_status_plotting)

        d = pdf.infodict()
        d["Title"] = "Error Rate Distributions"
        d["Author"] = "Christian Michelsen"
        d["CreationDate"] = datetime.datetime.today()


def plot_error_rates(cfg, df, d_fits, df_results):

    """
    freq_strings: list of frequencies encoded as string. First letter corresponds to reference and last letter to sequence. This way allows the following string:
    'AT' = 'A2T' = 'A.T' = 'A->T' ...
    """

    number_of_plots = cfg.number_of_plots

    df_plot_sorted = utils.get_sorted_and_cutted_df(
        cfg,
        df,
        df_results,
    )

    filename = cfg.filename_plot_error_rates

    if utils.is_pdf_valid(filename, cfg.force_plots, N_pages=number_of_plots):
        logger.info(f"Plot of error rates already exist.")
        return None

    logger.info(f"Plotting, please wait.")
    set_style()
    seriel_saving_of_error_rates(cfg, df_plot_sorted, filename, d_fits)


#%%


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


class ExpFormatter(EngFormatter):
    def __call__(self, x, pos=None):
        s = "%s%s" % (self.format_eng(10 ** x), self.unit)
        # Remove the trailing separator when there is neither prefix nor unit
        if self.sep and s.endswith(self.sep):
            s = s[: -len(self.sep)]
        return self.fix_minus(s)


def _get_legend_handles_names(kw_cols, use_monospace_font=True):
    # get legend for the names

    # plot all legend entries if less than 10 files
    if len(kw_cols) < 10:
        legend_names = []
        for name, kw_col in kw_cols.items():
            cmap = mpl.cm.get_cmap(kw_col["cmap"])
            marker = kw_col["marker"]
            kw = dict(marker=marker, color="w", markersize=20, alpha=0.8)
            label = name.replace("_", r"\_")
            if use_monospace_font:
                label = r"\texttt{" + label + r"}"
            circle = Line2D([0], [0], label=label, markerfacecolor=cmap(0.75), **kw)
            legend_names.append(circle)
        return legend_names

    # else plot only the first of each group

    legend_names = []
    names_already_plotted = set()
    for name, kw_col in kw_cols.items():
        part = split_name_into_first_part(name)

        # exclude if subgroup is already plotted
        if not part in names_already_plotted:
            cmap = mpl.cm.get_cmap(kw_col["cmap"])
            marker = kw_col["marker"]
            kw = dict(marker=marker, color="w", markersize=20, alpha=0.8)
            label = f"{part}..."
            if use_monospace_font:
                label = r"\texttt{" + label + r"}"
            circle = Line2D([0], [0], label=label, markerfacecolor=cmap(0.75), **kw)
            legend_names.append(circle)
            names_already_plotted.add(part)

    return legend_names


def make_custom_legend(zs, ax, vmin, vmax, func, kw_cols):

    log10 = np.log10(zs)
    log_max = int(log10.max()) + 1  # round up
    log_min = int(log10.min()) + 1  # round up

    if log_max - log_min > 1:
        z = 10 ** np.arange(log_min, log_max)
    else:
        z = 10 ** np.array([log_max - 1, log_max])

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

    legend_names = _get_legend_handles_names(kw_cols)

    return (handle, label), legend_names


def name_to_fontsize(name):
    if len(name) > 60:
        fontsize = 14
    elif len(name) > 50:
        fontsize = 16
    elif len(name) > 40:
        fontsize = 19
    elif len(name) > 30:
        fontsize = 23
    elif len(name) > 20:
        fontsize = 29
    else:
        fontsize = 35
    return fontsize


def set_custom_legends(zs, ax, vmin, vmax, func, kw_cols):

    legend_N_alignments, legend_names = make_custom_legend(
        zs,
        ax,
        vmin,
        vmax,
        func,
        kw_cols,
    )

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
    name = max(list(kw_cols.keys()))

    fontsize = name_to_fontsize(name)
    kw_leg_names = dict(
        loc="upper left",
        bbox_to_anchor=(-0.03, 0.999),
        fontsize=fontsize,
    )
    plt.legend(handles=legend_names, **kw_leg_names)


#%%


def minmax(x):
    return np.min(x), np.max(x)


def get_z_min_max(all_fit_results):
    z_minmax = np.array([minmax(df["N_alignments"]) for df in all_fit_results.values()])
    zmin = z_minmax[:, 0].min()
    zmax = z_minmax[:, 1].max()
    return zmin, zmax


# xlim = (-3, 18)
# ylim = (0, 1)


def find_fit_results_limits(all_fit_results):

    n_sigma_min = np.inf
    n_sigma_max = -np.inf

    D_max_min = np.inf
    D_max_max = -np.inf

    for df_res in all_fit_results.values():
        n_sigma = df_res["n_sigma"]
        D_max = df_res["D_max"]

        if min(n_sigma) < n_sigma_min:
            n_sigma_min = min(n_sigma)
        if max(n_sigma) > n_sigma_max:
            n_sigma_max = max(n_sigma)

        if min(D_max) < D_max_min:
            D_max_min = min(D_max)
        if max(D_max) > D_max_max:
            D_max_max = max(D_max)

    n_sigma_lim = (n_sigma_min - 0.5, n_sigma_max + 0.5)
    D_max_lim = (max(D_max_min, 0), min(D_max_max + 0.2, 1))
    return n_sigma_lim, D_max_lim


def split_name_into_first_part(name):
    return re.findall(r"\w+", name)[0]


def get_groups(all_fit_results):
    """ Split """
    filenames = list(all_fit_results.keys())
    groups = defaultdict(list)
    for filename in filenames:
        # get first part of filename (split in non-alphanumerics)
        part = split_name_into_first_part(filename)
        groups[part].append(filename)

    return list(groups.values())


def get_cmaps_and_colors(all_fit_results):

    cmaps = [
        "Blues",
        "Reds",
        "Greens",
        "Purples",
        "Oranges",
        "Greys",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
    ]

    markers = ["o", "s", "*", "D", "P", "X", "v", "^", "<", ">"]
    groups = get_groups(all_fit_results)
    kw = {}
    for i_group, group in enumerate(groups):
        for j_name, name in enumerate(group):
            # cmap, marker = markers_cmaps[i]
            kw[name] = {"marker": markers[j_name], "cmap": cmaps[i_group]}
    return kw


def plot_fit_results_single_N_aligment(
    all_fit_results, cfg, N_alignments_min=0, n_sigma_lim=(-3, 20), D_max_lim=(0, 1)
):

    # cmaps = {name: cmap for name, cmap in zip(all_fit_results.keys(), cmaps_list)}
    kw_marker_colors = get_cmaps_and_colors(all_fit_results)

    zmin, zmax = get_z_min_max(all_fit_results)
    vmin, vmax, func = 10, 600, np.sqrt

    zs = np.array([], dtype=int)
    kw_cols = {}

    fig, ax = plt.subplots(figsize=(10, 10))

    for name, df_res in all_fit_results.items():
        # break
        x = df_res["n_sigma"].values
        y = df_res["D_max"].values
        z = df_res["N_alignments"].values

        if N_alignments_min > 0:
            mask = z > N_alignments_min

            # if not values to plot, continue without plotting
            if mask.sum() == 0:
                continue

            x = x[mask]
            y = y[mask]
            z = z[mask]
        zs = np.append(zs, z)

        s = transform(
            z, vmin=vmin, vmax=vmax, func=func, xmin=func(zmin), xmax=func(zmax)
        )
        c = np.log10(z)

        kw_cols[name] = dict(
            vmin=c.min() / 10,
            vmax=c.max() * 1.25,
            ec=None,
            **kw_marker_colors[name],
        )
        ax.scatter(x, y, s=s, c=c, **kw_cols[name], alpha=0.5)

    # if not plotting anything at all, quit
    if len(zs) == 0:
        return None

    xlabel = r"$n_\sigma$"
    ylabel = r"$D_\mathrm{max}$"
    ax.set(xlim=n_sigma_lim, ylim=D_max_lim, xlabel=xlabel, ylabel=ylabel)

    set_custom_legends(zs, ax, vmin, vmax, func, kw_cols)

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

    return fig


def plot_fit_results(all_fit_results, cfg, N_alignments_mins=[-1]):

    if not isinstance(N_alignments_mins, (list, tuple)):
        N_alignments_mins = [N_alignments_mins]

    if not 0 in N_alignments_mins:
        N_alignments_mins = [0] + N_alignments_mins

    filename = cfg.filename_plot_fit_results

    if utils.is_pdf_valid(filename, cfg.force_plots, N_pages=len(N_alignments_mins)):
        logger.info(f"Plot of fit results already exist.")
        return None

    set_style()

    logger.info(f"Plotting fit results.")

    n_sigma_lim, D_max_lim = find_fit_results_limits(all_fit_results)

    utils.init_parent_folder(filename)
    with PdfPages(filename) as pdf:
        for N_alignments_min in N_alignments_mins:
            # break

            fig = plot_fit_results_single_N_aligment(
                all_fit_results,
                cfg,
                N_alignments_min=N_alignments_min,
                n_sigma_lim=n_sigma_lim,
                D_max_lim=D_max_lim,
            )
            if fig:
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
            # pdf.savefig(fig)

        d = pdf.infodict()
        d["Title"] = "Error Rate Distributions"
        d["Author"] = "Christian Michelsen"
        d["CreationDate"] = datetime.datetime.today()


# %%
