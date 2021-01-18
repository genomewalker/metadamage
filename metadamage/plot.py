import numpy as np
from tqdm.auto import tqdm
import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import EngFormatter
from matplotlib.lines import Line2D

from metadamage import utils
from metadamage import fit
from metadamage import fileloader


def set_rc_params(fig_dpi=300):
    plt.rcParams["figure.figsize"] = (16, 10)
    plt.rcParams["figure.dpi"] = fig_dpi
    mpl.rc("axes", edgecolor="k", linewidth=2)


def set_style(style_path=None, fig_dpi=50):
    if style_path is None:
        style_path = utils.find_style_file()

    tqdm.write("\n\n")
    tqdm.write(f"trying to use this stile file: {style_path}")
    tqdm.write("\n\n")

    try:
        plt.style.use(style_path)
    except OSError:
        tqdm.write(f"Could not find Matplotlib style file. Aesthetics might not be optimal.")
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
    dispersion = utils.human_format(fit_result["dispersion_mean"])
    s += r"$\mathrm{dispersion} = " + f"{dispersion}" + r"$" + "\n"
    s += r"$p_\mathrm{succes} = " + f"{fit_result['p_succes_mean']:.3f}" + r"$" + "\n"

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
        max_position = group.pos.max()
    else:
        max_position = cfg.max_position

    group_direction = {}
    group_direction["forward"] = utils.get_forward(group)
    group_direction["reverse"] = utils.get_reverse(group)

    fig, axes = plt.subplots(
        ncols=2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"wspace": 0.14},
    )

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

    fig.suptitle(f"Error rate frequency as a function of position.\nTaxID: {taxid}\n", fontsize=40)
    fig.subplots_adjust(top=0.75)

    return fig


#%%


def seriel_saving_of_error_rates(cfg, df_top_N, filename, d_fits):

    groupby = df_top_N.groupby("taxid", sort=False, observed=True)
    # desc = f"Plotting {utils.human_format(groupby.ngroups)} TaxIDs in seriel"
    desc = utils.string_pad_left_and_right("TaxIDs", left=8)

    utils.init_parent_folder(filename)
    with PdfPages(filename) as pdf:
        with tqdm(groupby, desc=desc, leave=False, dynamic_ncols=True) as it:
            for taxid, group in it:
                it.set_postfix(taxid=taxid)
                fig = plot_single_group(group, cfg, d_fits)
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
            d = pdf.infodict()
            d["Title"] = "Error Rate Distributions"
            d["Author"] = "Christian Michelsen"
            d["CreationDate"] = datetime.datetime.today()


def plot_error_rates(cfg, df, d_fits=None, number_of_fits=None):

    """
    freq_strings: list of frequencies encoded as string. First letter corresponds to reference and last letter to sequence. This way allows the following string:
    'AT' = 'A2T' = 'A.T' = 'A->T' ...
    """

    if number_of_fits is None:
        number_of_fits = cfg.number_of_fits

    filename = f"./figures/error_rates__{cfg.name}__number_of_fits__{number_of_fits}.pdf"
    if utils.is_pdf_valid(filename, cfg.force_plots, N_pages=number_of_fits):
        if cfg.verbose:
            tqdm.write(f"Plot of error rates already exist: {filename}\n")
        return None

    df_top_N = fileloader.get_top_max_fits(df, number_of_fits)
    seriel_saving_of_error_rates(cfg, df_top_N, filename, d_fits)


#%%

xlim = (-3, 18)
ylim = (0, 1)
alpha_plot = 0.1


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
    legend_names = []
    for name, kw_col in kw_cols.items():
        cmap = mpl.cm.get_cmap(kw_col["cmap"])
        kw = dict(marker="o", color="w", markersize=20, alpha=0.8)
        label = name.replace("_", "\_")
        if use_monospace_font:
            label = r"\texttt{" + label + r"}"
        circle = Line2D([0], [0], label=label, markerfacecolor=cmap(0.75), **kw)
        legend_names.append(circle)
    return legend_names


def make_custom_legend(zs, ax, vmin, vmax, func, kw_cols):

    log10 = np.log10(zs)
    log_max = int(log10.max()) + 1  # round up
    log_min = int(log10.min()) + 1  # round up

    if log_min != log_max:
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

    return (handle, label), _get_legend_handles_names(kw_cols)


def set_custom_legends(zs, ax, vmin, vmax, func, kw_cols):

    legend_N_alignments, legend_names = make_custom_legend(zs, ax, vmin, vmax, func, kw_cols)

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
    kw_leg_names = dict(loc="upper left", bbox_to_anchor=(-0.03, 0.999), fontsize=26)
    plt.legend(handles=legend_names, **kw_leg_names)


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

    return fig, ax


def plot_fit_results(all_fit_results, cfg, N_alignments_mins=[-1]):

    if not isinstance(N_alignments_mins, (list, tuple)):
        N_alignments_mins = [N_alignments_mins]

    if not 0 in N_alignments_mins:
        N_alignments_mins = [0] + N_alignments_mins

    filename = f"./figures/all_fit_results__number_of_fits__{cfg.number_of_fits}.pdf"

    if utils.is_pdf_valid(filename, cfg.force_plots, N_pages=len(N_alignments_mins)):
        if cfg.verbose:
            tqdm.write(f"\nPlot of fit results already exist: {filename}")  # flush=True
        return None

    if cfg.verbose:
        tqdm.write(f"\n\nPlotting fit results.")
    utils.init_parent_folder(filename)
    with PdfPages(filename) as pdf:
        for N_alignments_min in N_alignments_mins:
            fig, ax = plot_fit_results_single_N_aligment(all_fit_results, cfg, N_alignments_min)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
            # pdf.savefig(fig)

        d = pdf.infodict()
        d["Title"] = "Error Rate Distributions"
        d["Author"] = "Christian Michelsen"
        d["CreationDate"] = datetime.datetime.today()


# %%
