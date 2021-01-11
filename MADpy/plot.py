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

import subprocess

# from PyPDF2 import PdfFileMerger, PdfFileReader

# import utils
# import fit
# import fileloader

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
    plt.style.use(style_path)
    set_rc_params(fig_dpi=fig_dpi)


def str_round(x, **kwargs):
    try:
        return sigfig.round(str(x), **kwargs)
    except ValueError:
        return str(x)


def fit_results_to_string(fit_result, group):
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
    N_alignments = utils.human_format(group["N_alignments"].iloc[0])
    s += r"$N_\mathrm{alignments} = " + f"{N_alignments}" + r"$" + "\n"

    return s


from matplotlib.patches import Rectangle


def plot_single_group(group, cfg, d_fits=None, figsize=(18, 7)):

    taxid = group["taxid"].iloc[0]

    if d_fits and (taxid in d_fits):
        has_fits = True
        d_fit = d_fits[taxid]
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

        # plot_bayesian_fits(z, y_median, y_hpdi, ax_forward, ax_reverse, color="C2")

        s = fit_results_to_string(fit_result, group)
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


# def plot_bayesian_fits(z, y_median, y_hpdi, ax_forward, ax_reverse, color):

#     hpdi = y_hpdi.copy()
#     hpdi[0, :] = y_median - hpdi[0, :]
#     hpdi[1, :] = hpdi[1, :] - y_median

#     # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html
#     kw = dict(fmt="none", color=color)
#     ax_forward.errorbar(z[z > 0], y_median[z > 0], hpdi[:, z > 0], label=r"Fit (68\% HDPI)", **kw)
#     ax_reverse.errorbar(z[z < 0], y_median[z < 0], hpdi[:, z < 0], **kw)

# ax_forward.plot(z[z > 0], mean[z > 0], color=color)
# ax_reverse.plot(z[z < 0], mean[z < 0], color=color)

# kwargs = dict(alpha=0.1, color=color, label="Fit (68\% HDPI)")
# ax_forward.fill_between(z[z > 0], hpdi[0, z > 0], hpdi[1, z > 0], **kwargs)
# ax_reverse.fill_between(z[z < 0], hpdi[0, z < 0], hpdi[1, z < 0], **kwargs)


#%%


# def filename_to_tmp_file(filename):
#     return filename.replace("figures/", "figures/tmp/").replace(".pdf", "")


# # useful function for parallel saving of files
# def plot_and_save_single_group(i_group_group, filename, **kwargs):
#     i_group, group = i_group_group

#     try:
#         fig = plot_single_group(group, **kwargs)
#         filename = filename_to_tmp_file(filename) + f"__{i_group:06d}.pdf"
#         utils.init_parent_folder(filename)
#         fig.savefig(filename)
#         return None

#     except:
#         taxid = group.taxid.iloc[0]
#         return taxid


# def parallel_saving_of_error_rates(cfg, df_top_N, filename, d_fits):

#     groupby = df_top_N.groupby("taxid", sort=False, observed=True)

#     func = partial(
#         plot_and_save_single_group,
#         cfg=cfg,
#         d_fits=d_fits,
#         filename=filename,
#     )

#     generator = ((i_group, group) for i_group, (name, group) in enumerate(groupby))

#     desc = f"Plotting {utils.human_format(groupby.ngroups)} TaxIDs in parallel"
#     errors = p_umap(func, generator, length=groupby.ngroups, desc=desc)
#     errors = set([error for error in errors if error])
#     if len(errors) >= 1:
#         print(f"Got errors at TaxIDs: {errors}")

#     # Call the PdfFileMerger
#     mergedObject = PdfFileMerger()

#     # Loop through all of them and append their pages
#     pdfs = sorted(Path(".").rglob(f"{filename_to_tmp_file(filename)}*.pdf"))
#     for pdf in pdfs:
#         mergedObject.append(PdfFileReader(str(pdf), "rb"))

#     # Write all the files into a file which is named as shown below

#     filename_tmp = filename.replace(".pdf", "_tmp.pdf")

#     mergedObject.write(filename_tmp)

#     # delete temporary files
#     for pdf in pdfs:
#         pdf.unlink()

#     # make the combined pdf smaller by compression using the following command:
#     # ps2pdf filename_big filename_small
#     process = subprocess.run(["ps2pdf", filename_tmp, filename])
#     Path(filename_tmp).unlink()

#     Path("./figures/tmp").rmdir()


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


# # from p_tqdm import p_uimap
# from own_p_tqdm import p_map, p_umap
# from functools import partial


def plot_individual_error_rates(cfg, df, d_fits=None, max_plots=None):

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

    # if cfg.parallel_plots:
    # parallel_saving_of_error_rates(cfg, df, filename, d_fits)
    # else:

    seriel_saving_of_error_rates(cfg, df_top_N, filename, d_fits)


#%%


# def plot_df_results_matrix(cfg, df_results):

#     from hist_scatter import scatter_hist2d
#     from mpl_scatter_density import ScatterDensityArtist

#     from astropy.visualization import LogStretch
#     from astropy.visualization.mpl_normalize import ImageNormalize

#     if not cfg.make_plots:
#         if cfg.verbose:
#             print("Not plotting df_results since 'make_plots' is False")
#         return None

#     filename = f"figures/df_results__{cfg.name}__N_taxids__{cfg.N_taxids}.pdf"

#     if utils.file_exists(filename, cfg.force_plots):
#         if cfg.verbose:
#             print(f"Plots of df_results, {filename}", flush=True)
#         return None

#     df_results = df_results.copy()
#     df_results["index"] = np.arange(len(df_results))
#     for col in ["w_AIC", "P_LR"]:
#         df_results[col] = np.log10(df_results[col] + 1e-10)

#     cols = df_results.columns
#     N_cols = len(cols)

#     latex_translation = {
#         r"D_max": r"D_\mathrm{max}",
#         r"w_AIC": r"w_\mathrm{AIC}",
#         r"P_LR": r"P_\mathrm{LR}",
#         r"index": r"\mathrm{index}",
#     }

#     # norm = ImageNormalize(vmin=0.0, vmax=10, stretch=LogStretch())

#     fig, axes = plt.subplots(ncols=N_cols, nrows=N_cols, figsize=(20, 20))

#     for i, col1 in enumerate(cols):
#         for j, col2 in enumerate(cols):

#             # i = 1
#             # j = 0

#             # col1 = cols[i]
#             # col2 = cols[j]

#             print(i, j)

#             ax = axes[i, j]
#             xlim = utils.get_percentile_as_lim(df_results[col2], percentile_max=99)
#             ylim = utils.get_percentile_as_lim(df_results[col1], percentile_max=99)

#             if i == j:  # if diagonal
#                 ax.hist(df_results[col1], 50, range=xlim, histtype="step", lw=3)
#                 ax.set(xlim=xlim)

#             elif i < j:  # upper diagonal

#                 scatter_hist2d(
#                     df_results[col2].values,
#                     df_results[col1].values,
#                     s=20,
#                     marker="o",
#                     mode="mountain",
#                     bins=30,
#                     range=(xlim, ylim),
#                     edgecolors="none",
#                     ax=ax,
#                 )
#                 ax.set(xlim=xlim, ylim=ylim)

#             else:  # lower diagonal

#                 a = ScatterDensityArtist(
#                     ax,
#                     df_results[col2],
#                     df_results[col1],
#                     dpi=10,
#                     color="black",
#                     # norm=norm,
#                 )
#                 ax.add_artist(a)
#                 ax.set(xlim=xlim, ylim=ylim)

#             if j == 0:
#                 ax.set(ylabel=f"${latex_translation[col1]}$")
#             if i == N_cols - 1:
#                 ax.set(xlabel=f"${latex_translation[col2]}$")

#             # display(fig)

#     fig.savefig(filename)


# #%%

# colorbar = False
# xlim = (-3, 18)
# ylim = (0, 1)
# alpha_plot = 0.1
# alpha_hist = 0.1


def _plot_fit_results(
    all_fit_results, ax, colorbar=False, xlim=(-3, 18), ylim=(0, 1), alpha_plot=0.1, alpha_hist=0.0,
):

    # from matplotlib.ticker import LogFormatter

    # vmax = [len(df_res) for df_res in all_fit_results.values()]

    for i, (name, df_res) in enumerate(all_fit_results.items()):

        x = df_res["n_sigma"].values
        y = df_res["D_max"].values
        c = f"C{i}"

        # norm = ImageNormalize(vmin=0.0, vmax=vmax, stretch=LogStretch())
        if alpha_hist > 0:

            from mpl_scatter_density import ScatterDensityArtist
            from astropy.visualization import LogStretch
            from astropy.visualization.mpl_normalize import ImageNormalize

            kwargs = dict(dpi=10, color=c, alpha=alpha_hist)  # norm=norm
            a = ScatterDensityArtist(ax, x, y, **kwargs)
            ax.add_artist(a)

        if alpha_plot > 0:
            ax.plot(x, y, ".", color=c, alpha=alpha_plot)  # label=name
            # add fake points for legend
            ax.plot(np.NaN, np.NaN, ".", color=c, label=name)

        if colorbar:
            fig = plt.gcf()
            cb = fig.colorbar(a)  # shrink=0.8, pad=0.1, label="Signal"
            cb.ax.set_title(name, fontsize=40, pad=20)

            # if log_normalization:
            #     cb.formatter = LogFormatter()  # 10 is the default
            #     cb.set_ticks(10 ** np.arange(np.ceil(np.log10(cb.vmax))))
            #     cb.update_ticks()

    # ax.axis("equal")

    ax.legend(markerscale=2)
    # legend = ax.legend(markerscale=2)
    # for l in legend.get_lines():
    #     l.set_alpha(1)
    #     l.set_markersize(50)

    ax.set(
        xlim=xlim, ylim=ylim, xlabel=r"$n_\sigma$", ylabel=r"$D_\mathrm{max}$",
    )

    # return ax


def plot_fit_results(all_fit_results, cfg, savefig=True):

    if len(all_fit_results) >= 2 and cfg.make_plots:
        fig, ax = plt.subplots(figsize=(10, 10))
        _plot_fit_results(
            all_fit_results, ax, xlim=(-3, 18), ylim=(0, 0.7), alpha_plot=0.1, alpha_hist=0.0,
        )
        if savefig:
            fig.savefig(f"./figures/all_fit_results__N_taxids__{cfg.N_taxids}.pdf")
    else:
        print("No fits to plot")
