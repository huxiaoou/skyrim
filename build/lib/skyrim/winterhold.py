import os
import re
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.style
import seaborn as sns
import shutil
import platform
from typing import Union

matplotlib.use("Agg")  # to fix some complicated bugs which lead to IDE break down when Debug mode is activated.

this_platform = platform.system().upper()
if this_platform == "WINDOWS":
    # to use chinese code
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 设置正负号


def check_and_mkdir(t_path: str):
    if not os.path.exists(t_path):
        os.mkdir(t_path)
        return 1
    else:
        return 0


def remove_files_in_the_dir(t_path: str):
    for f in os.listdir(t_path):
        os.remove(os.path.join(t_path, f))
    return 0


def check_and_remove_tree(t_path: str):
    if os.path.exists(t_path):
        shutil.rmtree(t_path)
    return 0


def get_mix_string_len(t_mix_string: str, t_expected_len: int):
    """

    :param t_mix_string: example "食品ETF09"
    :param t_expected_len: length of expected output string
    :return: "{:ks}".format(t_mix_string) would occupy t_expected_len characters when print,
             which will make t_mix_string aligned with pure English string
    """
    # chs_string = re.sub("[0-9a-zA-Z]", "", t_mix_string)
    chs_string = re.sub("[\\da-zA-Z]", "", t_mix_string)
    chs_string_len = len(chs_string)
    k = max(t_expected_len - chs_string_len, len(t_mix_string) + chs_string_len)
    return k


def timer(func):
    def inner(**kwargs):
        t1 = dt.datetime.now()
        res = func(**kwargs)
        t2 = dt.datetime.now()
        print("... bgn @ {0} for {1}".format(t1, func.__name__))
        print("... end @ {0} for {1}".format(t2, func.__name__))
        print("... time consuming: {} seconds".format((t2 - t1).total_seconds()))
        print("\n")
        return res

    return inner


def date_format_converter_08_to_10(t_date: str):
    # "202100101" -> "2021-01-01"
    return t_date[0:4] + "-" + t_date[4:6] + "-" + t_date[6:8]


def date_format_converter_10_to_08(t_date: str):
    # "20210-01-01" -> "20210101"
    return t_date.replace("-", "")


def plot_lines(t_plot_df: pd.DataFrame, t_fig_name: str, t_save_dir: str = ".", t_line_width: float = 2,
               t_fig_size: tuple = (16, 9),
               t_colormap: Union[None, str] = None,
               t_xtick_count: int = 10, t_xlabel: str = "", t_ylim: tuple = (None, None),
               t_legend_loc="upper left", t_legend_fontsize: int = 12,
               t_tick_label_size: int = 12, t_tick_label_rotation: int = 0,
               t_vlines_index: list = None,
               t_hlines_value: list = None,
               t_style: str = "Solarize_Light2",
               t_ax_title: str = "", t_save_type: str = "pdf"
               ):
    """
    :param t_plot_df: a pd.DataFrame with columns to be plotted as lines, and string-like index. The typical scenario this function
                      is called is to plot NAV(Net Assets Value) curve(s).
    :param t_fig_name: the name of the file to save the figure
    :param t_save_dir: the directory where the file is saved
    :param t_line_width: line width
    :param t_fig_size: ratio of figure's length to width
    :param t_colormap: colormap to be used to change the line color, default is None, frequently used values are:
                       ["jet", "Paired", "RdBu", "spring", "summer", "autumn", "winter"]
    :param t_xtick_count: the number of ticks to be labeled on x-axis
    :param t_xlabel: the labels to be print on xticks
    :param t_ylim: plot limit for y-axis, default is (None, None), which means use limit automatically chose by Matplotlib
    :param t_legend_loc: the location of legend, frequently used values are:
                         ["best", "upper left", "upper right"]
    :param t_legend_fontsize:
    :param t_tick_label_size: the size of the tick labels
    :param t_tick_label_rotation: the rotation of the tick labels, 0 = norm, 90 = fonts are rotated 90 degree counter-clockwise
    :param t_vlines_index: a list of indexes of vertical lines
    :param t_hlines_value: a list of values of vertical lines
    :param t_style:
    :param t_ax_title: the title of the ax
    :param t_save_type: the type of file, frequently used values are:
                        ["pdf", "jpg"]
    :return:
    """
    plt.style.use(t_style)

    fig0, ax0 = plt.subplots(figsize=t_fig_size)
    t_plot_df.plot(ax=ax0, lw=t_line_width, colormap=t_colormap)
    xticks = np.arange(0, len(t_plot_df), max(int(len(t_plot_df) / t_xtick_count), 1))
    xticklabels = t_plot_df.index[xticks]
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels)
    ax0.set_xlabel(t_xlabel)
    ax0.set_ylim(t_ylim)
    if t_vlines_index:
        ax0.vlines(
            [t_plot_df.index.get_loc(z) for z in t_vlines_index],
            ymin=ax0.get_ylim()[0], ymax=ax0.get_ylim()[1],
            colors="r", linestyles="dashed"
        )
    if t_hlines_value:
        ax0.hlines(
            t_hlines_value,
            xmin=ax0.get_xlim()[0], xmax=ax0.get_xlim()[1],
            colors="g", linestyles="dashed"
        )

    ax0.legend(loc=t_legend_loc, fontsize=t_legend_fontsize)
    ax0.tick_params(axis="both", labelsize=t_tick_label_size, rotation=t_tick_label_rotation)
    ax0.set_title(t_ax_title)
    fig0_name = t_fig_name + "." + t_save_type
    fig0_path = os.path.join(t_save_dir, fig0_name)
    fig0.savefig(fig0_path, bbox_inches="tight")
    plt.close(fig0)
    return 0


def plot_corr(t_corr_df: pd.DataFrame, t_fig_name: str, t_save_dir: str = ".",
              t_fig_size: tuple = (16, 9),
              t_tick_label_rotation: int = 0,
              t_annot_size: int = 8, t_annot_format: str = ".2f", t_save_type: str = "pdf",
              t_style: str = "Solarize_Light2",
              ):
    plt.style.use(t_style)
    fig0, ax0 = plt.subplots(figsize=t_fig_size)
    sns.heatmap(t_corr_df, cmap="Blues", annot=True, fmt=t_annot_format, annot_kws={"size": t_annot_size})
    ax0.tick_params(axis="y", rotation=t_tick_label_rotation)
    fig0_name = t_fig_name + "." + t_save_type
    fig0_path = os.path.join(t_save_dir, fig0_name)
    fig0.savefig(fig0_path, bbox_inches="tight")
    plt.close(fig0)
    return 0


def plot_bar(t_bar_df: pd.DataFrame, t_stacked: bool, t_fig_name: str, t_save_dir: str = ".",
             t_fig_size: tuple = (16, 9),
             t_colormap: Union[None, str] = "jet",
             t_xtick_span: int = 1, t_xlabel: str = "", t_ylim: tuple = (None, None), t_legend_loc="upper left",
             t_tick_label_size: int = 12, t_tick_label_rotation: int = 0,
             t_ax_title: str = "", t_save_type: str = "pdf",
             t_style: str = "Solarize_Light2",
             ):
    """

    :param t_bar_df: if not transposed, each row(obs) of t_bar_df means a tick in the x-axis,
                     and a cluster the columns(variables) of the row are plot at the tick.
    :param t_stacked: whether bar plot should be stacked, most frequently used sense:
                      True: usually for weight plot
                      False: best designed for small obs, i.e., all the row labels can be plotted
    :param t_fig_name:
    :param t_save_dir:
    :param t_fig_size:
    :param t_colormap:
    :param t_xtick_span:
    :param t_xlabel:
    :param t_ylim:
    :param t_legend_loc:
    :param t_tick_label_size:
    :param t_tick_label_rotation:
    :param t_ax_title:
    :param t_save_type:
    :param t_style:
    :return:
    """
    plt.style.use(t_style)
    fig0, ax0 = plt.subplots(figsize=t_fig_size)
    t_bar_df.plot(ax=ax0, kind="bar", stacked=t_stacked, colormap=t_colormap)
    n_ticks = len(t_bar_df)
    xticks = np.arange(0, n_ticks, t_xtick_span)
    xticklabels = t_bar_df.index[xticks]
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels)
    ax0.set_xlabel(t_xlabel)
    ax0.set_ylim(t_ylim)
    ax0.legend(loc=t_legend_loc)
    ax0.tick_params(axis="both", labelsize=t_tick_label_size, rotation=t_tick_label_rotation)
    ax0.set_title(t_ax_title)
    fig0_name = t_fig_name + "." + t_save_type
    fig0_path = os.path.join(t_save_dir, fig0_name)
    fig0.savefig(fig0_path, bbox_inches="tight")
    plt.close(fig0)
    return 0


def plot_twinx(t_plot_df: pd.DataFrame, t_primary_cols: list, t_secondary_cols: list,
               t_primary_kind: str, t_secondary_kind: str,
               t_fig_name: str, t_save_dir: str = ".",
               t_fig_size: tuple = (16, 9),
               t_line_width: float = 2,
               t_primary_colormap: Union[None, str] = "jet", t_secondary_colormap: Union[None, str] = "jet",
               t_primary_style="-", t_secondary_style="-.",
               t_primary_alpha: float = 1.0, t_secondary_alpha: float = 1.0,
               t_primary_hlines: tuple = (), t_secondary_hlines: tuple = (),
               t_primary_ylim: tuple = (None, None), t_secondary_ylim: tuple = (None, None),
               t_legend_loc: str = "upper left",
               t_xtick_span: int = 1, t_xlabel: str = "",
               t_tick_label_size: int = 12, t_tick_label_rotation: int = 0,
               t_ax_title: str = "", t_save_type: str = "pdf",
               t_style: str = "Solarize_Light2",
               ):
    """

    :param t_plot_df:
    :param t_primary_cols: columns to be plotted at primary(left-y) axis
    :param t_secondary_cols: columns to be plotted at secondary(right-y) axis
    :param t_primary_kind: plot kind for main(left-y) axis, available options = ["bar", "barh", "line"]
    :param t_secondary_kind: plot kind for secondary(right-y) axis, available options = ["bar", "barh", "line"]
    :param t_fig_name:
    :param t_save_dir:
    :param t_fig_size:
    :param t_line_width:
    :param t_primary_colormap:
    :param t_secondary_colormap:
    :param t_primary_style:
    :param t_secondary_style:
    :param t_primary_alpha:
    :param t_secondary_alpha:
    :param t_primary_ylim:
    :param t_secondary_ylim:
    :param t_primary_hlines:
    :param t_secondary_hlines:
    :param t_legend_loc:
    :param t_xtick_span:
    :param t_xlabel:
    :param t_tick_label_size:
    :param t_tick_label_rotation:
    :param t_ax_title:
    :param t_save_type:
    :param t_style:
    :return:
    """
    plt.style.use(t_style)
    fig0, ax0 = plt.subplots(figsize=t_fig_size)
    ax1 = ax0.twinx()
    t_plot_df[t_primary_cols].plot(ax=ax0, kind=t_primary_kind, colormap=t_primary_colormap,
                                   lw=t_line_width, style=t_primary_style, alpha=t_primary_alpha, legend=None)
    t_plot_df[t_secondary_cols].plot(ax=ax1, kind=t_secondary_kind, colormap=t_secondary_colormap,
                                     lw=t_line_width, style=t_secondary_style, alpha=t_secondary_alpha)
    ax0.hlines(t_primary_hlines, xmin=0, xmax=len(t_plot_df))
    ax1.hlines(t_secondary_hlines, xmin=0, xmax=len(t_plot_df))

    # merge legends
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines0 + lines1, labels0 + labels1, loc=t_legend_loc)

    xticks = np.arange(0, len(t_plot_df), t_xtick_span)
    xticklabels = t_plot_df.index[xticks]
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels)
    ax0.set_xlabel(t_xlabel)
    ax0.set_ylim(t_primary_ylim)
    ax1.set_ylim(t_secondary_ylim)
    ax0.tick_params(axis="both", labelsize=t_tick_label_size, rotation=t_tick_label_rotation)
    ax0.set_title(t_ax_title)
    fig0_name = t_fig_name + "." + t_save_type
    fig0_path = os.path.join(t_save_dir, fig0_name)
    fig0.savefig(fig0_path, bbox_inches="tight")
    plt.close(fig0)

    return 0


def plot_single_line_piecewise(t_ax: matplotlib.axes.Axes, t_plot_df: pd.DataFrame,
                               t_line_val_label: str, t_line_tag_label: str,
                               t_line_style: str, t_piecewise_color_map: dict, t_line_width: float,
                               t_legend_loc: str,
                               t_style: str = "Solarize_Light2",
                               ):
    plt.style.use(t_style)
    ticks_n = len(t_plot_df)
    line_val_srs = t_plot_df[t_line_val_label]
    line_tag_srs = t_plot_df[t_line_tag_label]

    t_ax.plot(np.arange(ticks_n), line_val_srs,
              color="k", ls=t_line_style, lw=t_line_width, alpha=0.5,
              label=t_line_val_label)
    t_ax.legend(loc=t_legend_loc)

    piece_iloc_bgn = 0
    while piece_iloc_bgn <= (ticks_n - 2):
        i = 0
        piece_tag = line_tag_srs[piece_iloc_bgn + 1]
        while True:
            i += 1
            piece_iloc_stp = piece_iloc_bgn + i
            if piece_iloc_stp >= ticks_n:
                break
            if line_tag_srs[piece_iloc_stp] != piece_tag:
                break

        piece_x = range(piece_iloc_bgn, piece_iloc_stp)
        piece_y = line_val_srs.iloc[piece_iloc_bgn:piece_iloc_stp]
        t_ax.plot(piece_x, piece_y,
                  color=t_piecewise_color_map[piece_tag],
                  ls=t_line_style, lw=t_line_width)

        # sub_plot_df = t_plot_df[[t_line_val_label, t_line_tag_label]].iloc[piece_iloc_bgn:piece_iloc_stp]
        # print("-" * 12)
        # print(sub_plot_df)

        # for next loop
        piece_iloc_bgn = piece_iloc_stp - 1

    return 0


def plot_lines_piecewise(t_plot_df: pd.DataFrame,
                         t_piecewise_lines_list: list,
                         t_piecewise_color_map: dict,
                         t_fig_size: tuple = (16, 9),
                         t_xtick_count: int = 10, t_xlabel: str = "", t_ylim: tuple = (None, None),
                         t_tick_label_size: int = 12, t_tick_label_rotation: int = 0, t_ax_title: str = "",
                         t_legend_loc: str = "upper left",
                         t_fig_name: str = "test_example", t_save_dir: str = ".", t_save_format: str = "pdf",
                         t_style: str = "Solarize_Light2",
                         ):
    """

    :param t_plot_df: date string as index
    :param t_piecewise_lines_list: each element of this list has format =("line_val_label", "line_tag_lag", "line_style", "line_width")

                                   LINESTYLES
                                   Simple linestyles can be defined using the strings "solid", "dotted", "dashed" or "dashdot". More
                                   refined control can be achieved by providing a dash tuple (offset, (on_off_seq)).
                                   For example, (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt space) with no offset,
                                   while (5, (10, 3)), means (10pt line, 3pt space), but skip the first 5pt line.

                                   Available Options
                                   linestyle_str = [
                                         ('solid', 'solid'),      # Same as (0, ()) or '-'
                                         ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
                                         ('dashed', 'dashed'),    # Same as '--'
                                         ('dashdot', 'dashdot')]  # Same as '-.'

                                   linestyle_tuple = [
                                         ('loosely dotted',        (0, (1, 10))),
                                         ('dotted',                (0, (1, 1))),
                                         ('densely dotted',        (0, (1, 1))),
                                         ('long dash with offset', (5, (10, 3))),
                                         ('loosely dashed',        (0, (5, 10))),
                                         ('dashed',                (0, (5, 5))),
                                         ('densely dashed',        (0, (5, 1))),

                                         ('loosely dashdotted',    (0, (3, 10, 1, 10))),
                                         ('dashdotted',            (0, (3, 5, 1, 5))),
                                         ('densely dashdotted',    (0, (3, 1, 1, 1))),

                                         ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                                         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                                         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    :param t_piecewise_color_map: a dictionary describe line style for each segment.
                                     and unique values of t_piecewise_tag columns must
                                     be a subset of the keys of this dict.
    :param t_fig_name:
    :param t_fig_size: inches
    :param t_xtick_count:
    :param t_xlabel:
    :param t_ylim:
    :param t_tick_label_size:
    :param t_tick_label_rotation:
    :param t_ax_title: the title of the ax
    :param t_legend_loc: location of legend
    :param t_save_dir:
    :param t_save_format:
    :param t_style:

    :return:
    """

    plt.style.use(t_style)
    fig0, ax0 = plt.subplots(figsize=t_fig_size)

    for (line_val_label, line_tag_label, line_style, line_width) in t_piecewise_lines_list:
        plot_single_line_piecewise(
            t_ax=ax0, t_plot_df=t_plot_df,
            t_line_val_label=line_val_label,
            t_line_tag_label=line_tag_label,
            t_line_style=line_style,
            t_piecewise_color_map=t_piecewise_color_map,
            t_line_width=line_width,
            t_legend_loc=t_legend_loc,
        )

    # shared axis settings
    xticks = np.arange(0, len(t_plot_df), max(int(len(t_plot_df) / t_xtick_count), 1))
    xticklabels = t_plot_df.index[xticks]
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels)
    ax0.set_xlabel(t_xlabel)
    ax0.set_ylim(t_ylim)
    ax0.tick_params(axis="both", labelsize=t_tick_label_size, rotation=t_tick_label_rotation)
    ax0.set_title(t_ax_title)

    save_name = t_fig_name + "." + t_save_format
    save_path = os.path.join(t_save_dir, save_name)
    fig0.savefig(save_path)
    plt.close(fig0)

    return 0


if __name__ == "__main__":
    # for mp_style in mps.available:
    #     print(mp_style)
    import scipy.stats as sps

    n = 100
    mu, sd = 0.001, 0.02
    x0 = sps.norm.rvs(size=n, loc=mu, scale=sd)
    x1 = sps.norm.rvs(size=n, loc=mu, scale=sd)
    x2 = sps.norm.rvs(size=n, loc=mu, scale=sd)
    x3 = sps.norm.rvs(size=n, loc=mu, scale=sd)
    z0 = np.cumprod(1 + x0)
    z1 = np.cumprod(1 + x1)
    z2 = np.cumprod(1 + x2)
    z3 = np.cumprod(1 + x3)
    ct0 = [1 if _ > (mu + sd) else (0 if _ > (mu - sd) else -1) for _ in x0]
    ct1 = [1 if _ > (mu + sd) else (0 if _ > (mu - sd) else -1) for _ in x1]
    ct2 = [1 if _ > (mu + 0 * sd) else (0 if _ > (mu - 0 * sd) else -1) for _ in x2]
    ct3 = [1 if _ > (mu + 0 * sd) else (0 if _ > (mu - 0 * sd) else -1) for _ in x3]

    # z = [99, 98, 100]
    # ct = [0, 1, 0]

    df = pd.DataFrame({
        "z0": z0,
        "z1": z1,
        "z2": z2,
        "z3": z3,
        "ct0": ct0,
        "ct1": ct1,
        "ct2": ct2,
        "ct3": ct3,
    }, index=["2022{:04d}".format(_) for _ in range(len(z0))])

    plot_lines_piecewise(
        t_plot_df=df,
        t_piecewise_lines_list=[
            # ("z0", "ct0", (0, (2, 6)), 2),
            # ("z1", "ct1", (0, (4, 4, 8, 8)), 2),
            # ("z2", "ct2", (0, (6, 6, 18, 18)), 2),
            ("z0", "ct0", "-", 2),
            ("z1", "ct1", "-.", 2),
            ("z2", "ct2", ":", 2),
            ("z3", "ct3", "--", 2),
        ],
        t_piecewise_color_map={0: "y", 1: "r", -1: "g"},
        t_save_dir=os.path.join("E:\\", "tmp")
    )

    print("-" * 12)
    print(df)
