import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.style
import seaborn as sns
import shutil
from typing import Union

matplotlib.use("Agg")  # to fix some complicated bugs which lead to IDE break down when Debug mode is activated.
matplotlib.style.use("Solarize_Light2")


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
               t_colormap: Union[None, str] = None,
               t_xtick_count: int = 10, t_xlabel: str = "", t_ylim: tuple = (None, None), t_legend_loc="upper left",
               t_tick_label_size: int = 12, t_tick_label_rotation: int = 0,
               t_vlines_index: list = None,
               t_ax_title: str = "", t_save_type: str = "pdf"
               ):
    """
    :param t_plot_df: a pd.DataFrame with columns to be plotted as lines, and string-like index. The typical scenario this function
                      is called is to plot NAV(Net Assets Value) curve(s).
    :param t_fig_name: the name of the file to save the figure
    :param t_save_dir: the directory where the file is saved
    :param t_line_width: line width
    :param t_colormap: colormap to be used to change the line color, default is None, frequently used values are:
                       ["jet", "Paired", "RdBu", "spring", "summer", "autumn", "winter"]
    :param t_xtick_count: the number of ticks to be labeled on x-axis
    :param t_xlabel: the labels to be print on xticks
    :param t_ylim: plot limit for y-axis, default is (None, None), which means use limit automatically chose by Matplotlib
    :param t_legend_loc: the location of legend, frequently used values are:
                         ["best", "upper left", "upper right"]
    :param t_tick_label_size: the size of the tick labels
    :param t_tick_label_rotation: the rotation of the tick labels, 0 = norm, 90 = fonts are rotated 90 degree counter-clockwise
    :param t_vlines_index: a list of indexes of vertical lines
    :param t_ax_title: the title of the ax
    :param t_save_type: the type of file, frequently used values are:
                        ["pdf", "jpg"]
    :return:
    """
    fig0, ax0 = plt.subplots(figsize=(16, 9))
    t_plot_df.plot(ax=ax0, lw=t_line_width, colormap=t_colormap)
    xticks = np.arange(0, len(t_plot_df), int(len(t_plot_df) / t_xtick_count))
    xticklabels = t_plot_df.index[xticks]
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels)
    ax0.set_xlabel(t_xlabel)
    ax0.set_ylim(t_ylim)
    if t_vlines_index is not None:
        ax0.vlines(
            [t_plot_df.index.get_loc(z) for z in t_vlines_index],
            ymin=ax0.get_ylim()[0], ymax=ax0.get_ylim()[1],
            colors="r", linestyles="dashed"
        )
    ax0.legend(loc=t_legend_loc)
    ax0.tick_params(axis="both", labelsize=t_tick_label_size, rotation=t_tick_label_rotation)
    ax0.set_title(t_ax_title)
    fig0_name = t_fig_name + "." + t_save_type
    fig0_path = os.path.join(t_save_dir, fig0_name)
    fig0.savefig(fig0_path, bbox_inches="tight")
    plt.close(fig0)
    return 0


def plot_corr(t_corr_df: pd.DataFrame, t_fig_name: str, t_save_dir: str = ".",
              t_tick_label_rotation: int = 0,
              t_annot_size: int = 8, t_annot_format: str = ".2f", t_save_type: str = "pdf"):
    fig0, ax0 = plt.subplots(figsize=(16, 9))
    sns.heatmap(t_corr_df, cmap="Blues", annot=True, fmt=t_annot_format, annot_kws={"size": t_annot_size})
    ax0.tick_params(axis="y", rotation=t_tick_label_rotation)
    fig0_name = t_fig_name + "." + t_save_type
    fig0_path = os.path.join(t_save_dir, fig0_name)
    fig0.savefig(fig0_path, bbox_inches="tight")
    plt.close(fig0)
    return 0


def plot_bar(t_bar_df: pd.DataFrame, t_stacked: bool, t_fig_name: str, t_save_dir: str = ".",
             t_colormap: Union[None, str] = "jet",
             t_xtick_span: int = 1, t_xlabel: str = "", t_ylim: tuple = (None, None), t_legend_loc="upper left",
             t_tick_label_size: int = 12, t_tick_label_rotation: int = 0,
             t_ax_title: str = "", t_save_type: str = "pdf"):
    """

    :param t_bar_df: if not transposed, each row(obs) of t_bar_df means a tick in the x-axis,
                     and a cluster the columns(variables) of the row are plot at the tick.
    :param t_stacked: whether bar plot should be stacked, most frequently used sense:
                      True: usually for weight plot
                      False: best designed for small obs, i.e., all the row labels can be plotted
    :param t_fig_name:
    :param t_save_dir:
    :param t_colormap:
    :param t_xtick_span:
    :param t_xlabel:
    :param t_ylim:
    :param t_legend_loc:
    :param t_tick_label_size:
    :param t_tick_label_rotation:
    :param t_ax_title:
    :param t_save_type:
    :return:
    """
    fig0, ax0 = plt.subplots(figsize=(16, 9))
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
               t_line_width: float = 2,
               t_primary_colormap: Union[None, str] = "jet", t_secondary_colormap: Union[None, str] = "jet",
               t_primary_style="-", t_secondary_style="-.",
               t_primary_ylim: tuple = (None, None), t_secondary_ylim: tuple = (None, None),
               t_legend_loc: str = "upper left",
               t_xtick_span: int = 1, t_xlabel: str = "",
               t_tick_label_size: int = 12, t_tick_label_rotation: int = 0,
               t_ax_title: str = "", t_save_type: str = "pdf"):
    """

    :param t_plot_df:
    :param t_primary_cols: columns to be plotted at primary(left-y) axis
    :param t_secondary_cols: columns to be plotted at secondary(right-y) axis
    :param t_primary_kind: plot kind for main(left-y) axis, available options = ["bar", "barh", "line"]
    :param t_secondary_kind: plot kind for secondary(right-y) axis, available options = ["bar", "barh", "line"]
    :param t_fig_name:
    :param t_save_dir:
    :param t_line_width:
    :param t_primary_colormap:
    :param t_secondary_colormap:
    :param t_primary_style:
    :param t_secondary_style:
    :param t_legend_loc:
    :param t_primary_ylim:
    :param t_secondary_ylim:
    :param t_xtick_span:
    :param t_xlabel:
    :param t_tick_label_size:
    :param t_tick_label_rotation:
    :param t_ax_title:
    :param t_save_type:
    :return:
    """
    fig0, ax0 = plt.subplots(figsize=(16, 9))
    ax1 = ax0.twinx()
    t_plot_df[t_primary_cols].plot(ax=ax0, kind=t_primary_kind, colormap=t_primary_colormap, lw=t_line_width, style=t_primary_style, legend=None)
    t_plot_df[t_secondary_cols].plot(ax=ax1, kind=t_secondary_kind, colormap=t_secondary_colormap, lw=t_line_width, style=t_secondary_style)

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
