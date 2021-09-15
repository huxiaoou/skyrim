import os
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd
from typing import Union

matplotlib.style.use("Solarize_Light2")


def check_and_mkdir(t_path):
    if not os.path.exists(t_path):
        os.mkdir(t_path)
        return 1
    else:
        return 0


def remove_files_in_the_dir(t_path):
    for f in os.listdir(t_path):
        os.remove(os.path.join(t_path, f))
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


def plot_lines(t_plot_df: pd.DataFrame, t_fig_name: str, t_save_dir: str, t_line_width: float = 2, t_colormap: Union[None, str] = None,
               t_xtick_count: int = 10, t_xlabel: str = "", t_legend_loc="upper left", t_tick_label_size: int = 12,
               t_ax_title: str = "", t_save_type: str = "pdf"
               ):
    """
    :param t_plot_df: a pd.DataFrame with columns to be plot as lines, and string-like index. The typical scenario this function
                      is called is to plot NAV(Net Assets Value) curve(s).
    :param t_fig_name: the name of the file to save the figure
    :param t_save_dir: the directory where the file is saved
    :param t_line_width: line width
    :param t_colormap: colormap to be used to change the line color, default is None, frequently used values are:
                       ["jet", "Paired", "RdBu", "spring", "summer", "autumn", "winter"]
    :param t_xtick_count: the number of ticks to be labeled on x-axis
    :param t_xlabel: the labels to be print on xticks
    :param t_legend_loc: the location of legend, frequently used values are:
                         ["best", "upper left", "upper right"]
    :param t_tick_label_size: the size of the tick labels
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
    ax0.legend(loc=t_legend_loc)
    ax0.tick_params(axis="both", labelsize=t_tick_label_size)
    ax0.set_title(t_ax_title)
    fig0_name = t_fig_name + "." + t_save_type
    fig0_path = os.path.join(t_save_dir, fig0_name)
    fig0.savefig(fig0_path, bbox_inches="tight")
    plt.close(fig0)
    return 0
