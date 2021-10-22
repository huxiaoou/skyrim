import numpy as np
import pandas as pd
import datetime as dt
from skyrim.whiterun import CCalendar

'''
created @ 2021-02-22
0.  define a class to evaluate the performance of some portfolio
1.  this class provide methods to calculate some frequently used index
'''

RETURN_SCALE = 100


class CNAV(object):
    def __init__(self, t_raw_nav_srs: pd.Series, t_annual_rf_rate: float, t_freq: str):
        """

        :param t_raw_nav_srs: A. the Net-Assets-Value series, with datetime-like index in string format.
                                 The first item in this series can not be ONE, and the class will do the conversion
                                 when initialized.
                              B. the index of the series is supposed to be continuous, i.e., there are not any missing
                                 dates or timestamp in the index.
        :param t_annual_rf_rate: annualized risk free rate, must NOT be multiplied by the return scale.
                                 the class will do the conversion when initialized
        :param t_freq: a string to indicate the frequency the series, must be one of ["S", "D", "W", "M", "Q", "Y"]
        """
        self.m_nav_srs: pd.Series = t_raw_nav_srs / t_raw_nav_srs.iloc[0]  # set the first value to be 1
        self.m_rtn_srs: pd.Series = ((t_raw_nav_srs / t_raw_nav_srs.shift(1) - 1) * RETURN_SCALE).fillna(0)  # has the same length as nav srs
        self.m_obs: int = len(t_raw_nav_srs)

        self.m_annual_factor: int = {
            "S": 504,
            "D": 252,
            "W": 52,
            "M": 12,
            "Q": 4,
            "Y": 1,
        }[t_freq]

        self.m_annual_rf_rate: float = t_annual_rf_rate * RETURN_SCALE

        # frequently used performance index
        # primary
        self.m_return_mean: float = 0
        self.m_return_std: float = 0
        self.m_hold_period_return: float = 0
        self.m_annual_return: float = 0
        self.m_sharpe_ratio: float = 0

        # secondary - A max drawdown scale
        self.m_max_drawdown_scale: float = 0  # a non negative float, multiplied by RETURN_SCALE
        self.m_drawdown_scale_srs: pd.Series = pd.Series(data=0.0, index=self.m_nav_srs.index)

        # secondary - B max drawdown duration
        self.m_max_drawdown_duration: int = 0  # a non negative int, stands for the duration of drawdown
        self.m_drawdown_duration_srs: pd.Series = pd.Series(data=0, index=self.m_nav_srs.index)

        # secondary - C max recover duration
        self.m_max_recover_duration: int = 0
        self.m_recover_duration_srs: pd.Series = pd.Series(data=0, index=self.m_nav_srs.index)

    def cal_return_mean(self):
        self.m_return_mean = self.m_rtn_srs.mean()
        return 0

    def cal_return_std(self):
        self.m_return_std = self.m_rtn_srs.std()
        return 0

    def cal_hold_period_return(self):
        self.m_hold_period_return = (self.m_nav_srs.iloc[-1] / self.m_nav_srs.iloc[0] - 1) * RETURN_SCALE
        return 0

    def cal_annual_return(self, t_method: str = "linear"):
        if t_method == "linear":
            self.m_annual_return = self.m_rtn_srs.mean() * self.m_annual_factor
        else:
            self.m_annual_return = (np.power(self.m_hold_period_return / RETURN_SCALE + 1, self.m_annual_factor / len(self.m_rtn_srs)) - 1) * RETURN_SCALE
        return 0

    def cal_sharpe_ratio(self):
        diff_srs = self.m_rtn_srs - self.m_annual_rf_rate / self.m_annual_factor
        mu = diff_srs.mean()
        sd = diff_srs.std()
        self.m_sharpe_ratio = mu / sd * np.sqrt(self.m_annual_factor)
        return 0

    def cal_max_drawdown_scale(self):
        self.m_drawdown_scale_srs = 1 - self.m_nav_srs / self.m_nav_srs.cummax()
        self.m_max_drawdown_scale = self.m_drawdown_scale_srs.max()
        return 0

    def cal_max_drawdown_duration(self):
        prev_high = self.m_nav_srs.iloc[0]
        prev_high_loc = 0
        prev_drawdown_scale = 0.0
        drawdown_loc = 0
        for i, nav_i in enumerate(self.m_nav_srs):
            if nav_i >= prev_high:
                prev_high = nav_i
                prev_high_loc = i
                prev_drawdown_scale = 0
            drawdown_scale = 1 - nav_i / prev_high
            if drawdown_scale > prev_drawdown_scale:
                prev_drawdown_scale = drawdown_scale
                drawdown_loc = i
            self.m_drawdown_duration_srs.iloc[i] = drawdown_loc - prev_high_loc
        self.m_max_drawdown_duration = self.m_drawdown_duration_srs.max()
        return 0

    def cal_max_recover_duration(self):
        prev_high = self.m_nav_srs.iloc[0]
        prev_high_loc = 0
        for i, nav_i in enumerate(self.m_nav_srs):
            if nav_i >= prev_high:
                self.m_recover_duration_srs.iloc[i] = 0
                prev_high = nav_i
                prev_high_loc = i
            else:
                self.m_recover_duration_srs.iloc[i] = i - prev_high_loc
        self.m_max_recover_duration = self.m_recover_duration_srs.max()
        return

    def cal_all_indicators(self, t_method: str = "linear"):
        self.cal_return_mean()
        self.cal_return_std()
        self.cal_hold_period_return()
        self.cal_annual_return(t_method=t_method)
        self.cal_sharpe_ratio()
        self.cal_max_drawdown_scale()
        self.cal_max_drawdown_duration()
        self.cal_max_recover_duration()
        return 0

    def to_dict(self, t_type: str):
        """

        :param t_type: "eng": pure English characters, "chs": chinese characters can be read by Latex
        :return:
        """
        if t_type == "eng":
            d = {
                "return_mean": "{:.2f}".format(self.m_return_mean),
                "return_std": "{:.2f}".format(self.m_return_std),
                "hold_period_return": "{:.2f}".format(self.m_hold_period_return),
                "annual_return": "{:.2f}".format(self.m_annual_return),
                "sharpe_ratio": "{:.2f}".format(self.m_sharpe_ratio),
                "max_drawdown_scale": "{:.2f}".format(self.m_max_drawdown_scale),
                "max_drawdown_duration": "{:d}".format(self.m_max_drawdown_duration),
                "max_recover_duration": "{:d}".format(self.m_max_recover_duration),
            }
        elif t_type == "chs":
            d = {
                "收益率平均": "{:.2f}".format(self.m_return_mean),
                "收益率波动": "{:.2f}".format(self.m_return_std),
                "持有期收益": "{:.2f}".format(self.m_hold_period_return),
                "年化收益": "{:.2f}".format(self.m_annual_return),
                "夏普比率": "{:.2f}".format(self.m_sharpe_ratio),
                "最大回撤": "{:.2f}".format(self.m_max_drawdown_scale),
                "最长回撤期": "{:d}".format(self.m_max_drawdown_duration),
                "最长恢复期": "{:d}".format(self.m_max_recover_duration),
            }
        else:
            d = {}
        return d

    def display(self):
        print("| HPR = {:>7.4f} | AnnRtn = {:>7.4f} | MDD = {:>7.2f} | SPR = {:>7.4f} | ".format(
            self.m_hold_period_return,
            self.m_annual_return,
            self.m_max_drawdown_scale,
            self.m_sharpe_ratio,
        ))
        return 0


# class CNAVEnhanced(CNAV):
#     def __init__(self, t_raw_nav_srs: pd.Series, t_annual_rf_rate: float, t_freq: str):
#         super().__init__(t_raw_nav_srs, t_annual_rf_rate, t_freq)
#
#         self.m_max_drawdown_scale_date: str = self.m_nav_srs.index[0]
#         self.m_max_drawdown_scale_prev_high_date: str = ""
#         self.m_max_drawdown_scale_re_break_date: str = ""
#         self.m_max_drawdown_scale_duration: dict = {"natural": 0, "trade": 0}
#         self.m_max_drawdown_scale_recover_duration: dict = {"natural": 0, "trade": 0}
#
#     def cal_mdd_duration(self, t_calendar: CCalendar):
#         # mdd duration
#         _head_date = self.m_max_drawdown_scale_prev_high_date
#         _tail_date = self.m_max_drawdown_scale_date
#         self.m_max_drawdown_scale_duration["trade"] = int(t_calendar.get_sn(_tail_date) - t_calendar.get_sn(_head_date))
#         self.m_max_drawdown_scale_duration["natural"] = int((dt.datetime.strptime(_tail_date, "%Y%m%d") - dt.datetime.strptime(_head_date, "%Y%m%d")).days)
#
#         # recover duration
#         # days before the nav is beyond the prev high since the mdd happens
#         if self.m_max_drawdown_scale_re_break_date == "--":
#             self.m_max_drawdown_scale_recover_duration["trade"] = "--"
#             self.m_max_drawdown_scale_recover_duration["natural"] = "--"
#         else:
#             _head_date = self.m_max_drawdown_scale_date
#             _tail_date = self.m_max_drawdown_scale_re_break_date
#             self.m_max_drawdown_scale_recover_duration["trade"] = int(t_calendar.get_sn(_tail_date) - t_calendar.get_sn(_head_date))
#             self.m_max_drawdown_scale_recover_duration["natural"] = int((dt.datetime.strptime(_tail_date, "%Y%m%d") - dt.datetime.strptime(_head_date, "%Y%m%d")).days)
#         return 0
#
#     def to_dict(self, t_type: str):
#         d = super().to_dict(t_type)
#         if t_type == "eng":
#             d.update({
#                 "prev_high_date": self.m_max_drawdown_scale_prev_high_date,
#                 "max_drawdown_date": self.m_max_drawdown_scale_date,
#                 "mdd_duration_t": self.m_max_drawdown_scale_duration["trade"],
#                 "mdd_duration_n": self.m_max_drawdown_scale_duration["natural"],
#                 "re_break_date": self.m_max_drawdown_scale_re_break_date,
#                 "recover_duration_t": self.m_max_drawdown_scale_recover_duration["trade"],
#                 "recover_duration_n": self.m_max_drawdown_scale_recover_duration["natural"],
#             })
#         elif t_type == "chs":
#             d.update({
#                 "最大回撤开始日期": self.m_max_drawdown_scale_prev_high_date,
#                 "最大回撤持续时间": self.m_max_drawdown_scale_duration["trade"],
#                 "最大回撤结束日期": self.m_max_drawdown_scale_date,
#                 "恢复前高时间": self.m_max_drawdown_scale_recover_duration["trade"],
#                 "恢复前高日期": self.m_max_drawdown_scale_re_break_date,
#             })
#         else:
#             pass
#         return d
