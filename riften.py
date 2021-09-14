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

        :param t_raw_nav_srs: the Net-Assets-Value series, with datetime-like index in string format.
                          The first item in this series can not be ONE, and the class will do the conversion
                          when initialized.
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
        self.m_max_drawdown: float = 0  # a non negative float
        self.m_max_drawdown_date: str = self.m_nav_srs.index[0]

        # secondary
        self.m_max_drawdown_prev_high_date: str = ""
        self.m_max_drawdown_re_break_date: str = ""
        self.m_max_drawdown_duration: dict = {"natural": 0, "trade": 0}
        self.m_max_drawdown_recover_duration: dict = {"natural": 0, "trade": 0}

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

    def cal_max_drawdown(self):
        nav_hist_high = self.m_nav_srs.iloc[0]
        self.m_max_drawdown = 0
        for trade_date in self.m_nav_srs.index:
            nav_val = self.m_nav_srs[trade_date]
            if nav_val > nav_hist_high:
                nav_hist_high = nav_val
            new_drawdown = (1 - nav_val / nav_hist_high) * RETURN_SCALE  # relative way
            if new_drawdown >= self.m_max_drawdown:
                self.m_max_drawdown = new_drawdown
                self.m_max_drawdown_date = trade_date

        prev_mdd_srs: pd.Series = self.m_nav_srs[self.m_nav_srs.index < self.m_max_drawdown_date]  # nav series before max drawdown date
        this_mdd_srs: pd.Series = self.m_nav_srs[self.m_nav_srs.index >= self.m_max_drawdown_date]  # nav series after max drawdown date
        self.m_max_drawdown_prev_high_date = prev_mdd_srs.idxmax()
        prev_high_value = prev_mdd_srs.max()
        re_break_idx = this_mdd_srs >= prev_high_value
        if any(re_break_idx):
            self.m_max_drawdown_re_break_date = this_mdd_srs[re_break_idx].index[0]
        else:
            self.m_max_drawdown_re_break_date = "--"
        return 0

    def cal_mdd_duration(self, t_calendar: CCalendar):
        # mdd duration
        _head_date = self.m_max_drawdown_prev_high_date
        _tail_date = self.m_max_drawdown_date
        self.m_max_drawdown_duration["trade"] = int(t_calendar.get_sn(_tail_date) - t_calendar.get_sn(_head_date))
        self.m_max_drawdown_duration["natural"] = int((dt.datetime.strptime(_tail_date, "%Y%m%d") - dt.datetime.strptime(_head_date, "%Y%m%d")).days)

        # recover duration
        # days before the nav is beyond the prev high since the mdd happens
        if self.m_max_drawdown_re_break_date == "--":
            self.m_max_drawdown_recover_duration["trade"] = "--"
            self.m_max_drawdown_recover_duration["natural"] = "--"
        else:
            _head_date = self.m_max_drawdown_date
            _tail_date = self.m_max_drawdown_re_break_date
            self.m_max_drawdown_recover_duration["trade"] = int(t_calendar.get_sn(_tail_date) - t_calendar.get_sn(_head_date))
            self.m_max_drawdown_recover_duration["natural"] = int((dt.datetime.strptime(_tail_date, "%Y%m%d") - dt.datetime.strptime(_head_date, "%Y%m%d")).days)
        return 0

    def cal_all_indicators(self, t_calendar: CCalendar, t_method: str = "linear"):
        self.cal_return_mean()
        self.cal_return_std()
        self.cal_hold_period_return()
        self.cal_annual_return(t_method=t_method)
        self.cal_sharpe_ratio()
        self.cal_max_drawdown()
        self.cal_mdd_duration(t_calendar=t_calendar)
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
                "max_drawdown": "{:.2f}".format(self.m_max_drawdown),
                "prev_high_date": self.m_max_drawdown_prev_high_date,
                "max_drawdown_date": self.m_max_drawdown_date,
                "mdd_duration_t": self.m_max_drawdown_duration["trade"],
                "mdd_duration_n": self.m_max_drawdown_duration["natural"],
                "re_break_date": self.m_max_drawdown_re_break_date,
                "recover_duration_t": self.m_max_drawdown_recover_duration["trade"],
                "recover_duration_n": self.m_max_drawdown_recover_duration["natural"],
            }
        elif t_type == "chs":
            d = {
                "收益率平均": "{:.2f}".format(self.m_return_mean),
                "收益率波动": "{:.2f}".format(self.m_return_std),
                "持有期收益": "{:.2f}".format(self.m_hold_period_return),
                "年化收益": "{:.2f}".format(self.m_annual_return),
                "夏普比率": "{:.2f}".format(self.m_sharpe_ratio),
                "最大回撤": "{:.2f}".format(self.m_max_drawdown),
                "最大回撤开始日期": self.m_max_drawdown_prev_high_date,
                "最大回撤持续时间": self.m_max_drawdown_duration["trade"],
                "最大回撤结束日期": self.m_max_drawdown_date,
                "恢复前高时间": self.m_max_drawdown_recover_duration["trade"],
                "恢复前高日期": self.m_max_drawdown_re_break_date,
            }
        else:
            d = {}
        return d

    def display(self):
        print("| HPR = {:>7.4f} | AnnRtn = {:>7.4f} | MDD = {:>7.2f} | SPR = {:>7.4f} | ".format(
            self.m_hold_period_return,
            self.m_annual_return,
            self.m_max_drawdown,
            self.m_sharpe_ratio,
        ))
        return 0
