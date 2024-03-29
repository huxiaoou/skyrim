import numpy as np
import pandas as pd

'''
created @ 2021-02-22
0.  define a class to evaluate the performance of some portfolio
1.  this class provide methods to calculate some frequently used index
'''


class CNAV(object):
    def __init__(self, t_raw_nav_srs: pd.Series, t_annual_rf_rate: float, t_annual_factor: float = 252, t_ret_scale: int = 100, t_type: str = "NAV"):
        """

        :param t_raw_nav_srs: A. if t_type == "NAV":
                                    the Net-Assets-Value series, with datetime-like index in string format.
                                    The first item in this series could not be ONE, and the class will do the conversion
                                    when initialized.
                                 elif t_type == "RET":
                                    the Assets Return series, the return should NOT be multiplied by RETURN_SCALE

                              B. the index of the series is supposed to be continuous, i.e., there are not any missing
                                 dates or timestamp in the index.
        :param t_annual_rf_rate: annualized risk-free rate, must NOT be multiplied by the return scale.
                                 the class will do the conversion when initialized
        :param t_annual_factor: if the average of return series means a return with hold period = T trading days,
                                this should be 252 / T
                                daily returns, T = 1, default value
                                weekly returns, T = 5
                                monthly returns, T = 21
                                quarterly returns, T = 63
        :param t_type: "NAV" or "RET
        """
        self.m_ret_scale: int = t_ret_scale
        self.return_type = t_type.upper()

        if self.return_type == "NAV":
            self.m_nav_srs: pd.Series = t_raw_nav_srs / t_raw_nav_srs.iloc[0]  # set the first value to be 1
            self.m_rtn_srs: pd.Series = ((t_raw_nav_srs / t_raw_nav_srs.shift(1) - 1) * self.m_ret_scale).fillna(0)  # has the same length as nav srs
        elif self.return_type == "RET":
            self.m_rtn_srs: pd.Series = t_raw_nav_srs * self.m_ret_scale
            self.m_nav_srs: pd.Series = (t_raw_nav_srs + 1).cumprod()
        else:
            print("Not a right type parameter, please check again.")
            self.m_nav_srs = None
            self.m_rtn_srs = None

        self.m_obs: int = len(t_raw_nav_srs)

        self.m_annual_factor: float = t_annual_factor

        self.m_annual_rf_rate: float = t_annual_rf_rate * self.m_ret_scale

        # frequently used performance index
        # primary
        self.m_return_mean: float = 0
        self.m_return_std: float = 0
        self.m_hold_period_return: float = 0
        self.m_annual_return: float = 0
        self.m_annual_volatility: float = 0
        self.m_sharpe_ratio: float = 0
        self.m_calmar_ratio: float = 0
        self.m_value_at_risks: dict = {}

        # secondary - A max drawdown scale
        self.m_max_drawdown_scale: float = 0  # a non-negative float, multiplied by RETURN_SCALE
        self.m_max_drawdown_scale_idx: str = ""
        self.m_drawdown_scale_srs: pd.Series = pd.Series(data=0.0, index=self.m_nav_srs.index)

        # secondary - B max drawdown duration
        self.m_max_drawdown_duration: int = 0  # a non-negative int, stands for the duration of drawdown
        self.m_max_drawdown_duration_idx: str = ""
        self.m_drawdown_duration_srs: pd.Series = pd.Series(data=0, index=self.m_nav_srs.index)

        # secondary - C max recover duration
        self.m_max_recover_duration: int = 0
        self.m_max_recover_duration_idx: str = ""
        self.m_recover_duration_srs: pd.Series = pd.Series(data=0, index=self.m_nav_srs.index)

    def cal_return_mean(self):
        self.m_return_mean = self.m_rtn_srs.mean()
        return 0

    def cal_return_std(self):
        self.m_return_std = self.m_rtn_srs.std()
        return 0

    def cal_hold_period_return(self):
        if self.return_type == "NAV":
            self.m_hold_period_return = (self.m_nav_srs.iloc[-1] / self.m_nav_srs.iloc[0] - 1) * self.m_ret_scale
        else:
            self.m_hold_period_return = (self.m_nav_srs.iloc[-1] - 1) * self.m_ret_scale
        return 0

    def cal_annual_return(self, t_method: str = "linear"):
        self.m_annual_volatility = self.m_rtn_srs.std() * np.sqrt(self.m_annual_factor)
        if t_method == "linear":
            self.m_annual_return = self.m_rtn_srs.mean() * self.m_annual_factor
        else:
            self.m_annual_return = (np.power(self.m_hold_period_return / self.m_ret_scale + 1, self.m_annual_factor / len(self.m_rtn_srs)) - 1) * self.m_ret_scale
        return 0

    def cal_sharpe_ratio(self):
        diff_srs = self.m_rtn_srs - self.m_annual_rf_rate / self.m_annual_factor
        mu = diff_srs.mean()
        sd = diff_srs.std()
        self.m_sharpe_ratio = mu / sd * np.sqrt(self.m_annual_factor)
        return 0

    def cal_max_drawdown_scale(self):
        self.m_drawdown_scale_srs: pd.Series = (1 - self.m_nav_srs / self.m_nav_srs.cummax()) * self.m_ret_scale
        self.m_max_drawdown_scale = self.m_drawdown_scale_srs.max()
        self.m_max_drawdown_scale_idx = self.m_drawdown_scale_srs.idxmax()
        return 0

    def cal_calmar_ratio(self):
        self.cal_annual_return()
        self.cal_max_drawdown_scale()
        self.m_calmar_ratio = self.m_annual_return / self.m_max_drawdown_scale
        return 0

    def cal_max_drawdown_duration(self):
        prev_high = self.m_nav_srs.iloc[0]
        prev_high_loc = 0
        prev_drawdown_scale = 0.0
        drawdown_loc = 0
        for i, nav_i in enumerate(self.m_nav_srs):
            if nav_i > prev_high:
                prev_high = nav_i
                prev_high_loc = i
                prev_drawdown_scale = 0
            drawdown_scale = 1 - nav_i / prev_high
            if drawdown_scale > prev_drawdown_scale:
                prev_drawdown_scale = drawdown_scale
                drawdown_loc = i
            self.m_drawdown_duration_srs.iloc[i] = drawdown_loc - prev_high_loc
        self.m_max_drawdown_duration = self.m_drawdown_duration_srs.max()
        self.m_max_drawdown_duration_idx = self.m_drawdown_duration_srs.idxmax()
        return 0

    def cal_max_recover_duration(self):
        prev_high = self.m_nav_srs.iloc[0]
        prev_high_loc = 0
        for i, nav_i in enumerate(self.m_nav_srs):
            if nav_i > prev_high:
                self.m_recover_duration_srs.iloc[i] = 0
                prev_high = nav_i
                prev_high_loc = i
            else:
                self.m_recover_duration_srs.iloc[i] = i - prev_high_loc
        self.m_max_recover_duration = self.m_recover_duration_srs.max()
        self.m_max_recover_duration_idx = self.m_recover_duration_srs.idxmax()
        return

    def cal_value_at_risk(self, t_qs: tuple):
        self.m_value_at_risks.update({"q{:02d}".format(q): np.percentile(self.m_rtn_srs, q) for q in t_qs})

    def cal_all_indicators(self, t_method: str = "linear", t_qs: tuple = ()):
        """

        :param t_method:
        :param t_qs: Percentage or sequence of percentages for the percentiles to compute.
                     Values must be between 0 and 100 inclusive.
        :return:
        """
        self.cal_return_mean()
        self.cal_return_std()
        self.cal_hold_period_return()
        self.cal_annual_return(t_method=t_method)
        self.cal_sharpe_ratio()
        self.cal_max_drawdown_scale()
        self.cal_calmar_ratio()
        self.cal_max_drawdown_duration()
        self.cal_max_recover_duration()
        self.cal_value_at_risk(t_qs=t_qs)
        return 0

    def to_dict(self, t_type: str):
        """

        :param t_type: "eng": pure English characters, "chs": chinese characters can be read by Latex
        :return:
        """
        if t_type.lower() == "eng":
            d = {
                "return_mean": "{:.3f}".format(self.m_return_mean),
                "return_std": "{:.3f}".format(self.m_return_std),
                "hold_period_return": "{:.2f}".format(self.m_hold_period_return),
                "annual_return": "{:.2f}".format(self.m_annual_return),
                "annual_volatility": "{:.2f}".format(self.m_annual_volatility),
                "sharpe_ratio": "{:.2f}".format(self.m_sharpe_ratio),
                "calmar_ratio": "{:.2f}".format(self.m_calmar_ratio),
                "max_drawdown_scale": "{:.2f}".format(self.m_max_drawdown_scale),
                "max_drawdown_scale_idx": "{:s}".format(self.m_max_drawdown_scale_idx),
                "max_drawdown_duration": "{:d}".format(self.m_max_drawdown_duration),
                "max_drawdown_duration_idx": "{:s}".format(self.m_max_drawdown_duration_idx),
                "max_recover_duration": "{:d}".format(self.m_max_recover_duration),
                "max_recover_duration_idx": "{:s}".format(self.m_max_recover_duration_idx),
            }
            d.update({k: "{:.3f}".format(v) for k, v in self.m_value_at_risks.items()})
        elif t_type.lower() == "chs":
            d = {
                "收益率平均": "{:.3f}".format(self.m_return_mean),
                "收益率波动": "{:.3f}".format(self.m_return_std),
                "持有期收益": "{:.2f}".format(self.m_hold_period_return),
                "年化收益": "{:.2f}".format(self.m_annual_return),
                "年化波动": "{:.2f}".format(self.m_annual_volatility),
                "夏普比率": "{:.2f}".format(self.m_sharpe_ratio),
                "卡玛比率": "{:.2f}".format(self.m_calmar_ratio),
                "最大回撤": "{:.2f}".format(self.m_max_drawdown_scale),
                "最大回撤时点": "{:s}".format(self.m_max_drawdown_scale_idx),
                "最长回撤期": "{:d}".format(self.m_max_drawdown_duration),
                "最长回撤期时点": "{:s}".format(self.m_max_drawdown_duration_idx),
                "最长恢复期": "{:d}".format(self.m_max_recover_duration),
                "最长恢复期时点": "{:s}".format(self.m_max_recover_duration_idx),
            }
            d.update({k: "{:.3f}".format(v) for k, v in self.m_value_at_risks.items()})
        else:
            d = {}
        return d

    def display(self):
        print("| HPR = {:>7.4f} | AnnRtn = {:>7.4f} | MDD = {:>7.2f} | SPR = {:>7.4f} | CMR = {:>7.4f} |".format(
            self.m_hold_period_return,
            self.m_annual_return,
            self.m_max_drawdown_scale,
            self.m_sharpe_ratio,
            self.m_calmar_ratio
        ))
        return 0
