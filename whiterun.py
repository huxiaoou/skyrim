import os
import numpy as np
import pandas as pd
import re

SKYRIM_CONST_CALENDAR_PATH = os.path.join("E:\\", "Database", "Calendar", "cne_calendar.csv")
SKYRIM_CONST_INSTRUMENT_INFO_PATH = os.path.join("E:\\", "Database", "Futures", "InstrumentInfo.xlsx")


def parse_instrument_from_contract(t_contract_id: str) -> str:
    # s = 0
    # while t_contract_id[s] < "0" or t_contract_id[s] > "9":
    #     s += 1
    return re.sub("[0-9]", "", t_contract_id)


def parse_instrument_from_contract_wind(t_contract_id: str) -> str:
    return re.sub("[0-9]", "", t_contract_id)


class CCalendar(object):
    def __init__(self, t_path: os.path):
        self.calendar_df = pd.read_csv(t_path, dtype=str)
        self.calendar_df["trade_date"] = self.calendar_df["trade_date"].map(lambda x: x.replace("-", ""))
        self.reverse_df = self.calendar_df.copy()
        self.reverse_df["sn"] = range(len(self.reverse_df))
        self.reverse_df = self.reverse_df.set_index("trade_date")

    def get_iter_list(self, t_bgn_date, t_stp_date, t_ascending):
        res = []
        for t_date in self.calendar_df["trade_date"]:
            if t_date < t_bgn_date or t_date >= t_stp_date:
                continue
            res.append(t_date)
        if t_ascending:
            return res
        else:
            return sorted(res, reverse=True)

    def get_hist_date_since_begin_date(self, t_bgn_date: str, t_stp_date: str):
        idx_filter = (self.calendar_df["trade_date"] >= t_bgn_date) & (self.calendar_df["trade_date"] < t_stp_date)
        df = self.calendar_df.loc[idx_filter].reset_index(drop=True)
        return df

    def find_shift_date(self, t_base_date: str, t_shift: int):
        # t_shift > 0: date in the future
        # t_shift = 0: no shift
        # t_shift < 0: date in the past
        test_idx = self.reverse_df.at[t_base_date, "sn"]
        return self.calendar_df["trade_date"].iloc[test_idx + t_shift]

    def get_sn(self, t_base_date: str):
        return self.reverse_df.at[t_base_date, "sn"]

    def get_next_date(self, t_this_date, t_shift: int):
        '''
        t_shift : >0, in the future; <0, in the past
        '''
        t_this_sn = self.reverse_df.at[t_this_date, "sn"]
        t_next_sn = t_this_sn + t_shift
        if t_next_sn >= len(self.calendar_df):
            return ""
        else:
            return self.calendar_df.at[t_next_sn, "trade_date"]


class CInstrumentInfoTable(object):
    def __init__(self, t_path: os.path, t_index_label):
        self.instrument_info_df = pd.read_excel(t_path, sheet_name="InstrumentInfo").set_index(t_index_label)

    def get_multiplier(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "contractMultiplier"]

    def get_minispread(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "miniSpread"]

    def get_exchangeId(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "exchangeId"]

    def get_windCode(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "windCode"]


class CNAV(object):
    def __init__(self, t_nav_srs: pd.Series, t_annual_rf_rate: float, t_freq: str):
        """

        :param t_nav_srs: net assets value series, with date string as index.
        :param t_annual_rf_rate: annualized risk free rate, multiplied by 100.
        :param t_freq: freqency for nav series.
        """
        self.m_nav_srs: pd.Series = t_nav_srs
        self.m_rtn_srs: pd.Series = (np.log(t_nav_srs / t_nav_srs.shift(1)) * 100).fillna(0)
        self.m_obs = len(t_nav_srs)

        self.m_annual_factor = {
            "D": 252,
            "W": 52,
            "M": 12,
            "Y": 1,
        }[t_freq]

        self.m_annual_rf_rate = t_annual_rf_rate
        self.m_annual_return = 0
        self.m_hold_period_return = 0
        self.m_sharpe_ratio = 0
        self.m_max_drawdown = 0
        self.m_max_drawdown_date = self.m_nav_srs.index[0]

        self.m_max_drawdown_prev_high_date = ""
        self.m_max_drawdown_re_break_date = ""
        self.m_max_drawdown_duration = {"natural": 0, "trade": 0}
        self.m_max_drawdown_recover_duration = {"natural": 0, "trade": 0}

    def get_sharpe_ratio(self):
        return self.m_sharpe_ratio

    def cal_hold_period_return(self):
        self.m_hold_period_return = (self.m_nav_srs.iloc[-1] / self.m_nav_srs.iloc[0] - 1) * 100
        return 0

    def cal_annual_return(self):
        self.m_annual_return = self.m_rtn_srs.mean() * self.m_annual_factor
        return 0

    def cal_sharpe_ratio(self):
        diff_srs = self.m_rtn_srs - self.m_annual_rf_rate / self.m_annual_factor
        mu = diff_srs.mean()
        sd = diff_srs.std()
        self.m_sharpe_ratio = mu / sd * np.sqrt(self.m_annual_factor)
        return

    def cal_max_drawdown(self):
        nav_hist_high = 1.0
        self.m_max_drawdown = 0
        for trade_date in self.m_nav_srs.index:
            nav_val = self.m_nav_srs[trade_date]
            if nav_val > nav_hist_high:
                nav_hist_high = nav_val
            # new_drawback = nav_hist_high - nav_val  # absolute way
            new_drawback = (1 - nav_val / nav_hist_high) * 100  # relative way
            if new_drawback >= self.m_max_drawdown:
                self.m_max_drawdown = new_drawback
                self.m_max_drawdown_date = trade_date

        prev_mdd_srs = self.m_nav_srs[self.m_nav_srs.index < self.m_max_drawdown_date]  # type:pd.Series
        this_mdd_srs = self.m_nav_srs[self.m_nav_srs.index >= self.m_max_drawdown_date]  # type:pd.Series
        self.m_max_drawdown_prev_high_date = prev_mdd_srs.idxmax()
        prev_high_value = prev_mdd_srs.max()
        re_break_idx = this_mdd_srs >= prev_high_value
        if any(re_break_idx):
            re_break_srs = this_mdd_srs[re_break_idx]
            self.m_max_drawdown_re_break_date = re_break_srs.index[0]
        else:
            self.m_max_drawdown_re_break_date = "Never"
        return 0

    def cal_mdd_duration(self, t_calendar: CCalendar):
        # mdd duration
        _head_date = self.m_max_drawdown_prev_high_date.replace("-", "")
        _tail_date = self.m_max_drawdown_date.replace("-", "")
        self.m_max_drawdown_duration["trade"] = t_calendar.get_date_idx(_tail_date) - t_calendar.get_date_idx(_head_date)
        self.m_max_drawdown_duration["natural"] = (dt.datetime.strptime(_tail_date, "%Y%m%d") - dt.datetime.strptime(_head_date, "%Y%m%d")).days

        # recover duration
        if self.m_max_drawdown_re_break_date == "Never":
            self.m_max_drawdown_recover_duration["trade"] = np.inf
            self.m_max_drawdown_recover_duration["natural"] = np.inf
        else:
            _head_date = self.m_max_drawdown_date.replace("-", "")
            _tail_date = self.m_max_drawdown_re_break_date.replace("-", "")
            self.m_max_drawdown_recover_duration["trade"] = t_calendar.get_date_idx(_tail_date) - t_calendar.get_date_idx(_head_date)
            self.m_max_drawdown_recover_duration["natural"] = (dt.datetime.strptime(_tail_date, "%Y%m%d") - dt.datetime.strptime(_head_date, "%Y%m%d")).days
        return 0

    def to_dict(self):
        d = {
            "hold_period_return": self.m_hold_period_return,
            "annual_return": self.m_annual_return,
            "sharpe_ratio": self.m_sharpe_ratio,
            "max_drawdown": self.m_max_drawdown,
            "max_drawdown_date": self.m_max_drawdown_date,
            "prev_high_date": self.m_max_drawdown_prev_high_date,
            "re_break_date": self.m_max_drawdown_re_break_date,
            "mdd_duration_t": self.m_max_drawdown_duration["trade"],
            "mdd_duration_n": self.m_max_drawdown_duration["natural"],
            "recover_duration_t": self.m_max_drawdown_recover_duration["trade"],
            "recover_duration_n": self.m_max_drawdown_recover_duration["natural"],
        }
        return d

    def display(self):
        print("| HPR = {:>7.4f} | AnnRtn = {:>7.4f} | MDD = {:>7.2f} | SPR = {:>7.4f} | ".format(
            self.m_hold_period_return,
            self.m_annual_return,
            self.m_max_drawdown,
            self.m_sharpe_ratio,
        ))
        return 0
