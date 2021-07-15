import os
import numpy as np
import pandas as pd
import re


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

    def get_next_date(self, t_this_date: str, t_shift: int):
        '''
        t_shift : >0, in the future; <0, in the past
        '''
        t_this_sn = self.reverse_df.at[t_this_date, "sn"]
        t_next_sn = t_this_sn + t_shift
        if t_next_sn >= len(self.calendar_df):
            return ""
        else:
            return self.calendar_df.at[t_next_sn, "trade_date"]

    def get_fix_gap_dates_list(self, t_bgn_date: str, t_fix_gap: int):
        res = []
        for t_date in self.calendar_df["trade_date"]:
            if t_date < t_bgn_date:
                continue
            res.append(t_date)
        return res[::t_fix_gap]


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
