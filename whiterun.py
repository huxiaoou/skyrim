import os
import numpy as np
import pandas as pd
import re

"""
0.  provide two frequently used classes about trade date calendar and futures' instrument information
"""


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

    def get_date(self, t_sn: int):
        return self.calendar_df.at[t_sn, "trade_date"]

    def get_sn_ineq(self, t_base_date: str, t_type: str):
        if t_type == "<":
            return self.reverse_df.loc[self.reverse_df.index < t_base_date, "sn"].iloc[-1]
        if t_type == "<=":
            return self.reverse_df.loc[self.reverse_df.index <= t_base_date, "sn"].iloc[-1]
        if t_type == ">":
            return self.reverse_df.loc[self.reverse_df.index > t_base_date, "sn"].iloc[0]
        if t_type == ">=":
            return self.reverse_df.loc[self.reverse_df.index >= t_base_date, "sn"].iloc[0]
        return None

    def get_next_date(self, t_this_date: str, t_shift: int):
        """

        :param t_this_date:
        :param t_shift: t_shift : > 0, in the future; < 0, in the past
        :return:
        """

        if t_this_date in self.reverse_df.index:
            t_this_sn = self.reverse_df.at[t_this_date, "sn"]
            t_next_sn = t_this_sn + t_shift
            if t_next_sn >= len(self.calendar_df):
                return ""
            else:
                return self.calendar_df.at[t_next_sn, "trade_date"]
        else:
            return None

    def get_fix_gap_dates_list(self, t_bgn_date: str, t_fix_gap: int):
        t_bgn_sn = self.get_sn(t_base_date=t_bgn_date)
        return self.calendar_df["trade_date"].iloc[t_bgn_sn::t_fix_gap].tolist()


class CInstrumentCalendar(object):
    def __init__(self, t_instrument_id: str, t_calendar_dir: str):
        instru_calendar_file = "trade_calendar.{}.csv".format(t_instrument_id)
        instru_calendar_path = os.path.join(t_calendar_dir, instru_calendar_file)
        self.m_instru_calendar_df: pd.DataFrame = pd.read_csv(instru_calendar_path, dtype={"trade_date": str})

    def get_iter_list(self, t_bgn_date: str, t_stp_date: str):
        _iter_list = []
        for trade_date, section, contract in zip(
                self.m_instru_calendar_df["trade_date"],
                self.m_instru_calendar_df["section"],
                self.m_instru_calendar_df["trade_contract_id"],
        ):
            if t_bgn_date <= trade_date < t_stp_date:
                _iter_list.append((trade_date, section, contract))
        return _iter_list


class CInstrumentInfoTable(object):
    def __init__(self, t_path: os.path, t_index_label: str = "instrumentId", t_type: str = "EXCEL", t_sheet_name: str = "InstrumentInfo"):
        """

        :param t_path: InstrumentInfo file path, could be a txt(csv) or xlsx
        :param t_index_label: "instrumentId" or "windCode"
        :param t_type: "Excel" for xlsx, others for txt(csv)
        :param t_sheet_name: "InstrumentInfo", if t_type = "EXCEL"
        """
        if t_type.upper() == "EXCEL":
            self.instrument_info_df = pd.read_excel(t_path, sheet_name=t_sheet_name).set_index(t_index_label)
        else:
            self.instrument_info_df = pd.read_csv(t_path).set_index(t_index_label)
        self.instrument_info_df["precision"] = self.instrument_info_df["miniSpread"].map(lambda z: max(int(-np.floor(np.log10(z))), 0))

    def get_multiplier(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "contractMultiplier"]

    def get_minispread(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "miniSpread"]

    def get_exchangeId(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "exchangeId"]

    def get_precision(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "precision"]

    def get_exchangeId_chs(self, t_instrument_id: str):
        exchange_id_eng = self.instrument_info_df.at[t_instrument_id, "exchangeId"]
        exchange_id_chs = {
            "DCE": "大商所",
            "CZCE": "郑商所",
            "SHFE": "上期所",
            "INE": "上海能源",
            "CFFEX": "中金所",
        }[exchange_id_eng]
        return exchange_id_chs

    def get_windCode(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "windCode"]

    def get_ngt_sec_end_hour(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "ngtSecEndHour"]

    def get_ngt_sec_end_minute(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "ngtSecEndMinute"]

    def get_cost_rate_float(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "cost_float"]

    def get_cost_rate_fix(self, t_instrument_id: str):
        return self.instrument_info_df.at[t_instrument_id, "cost_fix"]

    def is_close_today_free(self, t_instrument_id: str) -> bool:
        return self.instrument_info_df.at[t_instrument_id, "isCloseTodayFree"] > 0

    def cal_abs_pnl(self, t_instrument_id: str, t_qty: int, t_open_price: float, t_close_price: float):
        _contract_multiplier = self.get_multiplier(t_instrument_id=t_instrument_id)
        return (t_close_price - t_open_price) * _contract_multiplier * t_qty

    def clearing_cost(self, t_instrument_id: str, t_qty: int, t_open_price: float, t_close_price: float, t_adjust_rate: float):
        """

        :param t_instrument_id:
        :param t_qty:
        :param t_open_price:
        :param t_close_price:
        :param t_adjust_rate: the ratio of real cost to base cost (charged by SHFE, DCE, CZCE), 1 means equal.
                             real cost = t_adjust_rate * base cost
        :return:
        """
        _contract_multiplier = self.get_multiplier(t_instrument_id=t_instrument_id)
        _rate_float = self.get_cost_rate_float(t_instrument_id=t_instrument_id)
        _rate_fix = self.get_cost_rate_fix(t_instrument_id=t_instrument_id)
        if self.is_close_today_free(t_instrument_id=t_instrument_id):
            _cost_float = (t_open_price + t_close_price) / 2 * _contract_multiplier * t_qty * _rate_float
            _cost_fix = _rate_fix
        else:
            _cost_float = (t_open_price + t_close_price) * _contract_multiplier * t_qty * _rate_float
            _cost_fix = _rate_fix * 2
        return (_cost_float + _cost_fix) * t_adjust_rate


def convert_contract_id_to_wind_format(t_contract_id: str, t_instru_info_table: CInstrumentInfoTable):
    """

    :param t_contract_id: general contract id, such as "j2209"
    :param t_instru_info_table:
    :return: "J2209.DCE"
    """

    _instrument_id = parse_instrument_from_contract(t_contract_id=t_contract_id)
    _exchange_id = t_instru_info_table.get_exchangeId(t_instrument_id=_instrument_id)
    return t_contract_id.upper() + "." + _exchange_id
