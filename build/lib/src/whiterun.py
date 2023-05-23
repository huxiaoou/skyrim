import os
import sys
import numpy as np
import pandas as pd
import re
from typing import List

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


class CCalendarBase(object):
    def __init__(self, t_save_path: str):
        self.save_path = t_save_path

    def update_all(self, t_df: pd.DataFrame):
        t_df.to_csv(self.save_path, index=False)
        print(t_df)
        return 0

    def update_increment(self, t_df: pd.DataFrame):
        try:
            calendar_df = pd.read_csv(self.save_path, dtype="str")
            print("Size of     calendar BEFORE update = {:>8d}".format(len(calendar_df)))
            print("Last day of calendar BEFORE update = {}".format(calendar_df["trade_date"].iloc[-1]))
            calendar_df = pd.concat([calendar_df, t_df], axis=0, ignore_index=True)
            calendar_df = calendar_df.sort_values(by="trade_date")
            calendar_df = calendar_df.drop_duplicates()
            print("Size of     calendar AFTER  update = {:>8d}".format(len(calendar_df)))
            print("Last day of calendar AFTER  update = {}".format(calendar_df["trade_date"].iloc[-1]))
            calendar_df.to_csv(self.save_path, index=False)
        except FileNotFoundError:
            print("Could not find {}, please check again".format(self.save_path))
        return 0


class CCalendar(CCalendarBase):
    def __init__(self, t_path: os.path):
        super().__init__(t_save_path=t_path)
        self.calendar_df = pd.read_csv(t_path, dtype=str)
        self.calendar_df["trade_date"] = self.calendar_df["trade_date"].map(lambda x: x.replace("-", ""))
        self.reverse_df = self.calendar_df.copy()
        self.reverse_df["sn"] = range(len(self.reverse_df))
        self.reverse_df = self.reverse_df.set_index("trade_date")

    def get_iter_list(self, t_bgn_date: str, t_stp_date: str, t_ascending: bool):
        res = []
        for t_date in self.calendar_df["trade_date"]:
            if t_date < t_bgn_date or t_date >= t_stp_date:
                continue
            res.append(t_date)
        if t_ascending:
            return res
        else:
            return sorted(res, reverse=True)

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
        :param t_shift: > 0, get date in the future; < 0, get date in the past
        :return:
        """

        try:
            t_this_sn = self.reverse_df.at[t_this_date, "sn"]
        except KeyError:
            print("{} is not in calendar".format(t_this_date))
            sys.exit()
        t_next_sn = t_this_sn + t_shift
        try:
            return self.calendar_df.at[t_next_sn, "trade_date"]
        except KeyError:
            print("{} {}{} days is not in calendar".format(
                t_this_date, "+" if t_shift > 0 else "", t_shift))
            sys.exit()

    def get_fix_gap_dates_list(self, t_bgn_date: str, t_fix_gap: int):
        t_bgn_sn = self.get_sn(t_base_date=t_bgn_date)
        return self.calendar_df["trade_date"].iloc[t_bgn_sn::t_fix_gap].tolist()


class CCalendarMonthly(CCalendar):
    def __init__(self, t_path: os.path):
        super().__init__(t_path)
        self.calendar_df["trade_month"] = self.calendar_df["trade_date"].map(lambda z: z[0:6])
        self.m_trade_months = list(self.calendar_df["trade_month"].unique())
        self.m_trade_months.sort()

    def get_trade_month_idx(self, t_trade_month: str):
        return self.m_trade_months.index(t_trade_month)

    def get_latest_month_from_trade_date(self, t_trade_date: str):
        _trade_month = t_trade_date[0:6]
        _trade_month_idx = self.get_trade_month_idx(_trade_month)
        return self.m_trade_months[_trade_month_idx - 1]

    def get_next_month(self, t_trade_month: str, t_shift: int):
        """

        :param t_trade_month:
        :param t_shift: > 0, in the future; < 0, in the past
        :return:
        """
        _trade_month_idx = self.get_trade_month_idx(t_trade_month)
        return self.m_trade_months[_trade_month_idx + t_shift]

    def get_iter_month(self, t_bgn_month: str, t_stp_month: str):
        _bgn_idx = self.get_trade_month_idx(t_bgn_month)
        _stp_idx = self.get_trade_month_idx(t_stp_month)
        return self.m_trade_months[_bgn_idx:_stp_idx]

    def get_trade_month_dates(self, t_trade_month: str):
        """

        :param t_trade_month: format = "YYYYMM", i.e. "202305"
        :return:
        """
        filter_trade_month = self.calendar_df["trade_month"] == t_trade_month
        trade_month_df = self.calendar_df.loc[filter_trade_month]
        return trade_month_df

    def get_first_date_of_month(self, t_trade_month: str):
        """

        :param t_trade_month: format = "YYYYMM", i.e. "202305"
        :return:
        """
        trade_month_df = self.get_trade_month_dates(t_trade_month)
        return trade_month_df["trade_date"].iloc[0]

    def get_last_date_of_month(self, t_trade_month: str):
        """

        :param t_trade_month: format = "YYYYMM", i.e. "202305"
        :return:
        """
        trade_month_df = self.get_trade_month_dates(t_trade_month)
        return trade_month_df["trade_date"].iloc[-1]

    def map_iter_dates_to_iter_months(self, bgn_date: str, stp_date: str):
        iter_dates = self.get_iter_list(bgn_date, stp_date, True)
        bgn_last_month = self.get_latest_month_from_trade_date(iter_dates[0])
        end_last_month = self.get_latest_month_from_trade_date(iter_dates[-1])
        stp_last_month = self.get_next_month(end_last_month, 1)
        iter_months = self.get_iter_month(bgn_last_month, stp_last_month)
        return iter_months

    def get_bgn_and_end_dates_for_trailing_window(self, end_month: str, trn_win: int):
        bgn_month = self.get_next_month(end_month, -trn_win + 1)
        bgn_date = self.get_first_date_of_month(bgn_month)
        end_date = self.get_last_date_of_month(end_month)
        return bgn_date, end_date


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
    _exchange_id = t_instru_info_table.get_windCode(t_instrument_id=_instrument_id).split(".")[1]
    return t_contract_id.upper() + "." + _exchange_id[0:3]


def fix_contract_id(x: str, t_exchange_id: str, t_instru_id_len: int, t_trade_date: str) -> str:
    """

    :param x: x should have a format like "MA105", in which "05" = May
              however "1" is ambiguous, since it could be either "2011" or "2021"
              this function is designed to solve this problem
    :param t_exchange_id: CZC, DCE, SHFE, INE.
    :param t_instru_id_len: len("MA") = 2
    :param t_trade_date: on which day, this contract is traded
    :return:
    """
    if t_exchange_id != "CZC":
        # this problem only happens for CZC
        return x

    if len(x) - t_instru_id_len > 3:
        # some old contract do have format like "MA1105"
        # in this case Nothing should be done
        return x

    td = int(t_trade_date[2])  # decimal year to be inserted, "X" in "20XYMMDD"
    ty = int(t_trade_date[3])  # trade year number,           "Y" in "20XYMMDD"
    cy = int(x[t_instru_id_len])  # contract year, "1" in "MA105"
    if cy < ty:
        # contract year should always be greater than or equal to the trade year
        # if not, decimal year +=1
        td += 1
    return x[0:t_instru_id_len] + str(td) + x[t_instru_id_len:]


# functions about Markdown
def df_to_md_strings(df: pd.DataFrame, using_index: bool = False, index_name: str = ""):
    def rejoin(s: List[str]):
        return "|" + "|".join(s) + "|"

    if using_index:
        df.index.name = index_name
    df_strs = df.to_string(index=using_index, index_names=using_index)
    rows = df_strs.split("\n")
    n_col = len(rows[0].split()) + int(using_index)
    md_rows = [rejoin(r.split()) for r in rows]
    if using_index:
        md_rows[0] = "|" + index_name + md_rows[0]
        md_rows.pop(1)
    md_rows.insert(1, rejoin(["---"] * n_col))
    return md_rows


def md_strings_to_md_file(md_rows: List[str], md_path: str):
    with open(md_path, "w+") as f:
        for md_row in md_rows:
            f.write(md_row + "\n")
    return 0


def df_to_md_files(df: pd.DataFrame, md_path: str, using_index: bool = False, index_name: str = ""):
    """

    :param df:
    :param md_path:
    :param using_index:
    :param index_name: if using_index, must be provided
    :return:
    """
    md_strings_to_md_file(md_rows=df_to_md_strings(df, using_index, index_name), md_path=md_path)
    return 0
