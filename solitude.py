import os
import numpy as np
import pandas as pd

"""
0.  this module is designed to load tick data recored by myself
1.  tick data are supposed to be saved by section
"""

NEW_FORMAT_BEGIN_DATE = "20190101"

OLD_CTP_FORMAT = [
    "dateTime",
    "lastPrice",
    "volume",
    "bidPrice",
    "bidQty",
    "askPrice",
    "askQty",
    "openInt",
    "turnover",
]

NEW_CTP_FORMAT = [
    "dateTime",
    "dateTimeLOC",
    "lastPrice",
    "volume",
    "bidPrice",
    "bidQty",
    "askPrice",
    "askQty",
    "openInt",
    "turnover",
]

VOL_SPLIT_DATE = "20200101"


# ---------- class calendar ----------
class CTradeCalendar(object):
    def __init__(self, t_instrument_id: str, t_ctp_database: str):
        self.m_instrument_id: str = t_instrument_id
        trade_calendar_path = os.path.join(t_ctp_database, "trade_calendar.{}.csv".format(t_instrument_id))
        self.trade_calendar_df: pd.DataFrame = pd.read_csv(trade_calendar_path, dtype={"trade_date": str})

    def get_iter_list(self):
        return zip(
            self.trade_calendar_df["trade_date"],
            self.trade_calendar_df["section"],
            self.trade_calendar_df["trade_contract_id"]
        )


# ---------- class tick data engine ----------
class CTickDataEngine(object):
    def __init__(self, t_instrument_id: str, t_bgn_date: str, t_stp_date: str, t_ctp_database: str, verbose=False):
        self.m_instrument_id: str = t_instrument_id
        self.m_trade_calendar: CTradeCalendar = CTradeCalendar(t_instrument_id, t_ctp_database)
        self.m_tick_dfs_qty: int = 0
        self.m_tick_dfs_list = []

        print("begin to initialize data engine for: {:<8s}".format(t_instrument_id))
        print("print loading progress ............: {:<8s}".format("Y" if verbose else "N"))
        print("begin date ........................: {:<8s}".format(t_bgn_date))
        print("stop  date ........................: {:<8s}".format(t_stp_date))

        # main loop
        for trade_date, section, contract in self.m_trade_calendar.get_iter_list():
            # date selection
            if trade_date < t_bgn_date or trade_date >= t_stp_date:
                continue

            # load file according different format
            tick_data_file = trade_date + "." + section + "." + contract + ".csv.gz"
            tick_path = os.path.join(t_ctp_database, "0.raw", trade_date[0:4], trade_date, tick_data_file)
            if trade_date < NEW_FORMAT_BEGIN_DATE:
                tick_df = pd.read_csv(tick_path, header=None, names=OLD_CTP_FORMAT, parse_dates=["dateTime"])
            else:
                tick_df = pd.read_csv(tick_path, header=None, names=NEW_CTP_FORMAT, parse_dates=["dateTime", "dateTimeLOC"])

            # UP AND DOWN STOP
            bid_null_idx = tick_df["bidPrice"] <= 0
            ask_null_idx = tick_df["askPrice"] <= 0
            tick_df.loc[bid_null_idx, "bidPrice"] = np.nan
            tick_df.loc[ask_null_idx, "askPrice"] = np.nan
            if np.any(tick_df["bidPrice"].isnull()):
                print("WARNING! A DOWN STOP AT {}:{}".format(trade_date, section))
                tick_df["bidPrice"] = tick_df["bidPrice"].fillna(tick_df["bidPrice"].min())
            if np.any(tick_df["askPrice"].isnull()):
                print("WARNING! A UP STOP AT {}:{}".format(trade_date, section))
                tick_df["askPrice"] = tick_df["askPrice"].fillna(tick_df["askPrice"].max())

            # add to set
            self.m_tick_dfs_list.append((trade_date, section, contract, tick_df))
            self.m_tick_dfs_qty += 1
            if verbose:
                print("| {} | {} | {} | loaded |".format(trade_date, section, contract))

        print("Data engine of {} from {} to {} initiated".format(self.m_instrument_id, t_bgn_date, t_stp_date))

    def get_tick_dfs(self):
        return self.m_tick_dfs_list
