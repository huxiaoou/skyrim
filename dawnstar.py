import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style
from typing import List, Dict
from skyrim.whiterun import CCalendar
from skyrim.winterhold import check_and_mkdir, remove_files_in_the_dir

'''
Add Module Dawnstar with classes for trade and classes for portfolio to skyrim
0.  this module provides some Classes to Equity simulation
1.  the simulation is like Strategy Dragon and Tiger:
    A.  suppose signals are calculated after the close phase of day T-1, k securities are selected.
    B.  k securities are bought(open position) at open price of day T, if rise stop condition of this security is not met,
        otherwise the strategy will abandon this opportunity.
        And the sum of their market cap should be about 1/h of total account value, which means 1/kh of the total account value
        is allocated for each of this k security.
    C.  the k securities are sold(close position) at close price of day T+h, if fall stop condition of this security is not met, 
        otherwise try to sell this security at next day close price.
    D.  Mirror operations are allowed.
'''

pd.set_option("display.width", 0)
matplotlib.style.use("Solarize_Light2")

RETURN_SCALE = 100
BASE_SHARES = 100


class CTradeBase(object):
    def __init__(self, t_sid: str, t_md_dir: str, t_cost_rate: float, t_stop_return: float):
        self.m_sid: str = t_sid
        md_file = "{}.md.csv.gz".format(t_sid)
        md_path = os.path.join(t_md_dir, md_file)
        self.m_md: pd.DataFrame = pd.read_csv(
            md_path, dtype={"trade_date": str}, usecols=["trade_date", "pct_chg", "open", "close"]
        ).set_index("trade_date")

        # pct_chg: has been multiplied by 100

        self.m_open_date: str = ""
        self.m_update_date: str = ""
        self.m_close_date: str = ""

        self.m_direction: int = 0
        self.m_quantity: int = 0

        self.m_open_price: float = -1
        self.m_nav: float = 1
        self.m_update_price: float = -1
        self.m_close_price: float = -1

        self.m_unrealized_pnl: float = 0
        self.m_realized_pnl: float = 0
        self.m_cost_rate: float = t_cost_rate

        self.m_stop_return: float = t_stop_return
        self.m_md_end_date: str = self.m_md.index[-1]

    def initialize_at_beginning_of_each_day(self):
        self.m_unrealized_pnl = 0
        return 0

    def open(self, t_open_date: str, t_allocated_amt: float, t_direction: int, t_prev_date: str) -> bool:
        prev_close = self.m_md.at[t_prev_date, "close"]
        this_open = self.m_md.at[t_open_date, "open"]
        if (this_open > prev_close * (1 + self.m_stop_return / RETURN_SCALE)) and (t_direction > 0):
            return False
        elif (this_open < prev_close * (1 - self.m_stop_return / RETURN_SCALE)) and (t_direction < 0):
            return False
        else:
            self.m_quantity = BASE_SHARES * int(t_allocated_amt / this_open / BASE_SHARES / (1 + self.m_cost_rate))
            if self.m_quantity <= 0:
                return False

            self.m_direction = t_direction
            self.m_open_date = t_open_date
            self.m_open_price = this_open
            return True

    def get_cash_locked(self):
        _cost = self.m_open_price * self.m_quantity * self.m_cost_rate
        _cash_locked = self.m_open_price * self.m_quantity
        return _cost, _cash_locked

    def update(self, t_update_date: str):
        self.m_update_date = t_update_date
        if t_update_date == self.m_open_date:
            ret = self.m_md.at[t_update_date, "close"] / self.m_md.at[t_update_date, "open"] - 1
        else:
            ret = self.m_md.at[t_update_date, "pct_chg"] / RETURN_SCALE
        self.m_nav *= 1 + ret
        self.m_update_price = self.m_open_price * self.m_nav
        self.m_unrealized_pnl = (self.m_update_price - self.m_open_price) * self.m_quantity * self.m_direction
        return 0

    def check_close_condition(self) -> bool:
        pass

    def close(self) -> bool:
        pct_chg = self.m_md.at[self.m_update_date, "pct_chg"]
        if self.check_close_condition():
            if (self.m_direction > 0) and (pct_chg >= -self.m_stop_return):
                self.m_close_price = self.m_update_price
                self.m_realized_pnl = self.m_unrealized_pnl
                self.m_unrealized_pnl = 0
                return True
            if (self.m_direction < 0) and (pct_chg < self.m_stop_return):
                self.m_close_price = self.m_update_price
                self.m_realized_pnl = self.m_unrealized_pnl
                self.m_unrealized_pnl = 0
                return True
        return False

    def get_cash_unlocked(self):
        _cost = self.m_close_price * self.m_quantity * self.m_cost_rate
        _cash_unlocked = (self.m_open_price + (self.m_close_price - self.m_open_price) * self.m_direction) * self.m_quantity
        return _cost, _cash_unlocked

    def get_realized_pnl(self):
        return self.m_realized_pnl

    def get_unrealized_pnl(self):
        return self.m_unrealized_pnl

    def to_dict(self):
        return {
            "sid": self.m_sid,
            "open_date": self.m_open_date,
            "update_date": self.m_update_date,
            "direction": self.m_direction,
            "quantity": self.m_quantity,
            "open_price": self.m_open_price,
            "update_price": self.m_update_price,
            "unrealized_pnl": self.m_unrealized_pnl,
            "mkt_val": self.m_quantity * self.m_update_price,
        }


class CTradeL1(CTradeBase):
    def __init__(self, t_sid: str, t_md_dir: str, t_cost_rate: float, t_stop_return: float):
        super().__init__(t_sid, t_md_dir, t_cost_rate, t_stop_return)
        self.m_last_hold_date: str = ""

    def set_close_condition(self, t_last_hold_date: str, t_simu_end_date: str):
        if self.m_md_end_date < t_simu_end_date:
            self.m_last_hold_date = min(self.m_md_end_date, t_last_hold_date)
        else:
            self.m_last_hold_date = t_last_hold_date
        return 0

    def check_close_condition(self) -> bool:
        if self.m_update_date >= self.m_last_hold_date:
            return True
        return False


class CTradeL2(CTradeL1):
    def __init__(self, t_sid: str, t_md_dir: str, t_cost_rate: float, t_stop_return: float):
        super().__init__(t_sid, t_md_dir, t_cost_rate, t_stop_return)
        self.m_mdd_cap: float = 0
        self.m_mdd: float = 0
        self.m_h: float = 1
        self.m_l: float = 1

    def set_close_condition_mdd(self, t_mdd_cap: float):
        self.m_mdd_cap = t_mdd_cap

    def check_close_condition(self) -> bool:
        if self.m_update_date >= self.m_last_hold_date:
            return True

        if self.m_direction > 0:
            self.m_h = max(self.m_h, self.m_nav)
            self.m_mdd = max(self.m_mdd, 1 - self.m_nav / self.m_h)
            if self.m_mdd >= self.m_mdd_cap:
                return True

        if self.m_direction < 0:
            self.m_l = min(self.m_l, self.m_nav)
            self.m_mdd = max(self.m_mdd, 1 - self.m_l / self.m_nav)
            if self.m_mdd >= self.m_mdd_cap:
                return True

        return False


class CPortfolio(object):
    def __init__(self, t_pid: str, t_groups_n: int, t_direction: int, t_init_cash: float, t_cost_rate: float,
                 t_md_dir: str, t_signal_dir: str, t_save_dir: str,
                 t_cne_calendar: CCalendar, t_simu_bgn_date: str, t_simu_stp_date: str,
                 t_stop_return: float = 9.90,
                 t_verbose: bool = False):
        """

        :param t_pid: portfolio ID
        :param t_groups_n: number of subgroup
        :param t_direction: 1 for long, -1 for short
        :param t_init_cash: available cash at the beginning
        :param t_cost_rate: single
        :param t_md_dir: market data directory
        :param t_signal_dir: signals directory
        :param t_save_dir: directory for saving results
        :param t_cne_calendar: CCalendar
        :param t_simu_bgn_date: begin date for simulation
        :param t_simu_stp_date: stop(not included) date for simulation
        :param t_stop_return: rise stop or fall stop, which should be multiplied by RETURN_SCALE already, i.e. 10 for 10%
        :param t_verbose: if true, details would be print
        """
        self.m_pid: str = t_pid
        self.m_max_groups_n: int = t_groups_n
        self.m_available_groups_n: int = t_groups_n
        self.m_available_groups_id: int = 0
        self.m_direction: int = t_direction
        self.m_mgr_tid: Dict[int, List[int]] = {z: [] for z in range(t_groups_n)}

        self.m_active_trades_n: int = 0
        self.m_active_trades_manager: Dict[int, CTradeL1] = {}  # tid |-> CTradeL1
        self.m_tid: int = 0

        self.m_update_date: str = ""
        self.m_signal_date: str = ""
        self.m_init_cash: float = t_init_cash
        self.m_unrealized_pnl_this_day: float = 0
        self.m_realized_pnl_this_day: float = 0
        self.m_realized_pnl_cumsum: float = 0
        self.m_nav: float = self.m_init_cash + self.m_realized_pnl_cumsum + self.m_unrealized_pnl_this_day
        self.m_available_cash: float = t_init_cash
        self.m_mkt_val_tot: float = 0
        self.m_trades_this_day = []

        self.m_signal_list: List[str] = []
        self.m_cost_rate: float = t_cost_rate

        self.m_md_dir: str = t_md_dir
        self.m_signal_dir: str = t_signal_dir
        self.m_save_dir: str = t_save_dir

        self.m_nav_data_list = []
        self.m_summary_df = None

        self.m_simu_bgn_date: str = t_simu_bgn_date
        self.m_simu_stp_date: str = t_simu_stp_date
        self.m_simu_end_date: str = t_cne_calendar.get_next_date(t_this_date=t_simu_stp_date, t_shift=-1)
        self.m_stop_return: float = t_stop_return
        self.m_verbose: bool = t_verbose

    def initialize(self, t_update_date: str, t_signal_date: str):
        self.m_update_date = t_update_date
        self.m_signal_date = t_signal_date
        for tid, trade in self.m_active_trades_manager.items():
            trade.initialize_at_beginning_of_each_day()
        self.m_unrealized_pnl_this_day = 0
        self.m_realized_pnl_this_day = 0
        self.m_trades_this_day = []
        self.m_mkt_val_tot = 0
        return 0

    def load_signal(self, t_sid_lbl: str) -> bool:
        """
        this function should be called only when initialized
        :return:
        """
        signal_file = "sig.{}.exe.{}.csv.gz".format(self.m_signal_date, self.m_update_date)
        signal_path = os.path.join(self.m_signal_dir, signal_file)
        if os.path.exists(signal_path):
            signal_df = pd.read_csv(signal_path)
            self.m_signal_list = signal_df[t_sid_lbl].to_list()
            return True
        else:
            self.m_signal_list = []
            return False

    def open_new_trades(self, t_direction: int, t_last_hold_date: str):
        """
        this function should be called only when load_signal return True
        """

        if self.m_available_groups_n > 0:
            amt_for_each_trade = self.m_available_cash / self.m_available_groups_n / len(self.m_signal_list)
            for sid in self.m_signal_list:
                new_trade = CTradeL1(t_sid=sid, t_md_dir=self.m_md_dir, t_cost_rate=self.m_cost_rate, t_stop_return=self.m_stop_return)
                if new_trade.open(
                        t_open_date=self.m_update_date,
                        t_allocated_amt=amt_for_each_trade,
                        t_direction=t_direction,
                        t_prev_date=self.m_signal_date,
                ):
                    new_trade.set_close_condition(t_last_hold_date=t_last_hold_date, t_simu_end_date=self.m_simu_end_date)
                    self.m_active_trades_manager[self.m_tid] = new_trade
                    cost, cash_locked = new_trade.get_cash_locked()
                    self.m_active_trades_n += 1
                    self.m_available_cash -= (cost + cash_locked)
                    self.m_realized_pnl_this_day -= cost
                    self.m_trades_this_day.append({
                        "sid": new_trade.m_sid,
                        "direction": new_trade.m_direction,
                        "operation": "open",
                        "quantity": new_trade.m_quantity,
                        "price": new_trade.m_open_price,
                        "realized_pnl": 0,
                        "cost": cost,
                    })

                    # new trade created successful, tid increase by 1
                    self.m_mgr_tid[self.m_available_groups_id].append(self.m_tid)
                    self.m_tid += 1
                else:
                    if self.m_verbose:
                        print("| {} | {} | fail to open  {:>5s} |".format(self.m_update_date, sid, "long" if t_direction > 0 else "short"))

            if len(self.m_mgr_tid[self.m_available_groups_id]) > 0:
                self.m_available_groups_n -= 1
                self.m_available_groups_id = -1

        return 0

    def update(self):
        for tid, trade in self.m_active_trades_manager.items():
            trade.update(t_update_date=self.m_update_date)
        return 0

    def close(self):
        to_be_removed_list = []
        for tid, trade in self.m_active_trades_manager.items():
            if trade.check_close_condition():
                if trade.close():
                    to_be_removed_list.append(tid)
                else:
                    if self.m_verbose:
                        print("| {} | {} | fail to close {:>5s} |".format(self.m_update_date, trade.m_sid, "long" if trade.m_direction > 0 else "short"))

        for tid in to_be_removed_list:
            close_trade = self.m_active_trades_manager[tid]
            cost, cash_unlocked = close_trade.get_cash_unlocked()
            self.m_active_trades_n -= 1
            self.m_available_cash += (cash_unlocked - cost)
            self.m_realized_pnl_this_day -= cost
            self.m_realized_pnl_this_day += close_trade.get_realized_pnl()
            self.m_trades_this_day.append({
                "sid": close_trade.m_sid,
                "direction": close_trade.m_direction,
                "operation": "close",
                "quantity": close_trade.m_quantity,
                "price": close_trade.m_close_price,
                "realized_pnl": close_trade.get_realized_pnl(),
                "cost": cost,
            })
            del self.m_active_trades_manager[tid]

            for gid in self.m_mgr_tid:
                if tid in self.m_mgr_tid[gid]:
                    self.m_mgr_tid[gid] = []  # if at least one trades in this group is closed, this group will be seen as available
                    self.m_available_groups_n += 1

        for gid in self.m_mgr_tid:
            if len(self.m_mgr_tid[gid]) == 0:
                self.m_available_groups_id = gid
        return 0

    def clearing_unrealized(self):
        for tid, trade in self.m_active_trades_manager.items():
            self.m_unrealized_pnl_this_day += trade.get_unrealized_pnl()
        self.m_realized_pnl_cumsum += self.m_realized_pnl_this_day
        self.m_nav = self.m_init_cash + self.m_realized_pnl_cumsum + self.m_unrealized_pnl_this_day
        return 0

    def take_positions_snapshots(self) -> int:
        pos_list = []
        for tid, trade in self.m_active_trades_manager.items():
            pos_list.append(trade.to_dict())
        if len(pos_list) > 0:
            pos_df = pd.DataFrame(pos_list)
            pos_file = "{}.position.csv.gz".format(self.m_update_date)
            pos_path = os.path.join(self.m_save_dir, "positions", pos_file)
            pos_df.to_csv(pos_path, index=False, float_format="%.4f")
            self.m_mkt_val_tot = pos_df["mkt_val"].sum()
        return 0

    def take_trades_records(self):
        if len(self.m_trades_this_day) > 0:
            trades_df = pd.DataFrame(self.m_trades_this_day)
            trades_file = "{}.trades.csv.gz".format(self.m_update_date)
            trades_path = os.path.join(self.m_save_dir, "trades", trades_file)
            trades_df.to_csv(trades_path, index=False, float_format="%.4f")
        return 0

    def to_dict(self):
        return {
            "trade_date": self.m_update_date,
            "init": self.m_init_cash,
            "pnl_realized_today": self.m_realized_pnl_this_day,
            "pnl_realized_cumsum": self.m_realized_pnl_cumsum,
            "pnl_unrealized": self.m_unrealized_pnl_this_day,
            "pnl_nav": self.m_nav,
            "position_n": self.m_active_trades_n,
            "position_mkt_val": self.m_mkt_val_tot,
            "available_group_n": self.m_available_groups_n,
            "available_cash": self.m_available_cash,
        }

    def update_nav_data(self):
        self.m_nav_data_list.append(self.to_dict())
        return 0

    def dir_preparation(self):
        check_and_mkdir(self.m_save_dir)
        check_and_mkdir(os.path.join(self.m_save_dir, "trades"))
        check_and_mkdir(os.path.join(self.m_save_dir, "positions"))
        remove_files_in_the_dir(os.path.join(self.m_save_dir, "trades"))
        remove_files_in_the_dir(os.path.join(self.m_save_dir, "positions"))
        return 0

    def main_loop(self, t_cne_calendar: CCalendar):
        self.dir_preparation()

        # core loop
        for trade_date in t_cne_calendar.get_iter_list(t_bgn_date=self.m_simu_bgn_date, t_stp_date=self.m_simu_stp_date, t_ascending=True):
            # set signal date and close date
            sig_date = t_cne_calendar.get_next_date(t_this_date=trade_date, t_shift=-1)
            last_hold_date = t_cne_calendar.get_next_date(t_this_date=trade_date, t_shift=self.m_max_groups_n - 1)

            # routine actions
            self.initialize(t_update_date=trade_date, t_signal_date=sig_date)
            if self.load_signal(t_sid_lbl="sid"):
                self.open_new_trades(t_last_hold_date=last_hold_date, t_direction=self.m_direction)
            self.update()
            self.close()
            self.clearing_unrealized()

            # save important info
            self.take_positions_snapshots()
            self.take_trades_records()
            self.update_nav_data()
        return 0

    def summary(self):
        self.m_summary_df = pd.DataFrame(self.m_nav_data_list)
        self.m_summary_df["navps"] = self.m_summary_df["pnl_nav"] / self.m_init_cash
        res_file = "complex_simu.{}.summary.csv".format(self.m_pid)
        res_path = os.path.join(self.m_save_dir, res_file)
        self.m_summary_df.to_csv(res_path, index=False, float_format="%.4f")
        return self.m_summary_df

    def plot_nav(self, t_ticks_num: int = 10, t_tick_label_size: int = 12, t_line_width: float = 2.0, t_twinx_plot: bool = False):
        fig0, ax0 = plt.subplots(figsize=(16, 9))
        navps = self.m_summary_df["navps"]
        navps.plot(ax=ax0, lw=t_line_width, style="b-")
        if t_twinx_plot:
            ret_sum = (navps / navps.shift(1).fillna(method="bfill") - 1).cumsum() + 1
            ax1 = ax0.twinx()
            ret_sum.plot(ax=ax1, lw=t_line_width, style="r-.")
        n_ticks = len(self.m_summary_df)
        xticks = np.arange(0, n_ticks, int(n_ticks / t_ticks_num))
        xticklabels = self.m_summary_df.loc[xticks, "trade_date"]
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels)
        ax0.tick_params(axis="both", labelsize=t_tick_label_size)
        fig0_file = "complex_simu.{}.nav.pdf".format(self.m_pid)
        fig0_path = os.path.join(self.m_save_dir, fig0_file)
        fig0.savefig(fig0_path, bbox_inches="tight")
        plt.close(fig0)
        return 0
