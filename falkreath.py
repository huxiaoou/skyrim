import os
import pandas as pd
import sqlite3 as sql3
from typing import Dict, List
import datetime as dt


class CTable(object):
    def __init__(self, t_table_name: str, t_primary_keys: Dict[str, str], t_value_columns: Dict[str, str]):
        self.m_table_name: str = t_table_name
        self.m_primary_keys: Dict[str, str] = t_primary_keys
        self.m_value_columns: Dict[str, str] = t_value_columns
        self.m_vars = list(self.m_primary_keys.keys()) + list(self.m_value_columns.keys())
        self.m_vars_n = len(self.m_vars)

        # cmd for update
        str_columns = ", ".join(self.m_vars)
        str_args = ", ".join(["?"] * self.m_vars_n)
        self.m_cmd_sql_update_template = "INSERT OR REPLACE INTO {} (" + str_columns + ") values(" + str_args + ")"


class CLib1Tab1(object):
    def __init__(self, t_lib_name: str, t_tab: CTable):
        self.m_lib_name: str = t_lib_name
        self.m_tab: CTable = t_tab


class CMangerLibBase(object):
    def __init__(self, t_db_save_dir: str, t_db_name: str):
        self.m_db_save_dir: str = t_db_save_dir
        self.m_db_name: str = t_db_name
        self.m_db_path: str = os.path.join(t_db_save_dir, t_db_name)
        self.m_connection = sql3.connect(self.m_db_path)
        self.m_cursor = self.m_connection.cursor()
        self.m_manager_table: Dict[str, CTable] = {}

    def close(self):
        self.m_connection.commit()
        self.m_cursor.close()
        self.m_connection.close()
        return 0

    def is_table_existence(self, t_table_name: str) -> bool:
        cmd_sql_check_existence = "SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{}'".format(t_table_name)
        table_counts = self.m_cursor.execute(cmd_sql_check_existence).fetchall()[0][0]
        if table_counts == 0:
            return False
        else:
            return True


class CManagerLibReader(CMangerLibBase):
    def read_by_date(self, t_table_name: str, t_trade_date: str, t_value_columns: List[str]):
        str_value_columns = ", ".join(t_value_columns)
        cmd_sql_for_inquiry = "SELECT {} FROM {} where trade_date = {}".format(str_value_columns, t_table_name, t_trade_date)
        rows = self.m_cursor.execute(cmd_sql_for_inquiry).fetchall()
        t_df = pd.DataFrame(data=rows, columns=t_value_columns)
        return t_df

    def read_by_instrument(self, t_table_name: str, t_instrument: str, t_value_columns: List[str]):
        str_value_columns = ", ".join(t_value_columns)
        cmd_sql_for_inquiry = "SELECT {} FROM {} where instrument = {}".format(str_value_columns, t_table_name, t_instrument)
        rows = self.m_cursor.execute(cmd_sql_for_inquiry).fetchall()
        t_df = pd.DataFrame(data=rows, columns=t_value_columns)
        return t_df

    def read_by_factor(self, t_table_name: str, t_factor: str, t_value_columns: List[str]):
        str_value_columns = ", ".join(t_value_columns)
        cmd_sql_for_inquiry = "SELECT {} FROM {} where factor = {}".format(str_value_columns, t_table_name, t_factor)
        rows = self.m_cursor.execute(cmd_sql_for_inquiry).fetchall()
        t_df = pd.DataFrame(data=rows, columns=t_value_columns)
        return t_df


class CManagerLibWriter(CManagerLibReader):
    def remove_table(self, t_table_name: str):
        self.m_cursor.execute("DROP TABLE {}".format(t_table_name))
        return 0

    def add_table(self, t_table: CTable):
        """

        :param t_table: "EXPOSURE"
        :return:
        """
        self.m_manager_table[t_table.m_table_name] = t_table

        # remove old table
        if self.is_table_existence(t_table.m_table_name):
            print("... Table {} is in database {} already".format(t_table.m_table_name, self.m_db_name))
            self.remove_table(t_table.m_table_name)
            print("... Table {} is removed from database {}".format(t_table.m_table_name, self.m_db_name))

        str_primary_keys = ["{} {}".format(k, v) for k, v in t_table.m_primary_keys.items()]
        str_value_columns = ["{} {}".format(k, v) for k, v in t_table.m_value_columns.items()]
        str_all_columns = ", ".join(str_primary_keys + str_value_columns)
        str_set_primary = "PRIMARY KEY({})".format(", ".join(list(t_table.m_primary_keys.keys())))
        cmd_sql_for_create_table = "CREATE TABLE IF NOT EXISTS {}({}, {})".format(
            t_table.m_table_name,
            str_all_columns,
            str_set_primary
        )
        self.m_cursor.execute(cmd_sql_for_create_table)
        print("... Table {} is added to {} as a new table".format(t_table.m_table_name, self.m_db_name))
        return 0

    def update(self, t_table_name: str, t_update_df: pd.DataFrame, t_using_index: bool = False):
        """

        :param t_table_name: the table to be updated
        :param t_update_df: new data, column orders must be the same as the columns orders of the new target table
        :param t_using_index: whether using index as a data column
        :return:
        """
        cmd_sql_update = self.m_manager_table[t_table_name].m_cmd_sql_update_template.format(t_table_name)
        for data_cell in t_update_df.itertuples(index=t_using_index):  # itertuples is much faster than iterrows
            self.m_cursor.execute(cmd_sql_update, data_cell)
        return 0


class CManagerLibWriterByDate(CManagerLibWriter):
    def update_by_date(self, t_table_name: str, t_date: str, t_update_df: pd.DataFrame, t_using_index: bool = False):
        """

        :param t_table_name: the table to be updated
        :param t_date: this class would treat date as one of the primary keys
        :param t_update_df: new data, column orders must be the same as the columns orders of the new target table
        :param t_using_index: whether using index as a data column
        :return:
        """
        cmd_sql_update = self.m_manager_table[t_table_name].m_cmd_sql_update_template.format(t_table_name)
        for data_cell in t_update_df.itertuples(index=t_using_index):  # itertuples is much faster than iterrows
            self.m_cursor.execute(cmd_sql_update, (t_date,) + data_cell)
        return 0

    def save_factor_by_date(self, t_table_name: str, t_all_factor_df: pd.DataFrame, t_bgn_date: str, t_stp_date: str):
        for _trade_date, _trade_date_df in t_all_factor_df.groupby(by="trade_date"):
            # noinspection PyTypeChecker
            if (_trade_date < t_bgn_date) or (_trade_date >= t_stp_date):
                continue
            _factor_df = _trade_date_df.rename(mapper={_trade_date: "value"}).T.dropna(axis=0)
            if len(_factor_df) > 0:
                # noinspection PyTypeChecker
                self.update_by_date(
                    t_table_name=t_table_name,
                    t_date=_trade_date,
                    t_update_df=_factor_df,
                    t_using_index=True
                )
        return 0


if __name__ == "__main__":
    import numpy as np

    print("The following scripts are used for TESTING ONLY")
    table_name = "EXPOSURE4"
    n = 100000
    update_df = pd.DataFrame({
        "trade_date": ["{:08d}".format(i) for i in range(3 * n)],
        "instrument": ["CU.SHF", "Y.DCE", "MA.CZC"] * n,
        "RSW": np.random.random(3 * n),
        "BASIS": np.random.random(3 * n),
    })
    print(update_df)

    # --- lib test
    mylib = CManagerLibWriter(t_db_name="test.db", t_db_save_dir="E://TMP")
    mylib.add_table(t_table=CTable(
        t_table_name=table_name,
        t_primary_keys={"trade_date": "TEXT", "instrument": "TEXT"},
        t_value_columns={"RSW": "REAL", "BASIS": "REAL"}
    ))

    t0 = dt.datetime.now()
    mylib.update(
        t_table_name=table_name,
        t_update_df=update_df,
    )
    t1 = dt.datetime.now()
    print("... time consuming {:.2f} seconds".format((t1 - t0).total_seconds()))

    mylib.close()
