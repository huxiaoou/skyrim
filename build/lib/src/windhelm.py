import os
import pandas as pd
from whiterun import parse_instrument_from_contract_wind

"""
created @ 2022-06-23
0.  add functions and classes which are most frequently used by Xuntou Trading System   
"""


def get_major_contract(t_instrument_id: str, t_sig_date: str, t_major_minor_dir: str, t_major_minor_file: str) -> str:
    _major_minor_path = os.path.join(t_major_minor_dir, t_major_minor_file.format(t_instrument_id))
    _major_minor_df = pd.read_csv(_major_minor_path, dtype={"trade_date": str}).set_index("trade_date")
    return _major_minor_df.at[t_sig_date, "n_contract"]


def get_contract_price(t_contract: str, t_sig_date: str, t_md_dir: str, t_price_type: str = "close") -> float:
    instrument_id = parse_instrument_from_contract_wind(t_contract)
    signal_date_md_file = "{}.cnf.{}.md.csv.gz".format(t_sig_date, instrument_id)
    signal_date_md_path = os.path.join(t_md_dir, t_sig_date[0:4], t_sig_date, signal_date_md_file)
    signal_date_md_df = pd.read_csv(signal_date_md_path).set_index("contract")
    return signal_date_md_df.at[t_contract, t_price_type]


def convert_mkt_code(x: str):
    if x == "SHF":
        return "SHFE"
    if x == "DCE":
        return "DCE"
    if x == "CZC":
        return "CZCE"
    return ""


def convert_contract_code(x: pd.Series):
    if x["市场"] == "CZCE":
        return x["代码"]
    else:
        return x["代码"].lower()


def split_quantity(t_qty: int, t_n_batch: int) -> list:
    """

    :param t_qty: quantity to be split
    :param t_n_batch: number of batches.
    :return: examples: (8, 3)->(3, 3, 2), (4, 2)->(2, 2), (6, 4)->(2, 2, 1, 1)
    """

    # qty = m * n + r
    r = t_qty % t_n_batch
    m = t_qty // t_n_batch
    res = [m] * t_n_batch
    for i in range(r):
        res[i] += 1
    return res


def split_xuntou_instruction(t_src_path: str, t_n_batch: int) -> int:
    if os.path.exists(t_src_path):
        src_df = pd.read_csv(t_src_path)
        # split quantity
        split_data = {}
        for idx, quantity in zip(src_df.index, src_df["数量"]):
            split_data[idx] = split_quantity(t_qty=quantity, t_n_batch=t_n_batch)
        split_df = pd.DataFrame(split_data)
        # create batch instruction
        for bi in range(t_n_batch):
            batch_df = src_df.copy()
            batch_df["数量"] = split_df.loc[bi]
            batch_path = t_src_path.replace(".csv", "_BATCH{:02d}.csv".format(bi))
            batch_df.to_csv(batch_path, index=False)
    else:
        print("{} does not exist, program will skip this split operation.".format(t_src_path))
    return 0


class CPosCell(object):
    def __init__(self, t_key: (str, int), t_quantity: int):
        self.m_key = t_key
        self.m_contract: str = t_key[0]
        self.m_direction: int = t_key[1]
        self.m_quantity: int = t_quantity

    def cal_operation(self, t_target_pos_cell: "CPosCell"):
        if self.m_key != t_target_pos_cell.m_key:
            print("Error! Keys does not match")
        else:
            dlt_qty = t_target_pos_cell.m_quantity - self.m_quantity
            if dlt_qty < 0:
                if self.m_direction > 0:
                    return {"operation": "SellClose", "contract": self.m_contract, "qty": -dlt_qty}
                else:
                    return {"operation": "BuyClose", "contract": self.m_contract, "qty": -dlt_qty}
            elif dlt_qty > 0:
                if self.m_direction > 0:
                    return {"operation": "BuyOpen", "contract": self.m_contract, "qty": dlt_qty}
                else:
                    return {"operation": "SellOpen", "contract": self.m_contract, "qty": dlt_qty}
            else:
                return None
