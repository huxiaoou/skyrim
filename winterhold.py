import os
import datetime as dt


def check_and_mkdir(t_path):
    if not os.path.exists(t_path):
        os.mkdir(t_path)
        return 1
    else:
        return 0


def remove_files_in_the_dir(t_path):
    for f in os.listdir(t_path):
        os.remove(os.path.join(t_path, f))
    return 0


def timer(func):
    def inner(**kwargs):
        t1 = dt.datetime.now()
        res = func(**kwargs)
        t2 = dt.datetime.now()
        print("... bgn @ {0} for {1}".format(t1, func.__name__))
        print("... end @ {0} for {1}".format(t2, func.__name__))
        print("... time consuming: {} seconds".format((t2-t1).total_seconds()))
        print("\n")
        return res
    return inner


def date_format_converter_08_to_10(t_date: str):
    # "202100101" -> "2021-01-01"
    return t_date[0:4] + "-" + t_date[4:6] + "-" + t_date[6:8]


def date_format_converter_10_to_08(t_date: str):
    # "20210-01-01" -> "20210101"
    return t_date.replace("-", "")
