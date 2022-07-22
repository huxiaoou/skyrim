import numpy as np
import cvxpy as cvp
import pandas as pd
import scipy.optimize
from scipy.optimize import minimize
from scipy.sparse.linalg import ArpackNoConvergence

"""
created @ 2021-06-16
0.  provide algorithm and functions to solve assets allocation problem
updated @ 2021-09-19
"""


def portfolio_variance(t_w: np.ndarray, t_sigma: np.ndarray) -> float:
    return t_w.dot(t_sigma).dot(t_w)


def jac_var(t_w: np.ndarray, t_sigma: np.ndarray) -> np.ndarray:
    return 2 * t_sigma.dot(t_w)


def portfolio_std(t_w: np.ndarray, t_sigma: np.ndarray) -> float:
    return np.sqrt(t_w.dot(t_sigma).dot(t_w))


def portfolio_return(t_w: np.ndarray, t_mu: np.ndarray) -> float:
    return t_w.dot(t_mu)


def jac_ret(t_w: np.ndarray, t_mu: np.ndarray) -> np.ndarray:  # d(kx) / dx = k, constant k is irrelevant to the x
    return t_mu


def portfolio_return_mirror(t_w: np.ndarray, t_mu: np.ndarray) -> float:
    return -t_w.dot(t_mu)


def portfolio_utility(t_w: np.ndarray, t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float) -> float:
    # u = -2 w*m/l +wSw  <=> v = w*m - l/2 * wSw
    return -2 * portfolio_return(t_w, t_mu) / t_lbd + portfolio_variance(t_w, t_sigma)


def portfolio_risk_budget(t_w: np.ndarray, t_sigma: np.ndarray, t_rb: np.ndarray, t_verbose: bool) -> float:
    _p, _ = t_sigma.shape
    _s = np.sqrt(portfolio_variance(t_w, t_sigma))
    _mrc = t_sigma.dot(t_w) / _s  # marginal risk cost
    _rc = t_w * _mrc  # risk cost
    _x = _rc / t_rb
    if t_verbose:
        print("RB  = {}".format(t_rb))
        print("MRC = {}".format(_mrc))
        print("RC  = {}".format(_rc))
        print("X   = {}".format(_x))
    return 2 * (_p ** 2) * np.var(_x, ddof=0)


# -------------------------------- Optimize Algorithm --------------------------------
# --- type 0: minimum variance
# target: min_{w} wSw  with w1 = 1
def minimize_variance(t_sigma: np.ndarray) -> (np.ndarray, float):
    """
    :param t_sigma: the covariance of available assets
    :return: weight of the portfolio with minimum variance
    """
    _p, _ = t_sigma.shape
    _u = np.ones(_p)
    _h = np.linalg.inv(t_sigma).dot(_u)
    _w = _h / (_u.dot(_h))
    _min_var = portfolio_variance(t_w=_w, t_sigma=t_sigma)
    return _w, _min_var


def minimize_variance_con(t_sigma: np.ndarray) -> (np.ndarray, float):
    """
    :param t_sigma: the covariance of available assets
    :return: weight of the portfolio with minimum variance and 0 <= x <= 1
    """
    _p, _ = t_sigma.shape
    _res = minimize(
        fun=portfolio_variance,
        x0=np.ones(_p) / _p,
        args=(t_sigma,),
        bounds=[(0, 1)] * _p,
        constraints={"type": "eq", "fun": lambda z: z.sum() - 1}
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


def minimize_variance_con2(t_sigma: np.ndarray) -> (np.ndarray, float):
    """
    :param t_sigma: the covariance of available assets
    :return: weight of the portfolio with minimum variance and -1 <= x <= 1, leverage is allowed, such as weight = [-1, 1, 1]
    """
    _p, _ = t_sigma.shape
    _res = minimize(
        fun=portfolio_variance,
        x0=np.ones(_p) / _p,
        args=(t_sigma,),
        bounds=[(-1, 1)] * _p,  # allow short
        constraints={"type": "eq", "fun": lambda z: z.sum() - 1},
        # constraints=scipy.optimize.LinearConstraint(np.ones(_p), -1, 1), # can not be, or the result would be ZERO vector
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


# --- type 1: minimum utility
# target: min_{w} -2/lambda * wm + wSw  with w1 = 1
def minimize_utility(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd) -> (np.ndarray, float):
    _p, _ = t_sigma.shape
    _u = np.ones(_p)
    _sigma_inv = np.linalg.inv(t_sigma)
    _m = _sigma_inv.dot(t_mu)
    _h = _sigma_inv.dot(_u)
    _d = 2 * (1 - 1 / t_lbd * _u.dot(_m)) / (_u.dot(_h))
    _w = 1 / t_lbd * _m + _d / 2 * _h
    _uty = portfolio_utility(_w, t_mu, t_sigma, t_lbd)
    return _w, _uty


def minimize_utility_con(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float, t_bounds: tuple = (0, 1)) -> (np.ndarray, float):
    """

    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :param t_bounds:
    :return: weight of the portfolio with maximum utility and x in bounds. Leverage is not allowed if default value for bounds are used.
    """
    _p, _ = t_sigma.shape
    _res = minimize(
        fun=portfolio_utility,
        x0=np.ones(_p) / _p,
        args=(t_mu, t_sigma, t_lbd),
        bounds=[t_bounds] * _p,
        constraints={"type": "eq", "fun": lambda z: z.sum() - 1}
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


def minimize_utility_con2(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float) -> (np.ndarray, float):
    """

    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :return: weight of the portfolio with maximum utility and x in bounds, leverage is allowed, such as weight = [-1, 1, 1]
    """
    _p, _ = t_sigma.shape
    # noinspection PyTypeChecker
    _res = minimize(
        fun=portfolio_utility,
        x0=np.ones(_p) / _p,
        args=(t_mu, t_sigma, t_lbd),
        bounds=[(-1, 1)] * _p,  # allow short
        constraints=scipy.optimize.LinearConstraint(np.ones(_p), -1, 1),  # control total market value

    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


def minimize_utility_con3(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float,
                          t_bound: float, t_max_iter: int = 50000) -> (np.ndarray, float):
    """

    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :param t_bound:
    :param t_max_iter:
    :return: weight of the portfolio with maximum utility and x in bounds, leverage is not allowed, 0<= sum(abs(weight)) <=1
    """
    _p, _ = t_sigma.shape
    # noinspection PyTypeChecker
    _res = minimize(
        fun=portfolio_utility,
        x0=np.ones(_p) / _p,
        args=(t_mu, t_sigma, t_lbd),
        bounds=[(-t_bound, t_bound)] * _p,  # allow short
        constraints=scipy.optimize.NonlinearConstraint(lambda z: np.sum(np.abs(z)), 0, 1),  # control total market value
        options={"maxiter": t_max_iter}
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


def minimize_utility_con4(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float,
                          t_sec: np.ndarray, t_sec_l_bound: np.ndarray, t_sec_r_bound: np.ndarray, t_max_iter: int = 50000) -> (np.ndarray, float):
    """

    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :param t_sec:
    :param t_sec_l_bound:
    :param t_sec_r_bound:
    :param t_max_iter:
    :return: weight of the portfolio with maximum utility, leverage is not allowed, 0<= sum(abs(weight)) <=1
             risk exposure control are also applied.
    """
    _p, _ = t_sigma.shape
    _a = np.vstack([np.ones(_p), t_sec])
    _lb = np.concatenate(([0], t_sec_l_bound))
    _rb = np.concatenate(([1], t_sec_r_bound))
    # noinspection PyTypeChecker
    _res = minimize(
        fun=portfolio_utility,
        x0=np.ones(_p) / _p,
        args=(t_mu, t_sigma, t_lbd),
        constraints=scipy.optimize.NonlinearConstraint(lambda z: _a.dot(np.abs(z)), _lb, _rb),  # control total market value
        options={"maxiter": t_max_iter}
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


def minimize_utility_con5(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float,
                          t_bound, t_sec: np.ndarray, t_sec_l_bound: np.ndarray, t_sec_r_bound: np.ndarray,
                          t_max_iter: int = 50000) -> (np.ndarray, float):
    """

    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :param t_bound:
    :param t_sec:
    :param t_sec_l_bound:
    :param t_sec_r_bound:
    :param t_max_iter:
    :return: weight of the portfolio with maximum utility and x in bounds, leverage is not allowed, 0<= sum(abs(weight)) <=1
             risk exposure control are also applied.
             constraints are just divide constraints into two parts: linear and non-linear
    """
    _p, _ = t_sigma.shape
    # noinspection PyTypeChecker
    _res = minimize(
        fun=portfolio_utility,
        x0=np.ones(_p) / _p,
        args=(t_mu, t_sigma, t_lbd),
        bounds=[(-t_bound, t_bound)] * _p,  # allow short
        constraints=[
            scipy.optimize.NonlinearConstraint(lambda z: np.sum(np.abs(z)), 0, 1),
            scipy.optimize.LinearConstraint(t_sec, t_sec_l_bound, t_sec_r_bound),
        ],  # control total market value
        options={"maxiter": t_max_iter}
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


def minimize_utility_con6(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float,
                          t_bound: tuple, t_sec: np.ndarray, t_sec_bound: np.ndarray,
                          t_max_iter: int = 50000) -> (np.ndarray, float):
    """
    Created @ 2021-12-22 For E:\\Works\\2021\\Project_2021_12_Commodity_Allocation_With_Risk_Model_V0
    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :param t_bound:
    :param t_sec:
    :param t_sec_bound:
    :param t_max_iter:
    :return: weight of the portfolio with maximum utility and x in bounds, leverage is allowed if lower bound is negative.
             bounds for each weight are given.
             risk exposure control are also applied.
             constraints are just divide constraints into two parts: linear and non-linear.
    """
    _p, _ = t_sigma.shape
    _a = np.vstack([np.ones(_p), t_sec])
    _lb = np.concatenate(([1], t_sec_bound))
    _rb = np.concatenate(([1], t_sec_bound))
    # noinspection PyTypeChecker
    _res = minimize(
        fun=portfolio_utility,
        x0=np.ones(_p) / _p,
        args=(t_mu, t_sigma, t_lbd),
        bounds=[t_bound] * _p,
        constraints=scipy.optimize.LinearConstraint(_a, _lb, _rb),  # control exposure at each sector
        options={"maxiter": t_max_iter}
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


def minimize_utility_con6_cvxpy(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float,
                                t_bound: tuple, t_sec: np.ndarray, t_sec_bound: np.ndarray,
                                t_solver: str = "ECOS",
                                t_max_iter_times: int = 20) -> (np.ndarray, float):
    """
    This function has the same interface as minimize_utility_con6, but its core is
    using cvxpy.

    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :param t_bound:
    :param t_sec:
    :param t_sec_bound:
    :param t_solver: frequently used solvers = ["ECOS", "OSQP", "SCS"], "SCS" solve all the problem
    :param t_max_iter_times:
    :return: weight of the portfolio with maximum utility and x in bounds, leverage is allowed if lower bound is negative.
             bounds for each weight are given.
             risk exposure control are also applied.
             constraints are just divide constraints into two parts: linear and non-linear.
    """

    _p, _ = t_sigma.shape
    _a = np.vstack([np.ones(_p), t_sec])
    _lb = np.concatenate(([1], t_sec_bound))
    _rb = np.concatenate(([1], t_sec_bound))

    _iter_times = 0
    while _iter_times < t_max_iter_times:
        try:
            _w = cvp.Variable(_p)
            _objective = cvp.Minimize(-2 / t_lbd * _w @ t_mu + cvp.quad_form(_w, t_sigma))
            # _constraints = [t_bound[0] <= _w, _w <= t_bound[1], _lb <= _a @ _w, _a @ _w <= _rb]
            # since _lb = _rb, the sentence above and below has the same effects
            _constraints = [t_bound[0] <= _w, _w <= t_bound[1], _a @ _w == _rb]
            _problem = cvp.Problem(_objective, _constraints)
            _problem.solve(solver=t_solver)
            if _problem.status == "optimal":
                _u = portfolio_utility(t_w=_w.value, t_mu=t_mu, t_sigma=t_sigma, t_lbd=t_lbd)
                return _w.value, _u
            else:
                _iter_times += 1
        except cvp.error.DCPError:
            # print("Function tried for {} time".format(_iter_times))
            # print("ERROR! Optimizer exits with a failure")
            # print("Problem does not follow DCP rules")
            _iter_times += 1
        except ArpackNoConvergence:
            # print("Function tried for {} time".format(_iter_times))
            # print("ERROR! Optimizer exits with a failure")
            # print("Arpack No Convergence Error")
            _iter_times += 1

    # print("Maximum iter times reached before an optimal solution is found.")
    return None, None


def minimize_utility_con7_cvxpy(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float,
                                t_bound: tuple, t_sec: np.ndarray, t_sec_bound: np.ndarray,
                                t_l_bound_offset: float, t_r_bound_offset: float,
                                t_solver: str = "ECOS",
                                t_max_iter_times: int = 20, t_verbose: bool = False) -> (np.ndarray, float):
    """
    Boundary offset are provided, which is the biggest difference between this function and minimize_utility_con6_cvxpy

    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :param t_bound:
    :param t_sec:
    :param t_sec_bound:
    :param t_l_bound_offset:
    :param t_r_bound_offset:
    :param t_solver: frequently used solvers = ["ECOS", "OSQP", "SCS"], "SCS" solve all the problem
    :param t_max_iter_times:
    :param t_verbose: whether to print error information
    :return: weight of the portfolio with maximum utility and x in bounds, leverage is not allowed, short is allowed.
             bounds for each weight are given.
             risk exposure control are also applied.
             constraints are just divide constraints into two parts: linear and non-linear.
    """

    _p, _ = t_sigma.shape
    _a = t_sec
    _lb = t_sec_bound + t_l_bound_offset
    _rb = t_sec_bound + t_r_bound_offset

    _iter_times = 0
    while _iter_times < t_max_iter_times:
        try:
            _w = cvp.Variable(_p)
            _objective = cvp.Minimize(-2 / t_lbd * _w @ t_mu + cvp.quad_form(_w, t_sigma))
            _constraints = [t_bound[0] <= _w, _w <= t_bound[1], cvp.sum(cvp.abs(_w)) <= 1, _lb <= _a @ _w, _a @ _w <= _rb]
            _problem = cvp.Problem(_objective, _constraints)
            _problem.solve(solver=t_solver)
            if _problem.status == "optimal":
                _u = portfolio_utility(t_w=_w.value, t_mu=t_mu, t_sigma=t_sigma, t_lbd=t_lbd)
                return _w.value, _u
            else:
                _iter_times += 1
        except cvp.error.DCPError:
            if t_verbose:
                print("Function tried for {} time".format(_iter_times))
                print("ERROR! Optimizer exits with a failure")
                print("Problem does not follow DCP rules")
            _iter_times += 1
        except ArpackNoConvergence:
            if t_verbose:
                print("Function tried for {} time".format(_iter_times))
                print("ERROR! Optimizer exits with a failure")
                print("Arpack No Convergence Error")
            _iter_times += 1

    # print("Maximum iter times reached before an optimal solution is found.")
    return None, None


def minimize_utility_con8_cvxpy(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float,
                                t_risk_factor_exposure: np.ndarray, t_ben_wgt: np.ndarray,
                                t_l_risk_exposure_offset: float, t_r_risk_exposure_offset: float,
                                t_constraint_type: int,
                                t_min_weight: float = 0,
                                t_solver: str = "ECOS",
                                t_max_iter_times: int = 20, t_verbose: bool = False) -> (np.ndarray, float):
    """
    Just another portfolio optimizer.

    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :param t_risk_factor_exposure: array with size = (number of risk factors, number of instruments)
    :param t_ben_wgt: array with size = (number of factors,), calculated from t_exposure_mat.dot(bench_weight_srs),
                             exposure of benchmark portfolio at each factor. number of factors = t_sec_num + t_sty_num
    :param t_l_risk_exposure_offset:  lower offset for sector exposure of optimal portfolio
    :param t_r_risk_exposure_offset:  higher offset for sector exposure of optimal portfolio
    :param t_constraint_type: 1 for long only, -1 for short allowed.
    :param t_min_weight: minimum weight
    :param t_solver: frequently used solvers = ["ECOS", "OSQP", "SCS"], "SCS" solve all the problem
    :param t_max_iter_times: maximum iteration times
    :param t_verbose: whether to print error information
    :return: weight of the portfolio with maximum utility and x in bounds, leverage is not allowed.
             short is allowed if t_constraint_type < 0.
             bounds for each weight are given.
             risk exposure control are also applied.
             constraints are just divide constraints into two parts: linear and non-linear.
    """

    _p, _ = t_sigma.shape
    _H = t_risk_factor_exposure
    _h = _H @ t_ben_wgt
    _lb = _h + t_l_risk_exposure_offset
    _rb = _h + t_r_risk_exposure_offset

    _iter_times = 0
    while _iter_times < t_max_iter_times:
        try:
            _w = cvp.Variable(_p)
            _objective = cvp.Minimize(-2 / t_lbd * t_mu @ _w + cvp.quad_form(_w, t_sigma))
            if t_constraint_type < 0:
                _constraints = [_H @ _w <= _rb, _H @ _w >= _lb, cvp.norm(_w, 1) <= 1]
            else:
                _constraints = [_H @ _w <= _rb, _H @ _w >= _lb, cvp.sum(_w) <= 1, _w >= t_min_weight]

            _problem = cvp.Problem(_objective, _constraints)
            _problem.solve(solver=t_solver)
            if _problem.status == "optimal":
                _u = portfolio_utility(t_w=_w.value, t_mu=t_mu, t_sigma=t_sigma, t_lbd=t_lbd)
                return _w.value, _u
            else:
                _iter_times += 1
        except cvp.error.DCPError:
            if t_verbose:
                print("Function tried for {} time".format(_iter_times))
                print("ERROR! Optimizer exits with a failure")
                print("Problem does not follow DCP rules")
            _iter_times += 1
        except ArpackNoConvergence:
            if t_verbose:
                print("Function tried for {} time".format(_iter_times))
                print("ERROR! Optimizer exits with a failure")
                print("Arpack No Convergence Error")
            _iter_times += 1

    if t_verbose:
        print("Maximum iter times reached before an optimal solution is found.")
    return None, None


def minimize_utility_con9_cvxpy(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float,
                                t_risk_factor_exposure: np.ndarray,
                                t_l_risk_exposure_offset: float, t_r_risk_exposure_offset: float,
                                t_weight_cap: float = 1,
                                t_solver: str = "ECOS",
                                t_max_iter_times: int = 20, t_verbose: bool = False) -> (np.ndarray, float):
    """
    Just another portfolio optimizer.

    :param t_mu:
    :param t_sigma:
    :param t_lbd:
    :param t_risk_factor_exposure: array with size = (number of risk factors, number of instruments)
    :param t_l_risk_exposure_offset:  lower offset for sector exposure of optimal portfolio
    :param t_r_risk_exposure_offset:  higher offset for sector exposure of optimal portfolio
    :param t_weight_cap: cap of absolute value of weight of each instrument
    :param t_solver: frequently used solvers = ["ECOS", "OSQP", "SCS"], "SCS" solve all the problem
    :param t_max_iter_times: maximum iteration times
    :param t_verbose: whether to print error information
    :return: weight of the portfolio with maximum utility and x in bounds, leverage is not allowed.
             short is allowed.
             bounds for each weight are given.
             risk exposure control are also applied.
             constraints are just divide constraints into two parts: linear and non-linear.
    """

    _p, _ = t_sigma.shape
    _num_fac, _num_ins = t_risk_factor_exposure.shape
    _H = t_risk_factor_exposure
    _lb = np.zeros(_num_fac) + t_l_risk_exposure_offset
    _rb = np.zeros(_num_fac) + t_r_risk_exposure_offset

    _iter_times = 0
    while _iter_times < t_max_iter_times:
        try:
            _w = cvp.Variable(_p)
            _objective = cvp.Minimize(-2 / t_lbd * t_mu @ _w + cvp.quad_form(_w, t_sigma))
            _constraints = [_H @ _w <= _rb, _H @ _w >= _lb, cvp.norm(_w, 1) <= 1, cvp.abs(_w) <= t_weight_cap]
            _problem = cvp.Problem(_objective, _constraints)
            _problem.solve(solver=t_solver)
            if _problem.status == "optimal":
                _u = portfolio_utility(t_w=_w.value, t_mu=t_mu, t_sigma=t_sigma, t_lbd=t_lbd)
                return _w.value, _u
            else:
                _iter_times += 1
        except cvp.error.DCPError:
            if t_verbose:
                print("Function tried for {} time".format(_iter_times))
                print("ERROR! Optimizer exits with a failure")
                print("Problem does not follow DCP rules")
            _iter_times += 1
        except ArpackNoConvergence:
            if t_verbose:
                print("Function tried for {} time".format(_iter_times))
                print("ERROR! Optimizer exits with a failure")
                print("Arpack No Convergence Error")
            _iter_times += 1

    if t_verbose:
        print("Maximum iter times reached before an optimal solution is found.")
    return None, None


def check_boundary(t_weight_df: pd.DataFrame, t_risk_factor_exposure: pd.DataFrame, t_verbose: bool):
    if t_verbose:
        print("-" * 80)
        print(t_weight_df)
        print("-" * 80)
        print(t_weight_df.abs().sum())
        print("-" * 80)
        print(t_weight_df.sum())
        print("-" * 80)
        print(t_risk_factor_exposure.dot(t_weight_df))
        print("-" * 80)
    return 0


def check_utility(t_w_opt: np.ndarray, t_w_ben: np.ndarray, t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float, t_verbose: bool):
    if t_verbose:
        _w = np.array([t_w_opt, t_w_ben])
        _mu_opt = t_w_opt.dot(t_mu)
        _mu_ben = t_w_ben.dot(t_mu)
        _sigma_opt = t_w_opt.dot(t_sigma).dot(t_w_opt)
        _sigma_ben = t_w_ben.dot(t_sigma).dot(t_w_ben)
        _uty_opt = _mu_opt - t_lbd * _sigma_opt / 2
        _uty_ben = _mu_ben - t_lbd * _sigma_ben / 2
        _res_df = pd.DataFrame({
            "mu": {"opt": _mu_opt, "ben": _mu_ben},
            "sigma": {"opt": _sigma_opt, "ben": _sigma_ben},
            "uty": {"opt": _uty_opt, "ben": _uty_ben},
        })
        print(_res_df)
    return 0


def minimize_utility_con_analytic(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float, t_H: np.ndarray, t_h: np.ndarray, t_F: np.ndarray, t_f: np.ndarray) -> (np.ndarray, float):
    """

    :param t_mu: the mean of assets returns, shape: P x 1
    :param t_sigma: the covariance of available assets, shape: P x P
    :param t_lbd: risk aversion coefficient, shape: 1 x 1
    :param t_H: equality confine left, shape: M x P, M = number of equality confine
    :param t_h: equality confine right, shape: M x 1
    :param t_F: inequality confine left, shape: N x P, N = number of inequality confine
    :param t_f: inequality confine right, shape: N x 1,
    :return: the optimal w'm- 2w'Sw/l with Hw=h, Fw <=f
    """

    """
    this function has some problems, can NOT be used yet.
    """
    _sigma_inv = np.linalg.inv(t_sigma)  # shape: p x p

    _z00 = t_H.dot(_sigma_inv).dot(t_H.T)  # shape: M x M
    _z01 = t_H.dot(_sigma_inv).dot(t_F.T)  # shape: M x N
    _z10 = t_F.dot(_sigma_inv).dot(t_H.T)  # shape: N x M
    _z11 = t_F.dot(_sigma_inv).dot(t_F.T)  # shape: N x N
    _z = np.concatenate([
        np.concatenate([_z00, _z01], axis=1),
        np.concatenate([_z10, _z11], axis=1),
    ], axis=0)  # shape: (M+N) x (M+N)

    _k0 = t_lbd * t_h - t_H.dot(_sigma_inv).dot(t_mu)  # shape: M x 1
    _k1 = t_lbd * t_f - t_F.dot(_sigma_inv).dot(t_mu)  # shape: N x 1
    _k = np.concatenate([_k0, _k1])  # shape: (M+N) x 1

    _z_inv = np.linalg.inv(_z)  # shape: (M+N) x (M+N)
    _alpha_beta = _z_inv.dot(_k)  # shape: (M+N) x 1
    _H_F = np.array([t_H.T, t_F.T])
    _w = _sigma_inv.dot(t_mu) / t_lbd + _sigma_inv.dot(_H_F).dot(_z_inv).dot(_alpha_beta) / t_lbd
    _uty = portfolio_utility(_w, t_mu, t_sigma, t_lbd)
    return _w, _uty


# --- type 2: risk parity/budget
# target: min_{w} sum_{i}(w_i(sigma*w)_i/s - s/n)
def minimize_risk_budget_con(t_sigma: np.ndarray, t_rb: np.ndarray, t_verbose) -> (np.ndarray, float):
    """
    :param t_sigma: the covariance of available assets
    :param t_rb: risk budget
    :param t_verbose: whether to print iteration details
    :return:
    """
    _p, _ = t_sigma.shape
    _res = minimize(
        fun=portfolio_risk_budget,
        x0=np.ones(_p) / _p,
        args=(t_sigma, t_rb, t_verbose),
        bounds=[(0, 1)] * _p,
        constraints={"type": "eq", "fun": lambda z: z.sum() - 1},
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


def minimize_risk_budget_con2(t_sigma: np.ndarray, t_rb: np.ndarray, t_verbose) -> (np.ndarray, float):
    """
    :param t_sigma: the covariance of available assets
    :param t_rb: risk budget
    :param t_verbose: whether to print iteration details
    :return:
    """
    _p, _ = t_sigma.shape
    # noinspection PyTypeChecker
    _res = minimize(
        fun=portfolio_risk_budget,
        x0=np.ones(_p) / _p,
        args=(t_sigma, t_rb, t_verbose),
        bounds=[(-1, 1)] * _p,  # allow short
        constraints=scipy.optimize.LinearConstraint(np.ones(_p), -1, 1),  # control total market value
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


# --- type 3: maximum return with confined variance
# may have convergence problem, can not be used yet.
def maximum_return_with_confined_variance(t_mu: np.ndarray, t_sigma: np.ndarray, t_benchmark_s2: float):
    """

    :param t_mu: the expected return of  available assets
    :param t_sigma: the covariance of available assets
    :param t_benchmark_s2: the expected variance of the portfolio
    :return:
    """
    _p, _ = t_sigma.shape
    # noinspection PyTypeChecker
    _res = minimize(
        fun=portfolio_return_mirror,
        x0=np.ones(_p) / _p,
        args=(t_mu,),
        method="SLSQP",
        jac=jac_ret,
        bounds=[(0, 1)] * _p,
        constraints=[
            {"type": "eq", "fun": lambda z: z.sum() - 1},
            {"type": "eq", "fun": lambda z: portfolio_variance(z, t_sigma) - t_benchmark_s2},
        ],
        options={"maxiter": 1000, "ftol": 0.1}
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None


# --- type 4: minimum variance with confined return
# may have convergence problem, can not be used yet.
def minimum_variance_with_confined_return(t_mu: np.ndarray, t_sigma: np.ndarray, t_benchmark_mu: float):
    """

    :param t_mu: the expected return of  available assets
    :param t_sigma: the covariance of available assets
    :param t_benchmark_mu: the expected return of the portfolio
    :return:
    """
    _p, _ = t_sigma.shape
    # noinspection PyTypeChecker
    _res = minimize(
        fun=portfolio_variance,
        x0=np.ones(_p) / _p,
        args=(t_sigma,),
        method="SLSQP",
        jac=jac_var,
        bounds=[(0, 1)] * _p,
        constraints=[
            {"type": "eq", "fun": lambda z: z.sum() - 1},
            {"type": "eq", "fun": lambda z: portfolio_return(z, t_mu) - t_benchmark_mu},
        ],
        options={"maxiter": 1000, "ftol": 0.1}
    )
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None
