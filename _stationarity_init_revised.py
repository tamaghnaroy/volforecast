"""
Stationarity Analysis Module for Time Series

Provides comprehensive stationarity testing including:
- Standard tests: ADF, KPSS, Phillips-Perron
- Structural break tests: Zivot-Andrews, Clemente-Montañés-Reyes, Lee-Strazicich
- Advanced/Rare tests: Variance Ratio, KPSS-based, Rank tests, SURADF
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.regression.linear_model import OLS
from scipy import stats
from scipy.stats import norm


def _try_import_arch():
    """Try to import arch package for Phillips-Perron test."""
    try:
        from arch.unitroot import PhillipsPerron
        return PhillipsPerron
    except ImportError:
        return None


class StationarityResult(Enum):
    STATIONARY = "stationary"
    NON_STATIONARY = "non_stationary"
    TREND_STATIONARY = "trend_stationary"
    MEAN_REVERTING = "mean_reverting"
    UNCERTAIN = "uncertain"


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    critical_values: Dict[str, float]
    conclusion: StationarityResult
    lags: Optional[int] = None
    used_lag: Optional[int] = None
    n_obs: Optional[int] = None
    regression: Optional[str] = None
    boot_iterations: Optional[int] = None
    structural_break_date: Optional[pd.Timestamp] = None
    break_type: Optional[str] = None
    additional_info: Optional[Dict] = None

    def is_significant(self, level: float = 0.05) -> bool:
        if self.p_value < level:
            return True
        return False

    def summary_dict(self) -> Dict:
        return {
            "test": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "critical_values": self.critical_values,
            "conclusion": self.conclusion.value,
            "significant_at_5pct": self.is_significant(0.05)
        }


def _ensure_array(ts: Union[pd.Series, pd.DataFrame, np.ndarray]) -> np.ndarray:
    if isinstance(ts, (pd.Series, pd.DataFrame)):
        if hasattr(ts, 'ndim') and ts.ndim > 1:
            raise ValueError(f"Input must be 1-D, got {ts.ndim} dimensions")
        arr = ts.dropna().values.flatten()
    else:
        arr = np.asarray(ts)
        if arr.ndim > 1:
            raise ValueError(f"Input must be 1-D, got {arr.ndim} dimensions")
        arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        raise ValueError("Input contains no valid data after removing NaNs")

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Input must be numeric, got dtype {arr.dtype}")

    if np.any(np.isinf(arr)):
        raise ValueError("Input contains infinite values")

    if np.std(arr) < 1e-10:
        raise ValueError("Input is constant (zero variance)")

    if arr.ndim > 1:
        raise ValueError(f"Input must be 1-D, got {arr.ndim} dimensions")

    return arr


def _get_critical_values_dict(cv_dict: Dict) -> Dict[str, float]:
    if isinstance(cv_dict, dict):
        return {str(k): float(v) for k, v in cv_dict.items()}
    return {}


def _schwert_maxlag(n_obs: int) -> int:
    return max(0, int(round(12.0 * (n_obs / 100.0) ** 0.25)))


def _approximate_left_tail_pvalue(statistic: float,
                                  critical_values: Dict[str, float]) -> float:
    if not np.isfinite(statistic):
        return np.nan

    points = []
    for level, critical_value in critical_values.items():
        if isinstance(level, str) and level.endswith("%") and np.isfinite(critical_value):
            points.append((float(level.rstrip("%")) / 100.0, float(critical_value)))

    if not points:
        return np.nan

    points.sort(key=lambda item: item[1])

    if statistic <= points[0][1]:
        return max(0.001, points[0][0] / 2.0)

    for (p_low, cv_low), (p_high, cv_high) in zip(points[:-1], points[1:]):
        if cv_low <= statistic <= cv_high:
            weight = (statistic - cv_low) / (cv_high - cv_low)
            return p_low + weight * (p_high - p_low)

    p_last, cv_last = points[-1]
    return min(0.999, p_last + max(0.0, statistic - cv_last) * 0.10)


def _approximate_right_tail_pvalue(statistic: float,
                                   critical_values: Dict[str, float]) -> float:
    if not np.isfinite(statistic):
        return np.nan

    points = []
    for level, critical_value in critical_values.items():
        if isinstance(level, str) and level.endswith("%") and np.isfinite(critical_value):
            points.append((float(level.rstrip("%")) / 100.0, float(critical_value)))

    if not points:
        return np.nan

    points.sort(key=lambda item: item[1], reverse=True)

    if statistic >= points[0][1]:
        return max(0.001, points[0][0] / 2.0)

    for (p_high, cv_high), (p_low, cv_low) in zip(points[:-1], points[1:]):
        if cv_low <= statistic <= cv_high:
            weight = (statistic - cv_low) / (cv_high - cv_low)
            return p_low + weight * (p_high - p_low)

    p_last, cv_last = points[-1]
    return min(0.999, p_last + max(0.0, cv_last - statistic) * 0.10)


def _infer_left_tail_conclusion(statistic: float,
                                critical_values: Dict[str, float],
                                reject_value: StationarityResult,
                                fail_value: StationarityResult) -> StationarityResult:
    critical_5 = critical_values.get("5%")
    if critical_5 is None or not np.isfinite(statistic):
        return StationarityResult.UNCERTAIN
    return reject_value if statistic < critical_5 else fail_value


def _coerce_break_timestamp(ts: Union[pd.Series, np.ndarray],
                            break_index: Optional[int]) -> Optional[pd.Timestamp]:
    if break_index is None:
        return None
    if isinstance(ts, pd.Series) and len(ts.index) > break_index:
        try:
            return pd.Timestamp(ts.index[break_index])
        except Exception:
            return None
    return None


def _joint_wald_pvalue(result, indices: List[int]) -> float:
    if not indices:
        return np.nan
    restriction = np.zeros((len(indices), len(result.params)))
    for row, idx in enumerate(indices):
        restriction[row, idx] = 1.0
    try:
        return float(result.f_test(restriction).pvalue)
    except Exception:
        return np.nan


def _newey_west_long_run_variance(values: np.ndarray, bandwidth: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n == 0:
        return np.nan

    gamma0 = float(np.dot(x, x) / n)
    lrv = gamma0
    max_bandwidth = min(int(bandwidth), n - 1)
    for lag in range(1, max_bandwidth + 1):
        gamma = float(np.dot(x[lag:], x[:-lag]) / n)
        weight = 1.0 - lag / (max_bandwidth + 1.0)
        lrv += 2.0 * weight * gamma

    return max(lrv, gamma0, 1e-12)


def _theil_sen_detrend(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    trend = np.arange(1, len(data) + 1, dtype=float)
    slope, intercept, _, _ = stats.theilslopes(data, trend)
    fitted = intercept + slope * trend
    return data - fitted, float(intercept), float(slope)


class ADFTest:
    """Augmented Dickey-Fuller Test for unit roots"""

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            regression: str = 'c',
            autolag: str = 'AIC',
            maxlag: Optional[int] = None) -> TestResult:
        """
        Run ADF test

        Parameters:
        -----------
        ts : time series data
        regression : 'c' (constant), 'ct' (constant + trend), 'n' (no constant)
        autolag : method for automatic lag selection
        maxlag : maximum lag to consider
        """
        data = _ensure_array(ts)

        if len(data) < 10:
            return TestResult(
                test_name="ADF",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=len(data)
            )

        try:
            result = adfuller(data, regression=regression, autolag=autolag, maxlag=maxlag)

            if len(result) >= 5:
                stat, pval, used_lag, nobs, cv, icbest = result[0], result[1], result[2], result[3], result[4], result[5] if len(result) > 5 else None
            else:
                stat, pval, used_lag, nobs, cv = result
                icbest = None

            cv_dict = _get_critical_values_dict(cv)

            if regression == 'n':
                conclusion = StationarityResult.NON_STATIONARY if pval > 0.05 else StationarityResult.STATIONARY
            elif regression == 'ct':
                conclusion = StationarityResult.NON_STATIONARY if pval > 0.05 else StationarityResult.TREND_STATIONARY
            else:
                if pval > 0.05:
                    conclusion = StationarityResult.NON_STATIONARY
                else:
                    conclusion = StationarityResult.STATIONARY

            return TestResult(
                test_name="ADF",
                statistic=stat,
                p_value=pval,
                critical_values=cv_dict,
                conclusion=conclusion,
                used_lag=used_lag,
                n_obs=nobs,
                regression=regression
            )
        except Exception as e:
            return TestResult(
                test_name="ADF",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                additional_info={"error": str(e)}
            )


class KPSSTest:
    """Kwiatkowski-Phillips-Schmidt-Shin Test for stationarity"""

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            regression: str = 'c',
            nlags: str = 'auto') -> TestResult:
        """
        Run KPSS test (null: stationarity)

        Parameters:
        -----------
        ts : time series data
        regression : 'c' (level), 'ct' (trend)
        nlags : 'auto' or integer for lag selection
        """
        data = _ensure_array(ts)

        if len(data) < 10:
            return TestResult(
                test_name="KPSS",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=len(data)
            )

        try:
            result = kpss(data, regression=regression, nlags=nlags)
            stat, pval, cv, lags = result

            cv_dict = _get_critical_values_dict(cv)

            if pval < 0.05:
                conclusion = StationarityResult.NON_STATIONARY
            else:
                conclusion = StationarityResult.STATIONARY if regression == 'c' else StationarityResult.TREND_STATIONARY

            return TestResult(
                test_name="KPSS",
                statistic=stat,
                p_value=pval,
                critical_values=cv_dict,
                conclusion=conclusion,
                lags=lags,
                n_obs=len(data),
                regression=regression
            )
        except Exception as e:
            return TestResult(
                test_name="KPSS",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                additional_info={"error": str(e)}
            )


class PhillipsPerronTest:
    """Phillips-Perron Test for unit roots using arch package
    
    Falls back to statsmodels if arch is not available.
    """

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            regression: str = 'c',
            nlags: Optional[int] = None) -> TestResult:
        """Run Phillips-Perron test"""
        data = _ensure_array(ts)

        if len(data) < 10:
            return TestResult(
                test_name="Phillips-Perron",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=len(data)
            )

        try:
            PhillipsPerron = _try_import_arch()
            
            if PhillipsPerron is not None:
                if regression == 'c':
                    trend = 'c'
                elif regression == 'ct':
                    trend = 'ct'
                else:
                    trend = 'n'
                
                pp = PhillipsPerron(data, trend=trend, lags=nlags)
                stat = pp.stat
                pval = pp.pvalue
                
                cv_dict = {
                    "1%": pp.critical_values['1%'],
                    "5%": pp.critical_values['5%'],
                    "10%": pp.critical_values['10%']
                }
                lags = pp.lags
            else:
                from statsmodels.tsa.stattools import phillipsperron as sm_pp
                result = sm_pp(data, regression=regression, nlags=nlags)
                stat, pval, cv, lags = result
                cv_dict = _get_critical_values_dict(cv)

            conclusion = StationarityResult.NON_STATIONARY if pval > 0.05 else StationarityResult.STATIONARY

            return TestResult(
                test_name="Phillips-Perron",
                statistic=stat,
                p_value=pval,
                critical_values=cv_dict,
                conclusion=conclusion,
                lags=lags,
                n_obs=len(data),
                regression=regression
            )
        except Exception as e:
            return TestResult(
                test_name="Phillips-Perron",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                additional_info={"error": str(e)}
            )


class ZivotAndrewsTest:
    """Zivot-Andrews test for structural break in unit root
    
    Wraps statsmodels implementation for statistical correctness.
    """

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            trim: float = 0.15,
            maxlag: Optional[int] = None,
            autolag: str = 'AIC') -> TestResult:
        """
        Zivot-Andrews test (null: unit root with structural break)

        Parameters:
        -----------
        ts : time series data
        trim : fraction trimmed from each end (default 15%)
        maxlag : maximum lag
        autolag : automatic lag selection method
        """
        data = _ensure_array(ts)
        n = len(data)

        if n < 30:
            return TestResult(
                test_name="Zivot-Andrews",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n
            )

        try:
            result = zivot_andrews(data, trim=trim, maxlag=maxlag, autolag=autolag)
            stat = result[0]
            pval = result[1]
            cv = result[4] if len(result) > 4 else {}
            
            cv_dict = _get_critical_values_dict(cv)
            
            conclusion = StationarityResult.NON_STATIONARY if pval > 0.05 else StationarityResult.STATIONARY

            break_date = None
            if isinstance(ts, pd.Series) and ts.index.inferred_type == 'datetime64':
                if hasattr(result, 'breakpoint'):
                    break_date = ts.index[result.breakpoint] if result.breakpoint < len(ts) else None

            return TestResult(
                test_name="Zivot-Andrews",
                statistic=stat,
                p_value=pval,
                critical_values=cv_dict,
                conclusion=conclusion,
                n_obs=n,
                structural_break_date=break_date,
                break_type="level"
            )
        except Exception as e:
            return TestResult(
                test_name="Zivot-Andrews",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                additional_info={"error": str(e)}
            )


class ClementeMontanesReyesTest:
    """Single-break AO/IO unit-root test from the Clemente-Montanes-Reyes toolkit.

    The implementation follows the public Stata reference procedures
    `clemao1.ado` and `clemio1.ado`.
    """

    _AO_CRITICAL_VALUES = {"1%": -4.38, "5%": -3.56, "10%": -3.24}
    _IO_CRITICAL_VALUES = {"1%": -4.95, "5%": -4.27, "10%": -3.99}

    @staticmethod
    def _break_dummies(n: int, break_index: int) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.arange(n)
        du = (positions > break_index).astype(float)
        dtb = (positions == break_index + 1).astype(float)
        return du, dtb

    @staticmethod
    def _fit_ao_regression(y_tilde: np.ndarray,
                           dtb: np.ndarray,
                           lag_order: int):
        n = len(y_tilde)
        diff_y = np.diff(y_tilde)
        start = lag_order + 1
        if n - start < 8:
            return None

        dep = []
        rows = []
        for t in range(start, n):
            row = [y_tilde[t - 1]]
            for lag in range(1, max(1, lag_order) + 1):
                row.append(dtb[t - lag])
            for lag in range(1, lag_order + 1):
                row.append(diff_y[t - lag - 1])
            dep.append(y_tilde[t])
            rows.append(row)

        return OLS(np.asarray(dep), np.asarray(rows)).fit()

    @staticmethod
    def _fit_io_regression(data: np.ndarray,
                           du: np.ndarray,
                           dtb: np.ndarray,
                           lag_order: int):
        n = len(data)
        diff_y = np.diff(data)
        start = lag_order + 1
        if n - start < 8:
            return None

        dep = []
        rows = []
        for t in range(start, n):
            row = [1.0, data[t - 1], dtb[t], du[t]]
            for lag in range(1, lag_order + 1):
                row.append(diff_y[t - lag - 1])
            dep.append(data[t])
            rows.append(row)

        return OLS(np.asarray(dep), np.asarray(rows)).fit()

    @staticmethod
    def _select_ao_lag(y_tilde: np.ndarray,
                       dtb: np.ndarray,
                       maxlag: int,
                       alpha: float) -> int:
        if maxlag <= 0:
            return 0

        full_result = ClementeMontanesReyesTest._fit_ao_regression(y_tilde, dtb, maxlag)
        if full_result is None:
            return 0

        kopt = 0
        for lag in range(maxlag, 0, -1):
            cumulative_indices = (
                list(range(1 + (lag - 1), 1 + maxlag)) +
                list(range(1 + maxlag + (lag - 1), 1 + (2 * maxlag)))
            )
            cumulative_p = _joint_wald_pvalue(full_result, cumulative_indices)

            truncated_result = ClementeMontanesReyesTest._fit_ao_regression(y_tilde, dtb, lag)
            boundary_p = np.nan
            if truncated_result is not None:
                boundary_p = _joint_wald_pvalue(truncated_result, [lag, 2 * lag])

            if (np.isfinite(cumulative_p) and cumulative_p < alpha) or (
                np.isfinite(boundary_p) and boundary_p < alpha
            ):
                kopt = lag
                break

        return kopt

    @staticmethod
    def _select_io_lag(data: np.ndarray,
                       du: np.ndarray,
                       dtb: np.ndarray,
                       maxlag: int,
                       alpha: float) -> int:
        if maxlag <= 0:
            return 0

        full_result = ClementeMontanesReyesTest._fit_io_regression(data, du, dtb, maxlag)
        if full_result is None:
            return 0

        kopt = 0
        for lag in range(maxlag, 0, -1):
            cumulative_p = _joint_wald_pvalue(
                full_result,
                list(range(4 + (lag - 1), 4 + maxlag))
            )

            truncated_result = ClementeMontanesReyesTest._fit_io_regression(data, du, dtb, lag)
            boundary_p = np.nan
            if truncated_result is not None:
                boundary_p = _joint_wald_pvalue(truncated_result, [3 + lag])

            if (np.isfinite(cumulative_p) and cumulative_p < alpha) or (
                np.isfinite(boundary_p) and boundary_p < alpha
            ):
                kopt = lag
                break

        return kopt

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            model: str = 'AO',
            trim: float = 0.05,
            maxlag: Optional[int] = None,
            lag_alpha: float = 0.05) -> TestResult:
        """
        Run the single-break additive-outlier or innovational-outlier test.

        Parameters:
        -----------
        ts : time series data
        model : 'AO' or 'IO'
        trim : trimming fraction used in the breakpoint search
        maxlag : maximum lag order for the augmented regression
        lag_alpha : significance level in the general-to-specific lag search
        """
        data = _ensure_array(ts)
        n = len(data)
        variant = model.upper()

        if n < 30:
            return TestResult(
                test_name=f"Clemente-Montanes-Reyes ({variant})",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n
            )

        if variant not in {"AO", "IO"}:
            raise ValueError("model must be 'AO' or 'IO'")

        if maxlag is None:
            maxlag = max(1, _schwert_maxlag(n))

        trim_n = int(trim * n + 0.49)
        candidate_breaks = range(trim_n, n - trim_n)

        best_search_stat = np.inf
        best_break = None
        best_level_coef = np.nan
        best_level_t = np.nan

        for break_index in candidate_breaks:
            try:
                du, dtb = ClementeMontanesReyesTest._break_dummies(n, break_index)
                if variant == 'AO':
                    detrend_fit = OLS(data, np.column_stack([np.ones(n), du])).fit()
                    y_tilde = detrend_fit.resid
                    search_result = ClementeMontanesReyesTest._fit_ao_regression(y_tilde, dtb, lag_order=1)
                    if search_result is None:
                        continue
                    search_stat = float((search_result.params[0] - 1.0) / search_result.bse[0])
                    level_coef = float(detrend_fit.params[1])
                    level_t = float(detrend_fit.tvalues[1])
                else:
                    search_result = ClementeMontanesReyesTest._fit_io_regression(data, du, dtb, lag_order=1)
                    if search_result is None:
                        continue
                    search_stat = float((search_result.params[1] - 1.0) / search_result.bse[1])
                    level_coef = float(search_result.params[3])
                    level_t = float(search_result.tvalues[3])

                if search_stat < best_search_stat:
                    best_search_stat = search_stat
                    best_break = break_index
                    best_level_coef = level_coef
                    best_level_t = level_t
            except Exception:
                continue

        if best_break is None:
            return TestResult(
                test_name=f"Clemente-Montanes-Reyes ({variant})",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n,
                additional_info={"error": "No admissible break date produced a valid regression"}
            )

        du, dtb = ClementeMontanesReyesTest._break_dummies(n, best_break)
        if variant == 'AO':
            y_tilde = OLS(data, np.column_stack([np.ones(n), du])).fit().resid
            used_lag = ClementeMontanesReyesTest._select_ao_lag(y_tilde, dtb, maxlag, lag_alpha)
            final_result = ClementeMontanesReyesTest._fit_ao_regression(y_tilde, dtb, used_lag)
            statistic = float((final_result.params[0] - 1.0) / final_result.bse[0])
            critical_values = ClementeMontanesReyesTest._AO_CRITICAL_VALUES
        else:
            used_lag = ClementeMontanesReyesTest._select_io_lag(data, du, dtb, maxlag, lag_alpha)
            final_result = ClementeMontanesReyesTest._fit_io_regression(data, du, dtb, used_lag)
            statistic = float((final_result.params[1] - 1.0) / final_result.bse[1])
            critical_values = ClementeMontanesReyesTest._IO_CRITICAL_VALUES

        p_value = _approximate_left_tail_pvalue(statistic, critical_values)
        conclusion = _infer_left_tail_conclusion(
            statistic,
            critical_values,
            reject_value=StationarityResult.STATIONARY,
            fail_value=StationarityResult.NON_STATIONARY
        )

        return TestResult(
            test_name=f"Clemente-Montanes-Reyes ({variant})",
            statistic=statistic,
            p_value=p_value,
            critical_values=critical_values,
            conclusion=conclusion,
            used_lag=used_lag,
            n_obs=n,
            structural_break_date=_coerce_break_timestamp(ts, best_break),
            break_type="additive_outlier" if variant == 'AO' else "innovational_outlier",
            additional_info={
                "break_index": int(best_break),
                "break_fraction": float((best_break + 1) / n),
                "level_shift_coefficient": best_level_coef,
                "level_shift_t_stat": best_level_t,
                "maxlag": int(maxlag),
                "lag_selection_alpha": float(lag_alpha)
            }
        )


class LeeStrazicichTest:
    """Minimum LM unit-root test with two endogenous structural breaks.

    This port follows the public R implementation in the repository
    `hannes101/LeeStrazicichUnitRoot`, which encodes the Lee-Strazicich
    minimum-LM search for two breaks.
    """

    _MODEL_C_CRITICAL_VALUES = {"1%": -4.545, "5%": -3.842, "10%": -3.504}
    _MODEL_CT_TABLE = {
        (0.2, 0.4): {"1%": -6.16, "5%": -5.59, "10%": -5.27},
        (0.2, 0.6): {"1%": -6.41, "5%": -5.74, "10%": -5.32},
        (0.2, 0.8): {"1%": -6.33, "5%": -5.71, "10%": -5.33},
        (0.4, 0.6): {"1%": -6.45, "5%": -5.67, "10%": -5.31},
        (0.4, 0.8): {"1%": -6.42, "5%": -5.65, "10%": -5.32},
        (0.6, 0.8): {"1%": -6.32, "5%": -5.73, "10%": -5.32},
    }

    @staticmethod
    def _resolve_model(model: str) -> str:
        normalized = model.upper()
        if normalized in {"C", "CRASH", "A"}:
            return "C"
        if normalized in {"CT", "BREAK"}:
            return "CT"
        raise ValueError("model must be one of 'C', 'CT', 'crash', or 'break'")

    @staticmethod
    def _build_break_design(n: int,
                            break1: int,
                            break2: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(1, n + 1, dtype=float)
        d1 = (np.arange(n) > break1).astype(float)
        d2 = (np.arange(n) > break2).astype(float)
        dt1 = np.maximum(0.0, np.arange(n, dtype=float) - break1)
        dt2 = np.maximum(0.0, np.arange(n, dtype=float) - break2)

        model_c = np.column_stack([t, d1, d2])
        model_ct = np.column_stack([t, d1, dt1, d2, dt2])
        return model_c, model_ct

    @staticmethod
    def _lm_components(data: np.ndarray,
                       z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dy = np.diff(data)
        dz = np.diff(z, axis=0)
        residuals = OLS(dy, dz).fit().resid
        s_tilde = np.concatenate([[0.0], np.cumsum(residuals)])
        ds = np.diff(s_tilde)
        return s_tilde, ds, dz

    @staticmethod
    def _fit_lm_regression(data: np.ndarray,
                           z: np.ndarray,
                           lag_order: int):
        s_tilde, ds, dz = LeeStrazicichTest._lm_components(data, z)
        n = len(data)
        start = lag_order + 1
        if n - start < 10:
            return None

        dep = []
        rows = []
        for t in range(start, n):
            row = [s_tilde[t - 1], *dz[t - 1]]
            for lag in range(1, lag_order + 1):
                row.append(ds[t - lag - 1])
            dep.append(ds[t - 1])
            rows.append(row)

        return OLS(np.asarray(dep), np.asarray(rows)).fit()

    @staticmethod
    def _select_lag(data: np.ndarray,
                    z: np.ndarray,
                    maxlag: int,
                    alpha: float) -> int:
        if maxlag <= 0:
            return 0

        for lag in range(maxlag, 0, -1):
            result = LeeStrazicichTest._fit_lm_regression(data, z, lag)
            if result is None:
                continue
            last_diff_index = z.shape[1] + lag
            if float(result.pvalues[last_diff_index]) <= alpha:
                return lag
        return 0

    @staticmethod
    def _critical_values(model: str,
                         lambda1: float,
                         lambda2: float) -> Dict[str, float]:
        if model == "C":
            return LeeStrazicichTest._MODEL_C_CRITICAL_VALUES

        grid_pair = min(
            LeeStrazicichTest._MODEL_CT_TABLE,
            key=lambda pair: abs(pair[0] - lambda1) + abs(pair[1] - lambda2)
        )
        return LeeStrazicichTest._MODEL_CT_TABLE[grid_pair]

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            model: str = 'C',
            trim: float = 0.10,
            maxlag: Optional[int] = None,
            lag_alpha: float = 0.10) -> TestResult:
        """
        Run the Lee-Strazicich minimum LM test with two endogenous breaks.

        Parameters:
        -----------
        ts : time series data
        model : 'C' / 'crash' for level shifts, 'CT' / 'break' for level+trend shifts
        trim : trimming share used in the breakpoint grid search
        maxlag : maximum augmentation lag order
        lag_alpha : significance level for the general-to-specific lag search
        """
        data = _ensure_array(ts)
        n = len(data)
        resolved_model = LeeStrazicichTest._resolve_model(model)

        if n < 50:
            return TestResult(
                test_name="Lee-Strazicich",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n
            )

        if maxlag is None:
            maxlag = max(0, _schwert_maxlag(n))

        trim_n = int(round(trim * n))
        start = trim_n
        end = n - trim_n - 1
        gap = 2 if resolved_model == "C" else 3

        best_stat = np.inf
        best_breaks = None
        best_lag = 0

        for break1 in range(start, end - gap + 1):
            for break2 in range(break1 + gap, end + 1):
                try:
                    z_c, z_ct = LeeStrazicichTest._build_break_design(n, break1, break2)
                    z = z_c if resolved_model == "C" else z_ct
                    used_lag = LeeStrazicichTest._select_lag(data, z, maxlag, lag_alpha)
                    result = LeeStrazicichTest._fit_lm_regression(data, z, used_lag)
                    if result is None:
                        continue
                    stat = float(result.tvalues[0])
                    if stat < best_stat:
                        best_stat = stat
                        best_breaks = (break1, break2)
                        best_lag = used_lag
                except Exception:
                    continue

        if best_breaks is None:
            return TestResult(
                test_name="Lee-Strazicich",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n,
                additional_info={"error": "No admissible break pair produced a valid LM regression"}
            )

        lambda1 = (best_breaks[0] + 1) / n
        lambda2 = (best_breaks[1] + 1) / n
        critical_values = LeeStrazicichTest._critical_values(resolved_model, lambda1, lambda2)
        p_value = _approximate_left_tail_pvalue(best_stat, critical_values)
        conclusion = _infer_left_tail_conclusion(
            best_stat,
            critical_values,
            reject_value=StationarityResult.STATIONARY,
            fail_value=StationarityResult.NON_STATIONARY
        )

        return TestResult(
            test_name="Lee-Strazicich",
            statistic=best_stat,
            p_value=p_value,
            critical_values=critical_values,
            conclusion=conclusion,
            used_lag=best_lag,
            n_obs=n,
            structural_break_date=_coerce_break_timestamp(ts, best_breaks[0]),
            break_type=f"two_breaks_{resolved_model.lower()}",
            additional_info={
                "break_indices": [int(best_breaks[0]), int(best_breaks[1])],
                "break_fractions": [float(lambda1), float(lambda2)],
                "second_break_date": _coerce_break_timestamp(ts, best_breaks[1]),
                "lag_selection_alpha": float(lag_alpha),
                "maxlag": int(maxlag)
            }
        )


class VarianceRatioTest:
    """Variance ratio test for random walk hypothesis (Lo-MacKinlay)

    Tests whether a series behaves like a random walk (VR ≈ 1).
    - VR > 1: indicates positive serial correlation (could mean-revert or trend)
    - VR < 1: indicates negative serial correlation (could mean-revert)
    - VR = 1: consistent with random walk (non-stationary)
    
    Note: This tests the random walk hypothesis, NOT stationarity directly.
    Rejection of VR=1 means the series is NOT a random walk - it could be
    mean-reverting or have a trend. Additional analysis needed for interpretation.
    """

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            max_lag: int = 10,
            random_walk_std: bool = True) -> TestResult:
        """
        Lo-MacKinlay variance ratio test for random walk

        Parameters:
        -----------
        ts : time series data
        max_lag : maximum holding period (horizon)
        random_walk_std : use heteroskedasticity-robust standard errors
        """
        data = _ensure_array(ts)
        n = len(data)

        if n < 20:
            return TestResult(
                test_name="Variance Ratio",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n
            )

        returns = np.diff(data)
        var_1 = np.var(returns, ddof=1)

        if var_1 == 0:
            return TestResult(
                test_name="Variance Ratio",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n,
                additional_info={"error": "Zero variance in returns"}
            )

        vr_values = []
        t_stats = []

        for lag in range(2, max_lag + 1):
            n_lag = n - lag
            cum_ret = data[lag:] - data[:n_lag]

            var_lag = np.var(cum_ret, ddof=1)
            vr = var_lag / (lag * var_1)

            vr_values.append(vr)

            if random_walk_std:
                theta = 0
                for k in range(1, lag):
                    if k < len(returns) - 1:
                        cov = np.cov(returns[k:], returns[:len(returns) - k], ddof=1)[0, 1]
                        if not np.isnan(cov):
                            theta += 2 * (lag - k) / lag * (cov ** 2) / (var_1 ** 2)

                se = np.sqrt((2 * (2 * lag - 1)) / (3 * lag * n_lag) * (1 + theta))
            else:
                se = np.sqrt(2 * (lag - 1) / (lag * n_lag))

            if se > 0:
                t_stat = (vr - 1) / se
                t_stats.append(t_stat)

        if not t_stats:
            return TestResult(
                test_name="Variance Ratio",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n
            )

        avg_vr = np.mean(vr_values)
        avg_t = np.mean(t_stats)
        p_value = 2 * (1 - norm.cdf(abs(avg_t)))

        if p_value > 0.05:
            conclusion = StationarityResult.NON_STATIONARY
        elif avg_vr < 1:
            conclusion = StationarityResult.MEAN_REVERTING
        else:
            conclusion = StationarityResult.TREND_STATIONARY

        return TestResult(
            test_name="Variance Ratio",
            statistic=avg_vr,
            p_value=p_value,
            critical_values={"5%": 1.96, "10%": 1.645},
            conclusion=conclusion,
            n_obs=n,
            additional_info={"vr_by_lag": dict(zip(range(2, max_lag + 1), vr_values)), "interpretation": "VR != 1 means not random walk"}
        )


class KPSSStationarityTest:
    """KPSS-based stationarity test (more powerful than standard KPSS)"""

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            bandwidth: Optional[int] = None) -> TestResult:
        """
        KPSS stationarity test (not unit root)

        Parameters:
        -----------
        ts : time series data
        bandwidth : Newey-West bandwidth
        """
        data = _ensure_array(ts)
        n = len(data)

        if n < 20:
            return TestResult(
                test_name="KPSS Stationarity",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n
            )

        if bandwidth is None:
            bandwidth = int(4 * (n / 100) ** 0.25)

        residuals = data - np.mean(data)

        s = np.zeros(n)
        for t in range(1, n):
            s[t] = s[t - 1] + residuals[t]

        s_sq = s ** 2
        eta = np.sum(s_sq) / (n ** 2)

        gamma = 0
        for k in range(1, bandwidth + 1):
            cov = np.cov(residuals[k:], residuals[:n - k], ddof=1)[0, 1]
            gamma += 2 * (1 - k / (bandwidth + 1)) * cov

        sigma_sq = np.var(residuals, ddof=1) + 2 * gamma

        if sigma_sq <= 0:
            sigma_sq = np.var(residuals, ddof=1)

        stat = eta / sigma_sq

        cv_dict = {
            "1%": 0.739,
            "5%": 0.463,
            "10%": 0.347
        }

        if stat > cv_dict["1%"]:
            p_value = 0.01
        elif stat > cv_dict["5%"]:
            p_value = 0.05
        elif stat > cv_dict["10%"]:
            p_value = 0.10
        else:
            p_value = 0.20

        conclusion = StationarityResult.NON_STATIONARY if p_value < 0.05 else StationarityResult.STATIONARY

        return TestResult(
            test_name="KPSS Stationarity",
            statistic=stat,
            p_value=p_value,
            critical_values=cv_dict,
            conclusion=conclusion,
            n_obs=n
        )


class RankTest:
    """KPSS-type nonparametric rank stationarity test.

    The test replaces raw residuals by centered ranks of the demeaned or
    detrended series. Let `r_t` denote the centered ranks and
    `S_t = sum_{i=1}^t r_i`. The test statistic is

        eta = (1 / n^2) * sum_{t=1}^n S_t^2 / sigma_r^2,

    where `sigma_r^2` is a Newey-West estimate of the long-run variance of the
    centered rank process. Under the null of stationarity the limiting law is
    the same as KPSS, so the usual KPSS critical values apply.
    """

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            regression: str = 'c',
            bandwidth: Optional[int] = None) -> TestResult:
        """
        Run the rank-based KPSS-style stationarity test.

        Parameters:
        -----------
        ts : time series data
        regression : 'c' for level-stationary null, 'ct' for trend-stationary null
        bandwidth : Newey-West bandwidth for the long-run variance estimate
        """
        data = _ensure_array(ts)
        n = len(data)
        regression = regression.lower()

        if n < 20:
            return TestResult(
                test_name="Rank Test",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n
            )

        if regression not in {'c', 'ct'}:
            raise ValueError("regression must be 'c' or 'ct'")

        if bandwidth is None:
            bandwidth = max(1, int(round(4.0 * (n / 100.0) ** 0.25)))

        if regression == 'ct':
            residuals, intercept, slope = _theil_sen_detrend(data)
            critical_values = {"1%": 0.216, "5%": 0.146, "10%": 0.119}
            null_conclusion = StationarityResult.TREND_STATIONARY
            detrend_info = {"intercept": intercept, "slope": slope}
        else:
            residuals = data - np.mean(data)
            critical_values = {"1%": 0.739, "5%": 0.463, "10%": 0.347}
            null_conclusion = StationarityResult.STATIONARY
            detrend_info = {"mean": float(np.mean(data))}

        ranks = stats.rankdata(residuals, method='average')
        centered_ranks = ranks - (n + 1.0) / 2.0
        partial_sums = np.cumsum(centered_ranks)
        eta = float(np.sum(partial_sums ** 2) / (n ** 2))
        sigma_r_sq = _newey_west_long_run_variance(centered_ranks, bandwidth)
        statistic = eta / sigma_r_sq
        p_value = _approximate_right_tail_pvalue(statistic, critical_values)

        conclusion = (
            StationarityResult.NON_STATIONARY
            if statistic > critical_values["5%"]
            else null_conclusion
        )

        return TestResult(
            test_name="Rank Test",
            statistic=statistic,
            p_value=p_value,
            critical_values=critical_values,
            conclusion=conclusion,
            lags=bandwidth,
            n_obs=n,
            regression=regression,
            additional_info={
                "sigma_r_sq": float(sigma_r_sq),
                "eta": eta,
                "centered_rank_mean": float(np.mean(centered_ranks)),
                **detrend_info,
            }
        )


class MultiLagADFTest:
    """Multi-lag ADF test - runs ADF at multiple lags and selects best by AIC

    Note: This is NOT true SURADF. It runs ADF with different lag caps and picks
    the minimum p-value, which introduces selection bias. Use with caution.
    """

    @staticmethod
    def run(ts: Union[pd.Series, np.ndarray],
            max_lags: int = 5,
            significance: float = 0.05) -> TestResult:
        """
        Multi-lag ADF test - select best lag by AIC

        Parameters:
        -----------
        ts : time series data
        max_lags : maximum number of lags to test
        significance : significance level
        """
        data = _ensure_array(ts)
        n = len(data)

        if n < 30:
            return TestResult(
                test_name="MultiLagADF",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n
            )

        results = []

        for lag in range(1, max_lags + 1):
            try:
                result = adfuller(data, maxlag=lag, regression='c')
                t_stat = result[0]
                p_val = result[1]
                results.append({
                    'lag': lag,
                    't_stat': t_stat,
                    'p_value': p_val,
                    'aic': result[5] if len(result) > 5 else np.nan
                })
            except:
                continue

        if not results:
            return TestResult(
                test_name="MultiLagADF",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                conclusion=StationarityResult.UNCERTAIN,
                n_obs=n
            )

        best_result = min(results, key=lambda x: x['aic'] if not np.isnan(x['aic']) else float('inf'))

        cv_dict = {
            "1%": -3.96,
            "5%": -3.41,
            "10%": -3.12
        }

        conclusion = StationarityResult.NON_STATIONARY if best_result['p_value'] > significance else StationarityResult.STATIONARY

        return TestResult(
            test_name="MultiLagADF",
            statistic=best_result['t_stat'],
            p_value=best_result['p_value'],
            critical_values=cv_dict,
            conclusion=conclusion,
            used_lag=best_result['lag'],
            n_obs=n,
            additional_info={"all_lags": results}
        )


class StationaritySuite:
    """Comprehensive stationarity analysis suite"""

    @staticmethod
    def full_analysis(ts: Union[pd.Series, np.ndarray],
                      include_structural: bool = True,
                      include_advanced: bool = True) -> Dict[str, TestResult]:
        """
        Run comprehensive stationarity analysis

        Parameters:
        -----------
        ts : time series data
        include_structural : include structural break tests
        include_advanced : include advanced/rare tests
        """
        results = {}

        results['adf_const'] = ADFTest.run(ts, regression='c')
        results['adf_trend'] = ADFTest.run(ts, regression='ct')
        results['kpss_level'] = KPSSTest.run(ts, regression='c')
        results['kpss_trend'] = KPSSTest.run(ts, regression='ct')
        results['phillips_perron'] = PhillipsPerronTest.run(ts, regression='c')

        if include_structural:
            results['zivot_andrews'] = ZivotAndrewsTest.run(ts)
            results['clemente_montanes'] = ClementeMontanesReyesTest.run(ts)
            results['lee_strazicich'] = LeeStrazicichTest.run(ts)

        if include_advanced:
            results['variance_ratio'] = VarianceRatioTest.run(ts)
            results['kpss_stationarity'] = KPSSStationarityTest.run(ts)
            results['rank_test'] = RankTest.run(ts)
            results['multilag_adf'] = MultiLagADFTest.run(ts)

        return results

    @staticmethod
    def consensus(results: Dict[str, TestResult]) -> StationarityResult:
        """
        Determine consensus from multiple tests

        Note: Accounts for different null hypotheses:
        - ADF, PP, Zivot-Andrews, Clemente-Montañés-Reyes, Lee-Strazicich: null = unit root
        - KPSS, KPSSStationarityTest, RankTest: null = stationarity
        """
        unit_root_tests = {'adf_const', 'adf_trend', 'phillips_perron', 'zivot_andrews',
                          'clemente_montanes', 'lee_strazicich'}
        stationarity_tests = {'kpss_level', 'kpss_trend', 'kpss_stationarity', 'rank_test'}

        unit_root_rejects = 0
        stationarity_rejects = 0

        for name, r in results.items():
            if r.conclusion == StationarityResult.UNCERTAIN:
                continue
            if name in unit_root_tests:
                if r.conclusion in (StationarityResult.STATIONARY, StationarityResult.TREND_STATIONARY):
                    unit_root_rejects += 1
            elif name in stationarity_tests:
                if r.conclusion == StationarityResult.NON_STATIONARY:
                    stationarity_rejects += 1

        if unit_root_rejects > stationarity_rejects:
            return StationarityResult.STATIONARY
        elif stationarity_rejects > unit_root_rejects:
            return StationarityResult.NON_STATIONARY
        else:
            return StationarityResult.UNCERTAIN

    @staticmethod
    def summary_report(results: Dict[str, TestResult]) -> pd.DataFrame:
        """Generate summary DataFrame of all test results"""
        summary_data = []
        for name, result in results.items():
            summary_data.append(result.summary_dict())
        return pd.DataFrame(summary_data)
