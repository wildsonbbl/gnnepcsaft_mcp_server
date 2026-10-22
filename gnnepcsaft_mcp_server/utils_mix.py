"""Utilities for thermodynamic calculations in fluid mixtures.

This module provides shared data structures and helpers for mixture
vapor–liquid equilibrium, liquid–liquid equilibrium, density, vapor-pressure,
and critical-point calculations using PC-SAFT models.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from gnnepcsaft.pcsaft.feos.equilibria import mix_bp_feos, mix_dp_feos
from gnnepcsaft.pcsaft.pcsaft_feos import (
    critical_points_feos,
    mix_den_feos,
    mix_lle_diagram_feos,
    mix_lle_feos,
    mix_vle_diagram_feos,
    mix_vlle_diagram_feos,
    pure_vp_feos,
)

from . import logger
from .utils import predict_pcsaft_parameters

EPS = 1e-6


@dataclass
class BinaryVlePxyContext:
    """Shared context for binary VLE P-x-y calculations."""

    parameters_list: List[List[float]]
    kij_matrix: List[List[float]]
    temperature: float
    min_pc: float
    max_pc: float


@dataclass
class BinaryVleSeries:
    """Series data for binary VLE P-x-y output."""

    xs: List[float]
    bps: List[float]
    dps: List[float]


@dataclass
class TernaryVleContext:
    """Shared context for ternary VLE P-x calculations."""

    parameters_list: List[List[float]]
    kij_matrix: List[List[float]]
    temperature: float
    min_pc: float
    max_pc: float


@dataclass
class TernaryVleSeries:
    """Series data for ternary VLE P-x output."""

    x1_values: List[float]
    bubble_pressures: List[float]
    dew_pressures: List[float]


@dataclass
class MixDenParams:
    """Inputs for mixture density calculations."""

    smiles_list: List[str]
    mole_fractions: List[float]
    kij_matrix: List[List[float]]
    min_temp: float
    max_temp: float
    pressure: float
    npoints: int


@dataclass
class MixLLEParams:
    """Inputs for mixture LLE calculations."""

    smiles_list: List[str]
    mole_fractions: List[float]
    kij_matrix: List[List[float]]
    temperature_min: float
    temperature_max: float
    pressure: float
    npoints: int


@dataclass
class TernaryVleTxParams:
    """Inputs for ternary VLE P-x calculations."""

    smiles_list: List[str]
    kij_matrix: List[List[float]]
    temperature: float
    solvent_ratio: float
    npoints: int
    mole_fractions: Optional[List[float]] = None


@dataclass
class MixVpParams:
    """Inputs for mixture vapor pressure calculations."""

    smiles_list: List[str]
    mole_fractions: List[float]
    kij_matrix: List[List[float]]
    min_temp: float
    max_temp: float
    npoints: int


def _binary_critical_points(
    parameters_list: List[List[float]],
) -> Tuple[List[float], List[float]]:
    tcs = []
    pcs = []
    for params in parameters_list:
        tc, pc, _ = critical_points_feos(parameters=params)
        tcs.append(tc)
        pcs.append(pc)
    return tcs, pcs


def _binary_pure_vps(
    parameters_list: List[List[float]],
    temperature: float,
    tcs: List[float],
) -> List[Optional[float]]:
    vps = []
    for params, tc in zip(parameters_list, tcs):
        try:
            if temperature < tc:
                vps.append(pure_vp_feos(parameters=params, state=[temperature]))
            else:
                vps.append(None)
        except RuntimeError:
            vps.append(None)
    return vps


def _validate_binary_vle_pxy(
    temperature: float,
    tcs: List[float],
    vps: List[Optional[float]],
):
    if temperature >= tcs[0] and temperature >= tcs[1]:
        raise ValueError(
            f"Temperature {temperature} K is above the critical temperature "
            f"of both components, which is {tcs[0]:.2f} K and {tcs[1]:.2f} K. "
            "VLE calculation is not meaningful.",
        )
    if vps[0] is None and vps[1] is None:
        raise ValueError(
            f"Unable to compute pure vapor pressures at {temperature} K for either component; "
            "VLE calculation is not meaningful.",
        )


def _binary_more_volatile_is_first(
    vps: List[Optional[float]],
    tcs: List[float],
    temperature: float,
) -> bool:
    vp0, vp1 = vps
    if vp0 is not None and vp1 is not None:
        return vp0 >= vp1
    if vp0 is None and vp1 is not None:
        return True
    if vp1 is None and vp0 is not None:
        return False
    if temperature >= tcs[0] and temperature >= tcs[1]:
        return True
    return True


def _append_binary_pure_point(
    temperature: float,
    tcs: List[float],
    vps: List[Optional[float]],
    series: BinaryVleSeries,
):
    if temperature < tcs[1] and vps[1] is not None:
        series.xs.append(0.0)
        series.bps.append(vps[1])
        series.dps.append(vps[1])


def _build_binary_vle_pxy_series(
    x0s: List[float],
    temperature: float,
    tcs: List[float],
    vps: List[Optional[float]],
    context: BinaryVlePxyContext,
) -> BinaryVleSeries:
    series = BinaryVleSeries(xs=[], bps=[], dps=[])
    _append_binary_pure_point(temperature, tcs, vps, series)

    for x0 in x0s:
        state = [context.temperature, np.nan, x0, 1 - x0]
        bp = _get_bp_dp_pressure(
            parameters_list=context.parameters_list,
            kij_matrix=context.kij_matrix,
            function_call=mix_bp_feos,
            state=state,
        )
        dp = _get_bp_dp_pressure(
            parameters_list=context.parameters_list,
            kij_matrix=context.kij_matrix,
            function_call=mix_dp_feos,
            state=state,
        )
        series.bps.append(bp)
        series.dps.append(dp)
        series.xs.append(x0)
    return series


def _ternary_critical_points(
    parameters_list: List[List[float]],
) -> Tuple[List[float], List[float]]:
    tcs = []
    pcs = []
    for params in parameters_list:
        tc, pc, _ = critical_points_feos(parameters=params)
        tcs.append(tc)
        pcs.append(pc)
    return tcs, pcs


def _ternary_pure_vps(
    parameters_list: List[List[float]],
    temperature: float,
    tcs: List[float],
) -> List[Optional[float]]:
    vps = []
    for params, tc in zip(parameters_list, tcs):
        try:
            if temperature < tc:
                vps.append(pure_vp_feos(parameters=params, state=[temperature]))
            else:
                vps.append(None)
        except RuntimeError:
            vps.append(None)
    return vps


def _validate_ternary_vle_tx_fixed(
    temperature: float,
    tcs: List[float],
    vps: List[Optional[float]],
):
    if all(temperature >= tc for tc in tcs):
        raise ValueError(
            f"Temperature {temperature} K is above the critical temperature of all components; "
            "VLE calculation is not meaningful.",
        )
    if all(vp is None for vp in vps):
        raise ValueError(
            f"Unable to compute pure vapor pressures at {temperature} K for any component; "
            "VLE calculation is not meaningful.",
        )


def _volatility_rank(vp: Optional[float], tc: float, temperature: float) -> float:
    if vp is not None:
        return vp
    if temperature >= tc:
        return float("inf")
    return float("-inf")


def _validate_ternary_volatility(
    smiles_list: List[str],
    vps: List[Optional[float]],
    tcs: List[float],
    temperature: float,
):
    ranks = [_volatility_rank(vps[i], tcs[i], temperature) for i in range(3)]
    most_volatile_index = int(np.argmax(ranks))
    if most_volatile_index != 0:
        raise ValueError(
            f"More volatile component ({smiles_list[most_volatile_index]}) must be listed first "
            "in ternary P-x calculation",
        )


def _validate_solvent_ratio(solvent_ratio: float):
    if not 0.0 < solvent_ratio < 1.0:
        raise ValueError(
            f"For ternary P-x, solvent ratio must be between 0 and 1, got ratio = {solvent_ratio}"
        )


def _ternary_solvent_split(x1: float, solvent_ratio: float) -> Tuple[float, float]:
    remaining = 1.0 - x1
    x2 = remaining * solvent_ratio
    x3 = remaining * (1.0 - solvent_ratio)
    return x2, x3


def _get_bp_dp_pressure(
    parameters_list: List[List[float]],
    kij_matrix: List[List[float]],
    function_call: Callable,
    state: List[float],
) -> float:
    try:
        return function_call(
            parameters=parameters_list,
            state=state,
            kij_matrix=kij_matrix,
        )
    except RuntimeError as exc:
        logger.debug(
            "_get_bp_dp_pressure: Runtime Error at state=%r: %s",
            state,
            exc,
        )
        return float("nan")
    except BaseException as exc:  # pylint: disable=W0718
        exception_type = type(exc).__name__
        if exception_type == "PanicException":
            logger.warning(
                "_get_bp_dp_pressure: PanicException at state=%r: %s",
                state,
                exc,
            )
            return float("nan")
        logger.exception(
            "_get_bp_dp_pressure: unexpected %s at state=%r",
            exception_type,
            state,
        )
        raise


def _build_ternary_vle_series(
    x1_grid: List[float],
    solvent_ratio: float,
    context: TernaryVleContext,
) -> TernaryVleSeries:
    series = TernaryVleSeries(x1_values=[], bubble_pressures=[], dew_pressures=[])

    for x1 in x1_grid:
        x2, x3 = _ternary_solvent_split(x1, solvent_ratio)
        if x2 <= 0.0 or x3 <= 0.0:
            continue
        state = [context.temperature, np.nan, float(x1), float(x2), float(x3)]
        bubble_p = _get_bp_dp_pressure(
            parameters_list=context.parameters_list,
            kij_matrix=context.kij_matrix,
            function_call=mix_bp_feos,
            state=state,
        )
        dew_p = _get_bp_dp_pressure(
            parameters_list=context.parameters_list,
            kij_matrix=context.kij_matrix,
            function_call=mix_dp_feos,
            state=state,
        )
        series.x1_values.append(float(x1))
        series.bubble_pressures.append(float(bubble_p))
        series.dew_pressures.append(float(dew_p))
    return series


def _build_fraction_grid(
    mole_fractions: Optional[List[float]],
    npoints: int,
) -> List[float]:
    x_grid = np.linspace(0.0, 1.0, num=npoints, dtype=np.float64).tolist()
    if mole_fractions:
        x_grid.extend(mole_fractions)
        x_grid = sorted(x_grid)
    return x_grid


def mix_den(params: MixDenParams) -> Tuple[List[float], List[float]]:
    """Calculate mixture density using PC-SAFT EOS."""
    parameters_list = [
        predict_pcsaft_parameters(smiles) for smiles in params.smiles_list
    ]
    temperatures = np.linspace(
        params.min_temp, params.max_temp, num=params.npoints
    ).tolist()

    densities = [
        mix_den_feos(
            parameters=parameters_list,
            state=[T, params.pressure, *params.mole_fractions],
            kij_matrix=params.kij_matrix,
        )
        for T in temperatures
    ]
    return temperatures, densities


def mix_vp(context: MixVpParams) -> Tuple[List[float], List[float], List[float]]:
    """Calculate mixture vapor pressure using PC-SAFT EOS."""
    parameters_list = [
        predict_pcsaft_parameters(smiles) for smiles in context.smiles_list
    ]
    temperatures = np.linspace(
        context.min_temp, context.max_temp, num=context.npoints
    ).tolist()

    bubble_points = []
    dew_point = []
    valid_temperatures = []
    for temp in temperatures:
        state = [temp, np.nan, *context.mole_fractions]
        bp = _get_bp_dp_pressure(
            parameters_list=parameters_list,
            kij_matrix=context.kij_matrix,
            function_call=mix_bp_feos,
            state=state,
        )
        dp = _get_bp_dp_pressure(
            parameters_list=parameters_list,
            kij_matrix=context.kij_matrix,
            function_call=mix_dp_feos,
            state=state,
        )
        bubble_points.append(bp)
        dew_point.append(dp)
        valid_temperatures.append(temp)
    return valid_temperatures, bubble_points, dew_point


def mix_vle(
    smiles_list: List[str], kij_matrix: List[List[float]], pressure: float, npoints: int
) -> Optional[Dict[str, List[float]]]:
    "Calculate mixture VLE (T-x-y) using PC-SAFT EOS"
    parameters_list = [predict_pcsaft_parameters(smiles) for smiles in smiles_list]

    try:
        return mix_vle_diagram_feos(
            parameters=parameters_list,
            state=[pressure],
            kij_matrix=kij_matrix,
            npoints=npoints if npoints < 500 else 500,
        )
    except RuntimeError as exc:
        logger.debug("mix_vle: Runtime Error at pressure=%.4f: %s", pressure, exc)
        raise
    except BaseException as exc:  # pylint: disable=W0718
        exception_type = type(exc).__name__
        if exception_type == "PanicException":
            logger.warning(
                "mix_vle: PanicException at pressure=%.4f: %s", pressure, exc
            )
        else:
            logger.exception(
                "mix_vle: unexpected %s at pressure=%.4f",
                exception_type,
                pressure,
            )
            raise
    return None


def mix_vlle(
    params: MixLLEParams,
) -> Optional[
    Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]
]:
    "Calculate mixture VLLE using PC-SAFT EOS"
    parameters_list = [
        predict_pcsaft_parameters(smiles) for smiles in params.smiles_list
    ]

    try:
        return mix_vlle_diagram_feos(
            parameters=parameters_list,
            state=[params.temperature_min, params.pressure, *params.mole_fractions],
            kij_matrix=params.kij_matrix,
            npoints=params.npoints if params.npoints < 500 else 500,
        )
    except RuntimeError as exc:
        logger.debug(
            "mix_vlle: Runtime Error at temperature=%.4f, pressure=%.4f, molefractions=%s: %s",
            params.temperature_min,
            params.pressure,
            params.mole_fractions,
            exc,
        )
        raise
    except BaseException as exc:  # pylint: disable=W0718
        exception_type = type(exc).__name__
        if exception_type == "PanicException":
            logger.warning(
                "mix_vlle: PanicException at temperature=%.4f, "
                "pressure=%.4f, molefractions=%s: %s",
                params.temperature_min,
                params.pressure,
                params.mole_fractions,
                exc,
            )
        else:
            logger.exception(
                "mix_vlle: unexpected %s at temperature=%.4f, pressure=%.4f, molefractions=%s",
                exception_type,
                params.temperature_min,
                params.pressure,
                params.mole_fractions,
            )
            raise
    return None


def mix_vle_pxy(
    smiles_list: List[str],
    kij_matrix: List[List[float]],
    temperature: float,
    npoints: int,
    mole_fractions: Optional[List[float]] = None,
) -> Tuple[List[float], List[float], List[float]]:
    "Calculate mixture VLE (P-x-y) using PC-SAFT EOS"
    parameters_list = [predict_pcsaft_parameters(smiles) for smiles in smiles_list]
    x0s = _build_fraction_grid(mole_fractions, npoints)
    tcs, pcs = _binary_critical_points(parameters_list)
    vps = _binary_pure_vps(parameters_list, temperature, tcs)

    _validate_binary_vle_pxy(temperature, tcs, vps)
    if not _binary_more_volatile_is_first(vps, tcs, temperature):
        raise ValueError(
            f"More volatile component ({smiles_list[1]}) must be listed first in P-x-y calculation",
        )

    min_pc = min(pcs)
    max_pc = max(pcs)

    context = BinaryVlePxyContext(
        parameters_list=parameters_list,
        kij_matrix=kij_matrix,
        temperature=temperature,
        min_pc=min_pc,
        max_pc=max_pc,
    )
    series = _build_binary_vle_pxy_series(x0s, temperature, tcs, vps, context)

    return series.xs, series.bps, series.dps


def mix_lle(
    params: MixLLEParams,
) -> Optional[Dict[str, List[float]]]:
    "Calculate mixture LLE using PC-SAFT EOS"
    parameters_list = [
        predict_pcsaft_parameters(smiles) for smiles in params.smiles_list
    ]

    try:
        return mix_lle_diagram_feos(
            parameters=parameters_list,
            state=[
                params.temperature_min,
                params.temperature_max,
                params.pressure,
                *params.mole_fractions,
            ],
            kij_matrix=params.kij_matrix,
            npoints=params.npoints if params.npoints < 500 else 500,
        )
    except RuntimeError as exc:
        logger.debug(
            "mix_lle: Runtime Error at temperature=%.4f, pressure=%.4f, molefractions=%s: %s",
            params.temperature_min,
            params.pressure,
            params.mole_fractions,
            exc,
        )
        raise
    except BaseException as exc:  # pylint: disable=W0718
        exception_type = type(exc).__name__
        if exception_type == "PanicException":
            logger.warning(
                "mix_lle: PanicException at temperature=%.4f, "
                "pressure=%.4f, molefractions=%s: %s",
                params.temperature_min,
                params.pressure,
                params.mole_fractions,
                exc,
            )
        else:
            logger.exception(
                "mix_lle: unexpected %s at temperature=%.4f, pressure=%.4f, molefractions=%s",
                exception_type,
                params.temperature_min,
                params.pressure,
                params.mole_fractions,
            )
            raise
    return None


def _get_ternary_vle_or_lle_data(
    params: List[List[float]],
    state: List[float],
    kij_matrix: List[List[float]],
    npoints: int,
) -> Dict[str, List[float]]:
    t, p = state  # Temperatura (K) e pressão (Pa)

    def _grid(n_pts: int = npoints):
        xi = np.linspace(1e-5, 0.999, n_pts, dtype=np.float64)
        x1_m, x2_m = np.meshgrid(xi, xi, indexing="xy")
        x3_m = 1.0 - x1_m - x2_m
        return x1_m, x2_m, x3_m, (x3_m >= 0.0)

    def _collect_tie_lines(x1_m, x2_m, x3_m, mask):
        valid_idx = np.argwhere(mask)
        ternary_data = {"x0": [], "x1": [], "x2": [], "y0": [], "y1": [], "y2": []}
        for i, j in valid_idx:
            try:
                lle = mix_lle_feos(  # this also calculates vle at T and P
                    params,
                    [t, p, x1_m[i, j].item(), x2_m[i, j].item(), x3_m[i, j].item()],
                    kij_matrix,
                )
            except (RuntimeError, ValueError):
                continue
            except BaseException as exc:  # pylint: disable=W0718
                exception_type = type(exc).__name__
                if exception_type == "PanicException":
                    logger.warning(
                        "_get_ternary_vle_or_lle_data: PanicException at temperature=%.4f, "
                        "pressure=%.4f, molefractions=%s: %s",
                        t,
                        p,
                        [x1_m[i, j].item(), x2_m[i, j].item(), x3_m[i, j].item()],
                        exc,
                    )
                    continue
                logger.exception(
                    "_get_ternary_vle_or_lle_data: unexpected %s at temperature=%.4f, "
                    "pressure=%.4f, molefractions=%s",
                    exception_type,
                    t,
                    p,
                    [x1_m[i, j].item(), x2_m[i, j].item(), x3_m[i, j].item()],
                )
                raise
            # For LLE, y is one phase and x is the other phase
            if lle["density liquid"][0] > lle["density vapor"][0]:
                ternary_data["x0"].extend(lle["x0"])
                ternary_data["x1"].extend(lle["x1"])
                ternary_data["x2"].extend(lle["x2"])
                ternary_data["y0"].extend(lle["y0"])
                ternary_data["y1"].extend(lle["y1"])
                ternary_data["y2"].extend(lle["y2"])
            else:
                ternary_data["x0"].extend(lle["y0"])
                ternary_data["x1"].extend(lle["y1"])
                ternary_data["x2"].extend(lle["y2"])
                ternary_data["y0"].extend(lle["x0"])
                ternary_data["y1"].extend(lle["x1"])
                ternary_data["y2"].extend(lle["x2"])
        return ternary_data

    x1, x2, x3, mask = _grid()
    return _collect_tie_lines(x1, x2, x3, mask)


def mix_ternary_lle(
    smiles_list: List[str],
    kij_matrix: List[List[float]],
    temperature: float,
    pressure: float,
    npoints: int,
) -> Dict[str, List[float]]:
    "Calculate ternary LLE using PC-SAFT EOS"
    parameters_list = [predict_pcsaft_parameters(smiles) for smiles in smiles_list]

    return _get_ternary_vle_or_lle_data(
        params=parameters_list,
        state=[temperature, pressure],
        kij_matrix=kij_matrix,
        npoints=npoints if npoints < 100 else 100,
    )


def mix_ternary_vle(
    smiles_list: List[str],
    kij_matrix: List[List[float]],
    temperature: float,
    pressure: float,
    npoints: int,
) -> Dict[str, List[float]]:
    "Calculate ternary VLE using PC-SAFT EOS"
    parameters_list = [predict_pcsaft_parameters(smiles) for smiles in smiles_list]

    return _get_ternary_vle_or_lle_data(
        params=parameters_list,
        state=[temperature, pressure],
        kij_matrix=kij_matrix,
        npoints=npoints if npoints < 100 else 100,
    )


def mix_ternary_vle_tx_fixed(
    params: TernaryVleTxParams,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate ternary isothermal VLE curve (P-x) at fixed solvent ratio.

    Solvent_ratio = x2 / (x2 + x3). The first component is scanned in composition.
    """
    parameters_list = [
        predict_pcsaft_parameters(smiles) for smiles in params.smiles_list
    ]
    tcs, pcs = _ternary_critical_points(parameters_list)
    vps = _ternary_pure_vps(parameters_list, params.temperature, tcs)

    _validate_ternary_vle_tx_fixed(params.temperature, tcs, vps)
    _validate_ternary_volatility(params.smiles_list, vps, tcs, params.temperature)
    _validate_solvent_ratio(params.solvent_ratio)

    context = TernaryVleContext(
        parameters_list=parameters_list,
        kij_matrix=params.kij_matrix,
        temperature=params.temperature,
        min_pc=min(pcs),
        max_pc=max(pcs),
    )
    x1_grid = _build_fraction_grid(params.mole_fractions, npoints=params.npoints)
    series = _build_ternary_vle_series(x1_grid, params.solvent_ratio, context)

    return series.x1_values, series.bubble_pressures, series.dew_pressures
