"""Utilities for fitting binary PC-SAFT interaction parameters (kij).

This module predicts bubble-point pressures and optimizes the binary
interaction parameter against available vapor–liquid equilibrium data.
"""

from typing import List, Union

import numpy as np
from gnnepcsaft.pcsaft.pcsaft_feos import mix_vp_feos
from numpy import float64
from numpy.typing import NDArray
from scipy.optimize import least_squares

from . import logger
from .utils import predict_pcsaft_parameters
from .utils_data import retrieve_vle_for_kij

EPS = 1e-6


def _pred_bp_worker(
    t: float, x1: float, k_12: float, params: List[List[float]]
) -> float:
    """Predict bubble point pressure for one state point.

    Must be at module level for pickle compatibility on Windows.

    Args:
        t (float): Temperature in Kelvin.
        x1 (float): Mole fraction of component 1.
        k_12 (float): Binary interaction parameter.
        params (List[List[float]]): PC-SAFT parameters for the binary mixture.

    Returns:
        out (float): Predicted bubble point pressure in kPa.
            Returns np.nan when bubble point calculation fails.
    """
    try:
        bp_pa, dp_pa = mix_vp_feos(
            parameters=params,
            state=[t, 0.0, x1, 1 - x1],
            kij_matrix=[[0.0, k_12], [k_12, 0.0]],
            epsilon_ab=None,
        )
        return float(max(bp_pa, dp_pa) / 1e3)  # Convert Pa to kPa
    except RuntimeError:
        return np.nan
    except BaseException as exc:  # pylint: disable=W0718
        exception_type = type(exc).__name__
        if exception_type == "PanicException":
            return np.nan
        raise


def _loss_fn_bubble_point(
    k_12_arr: np.ndarray,
    params: List[List[float]],
    x1: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """Compute residual vector for bubble point optimization.

    Args:
        k_12_arr (np.ndarray): Array containing a single optimization variable (k_12).
        params (List[List[float]]): PC-SAFT parameters for the binary mixture.
        x1 (np.ndarray): Mole fractions of component 1.
        temperature (np.ndarray): Temperatures in Kelvin.
        pressure (np.ndarray): Pressures in kPa.

    Returns:
        out (np.ndarray): Residual vector defined as log(predicted_P/experimental_P).
            Failed calculations are penalized with a large residual.
    """
    k_12 = k_12_arr[0]

    # Run predictions in parallel using the active pool
    pred_p = np.asarray(
        [_pred_bp_worker(T, X1, k_12, params) for T, X1 in zip(temperature, x1)]
    )

    # Calculate residuals: log(pred_P/exp_P)
    residuals = np.log((pred_p + EPS) / (pressure + EPS))

    # Handle NaNs (failed calculation) by assigning a large penalty
    nan_mask = np.isnan(residuals)
    residuals[nan_mask] = 10.0

    return residuals


def optimize_binary_kij_with_vle(
    smiles_list: List[str],
    initial_kij: float,
) -> Union[float, str]:
    """
    Optimize the kij interaction parameter for a binary mixture with VLE
    experimental data if available.

    Args:
        smiles_list (List[str]): List of SMILES strings [SMILE_1, SMILES_2] for the components.
        initial_kij (float): Initial guess for the kij interaction parameter.
    """

    parameters = [predict_pcsaft_parameters(smiles) for smiles in smiles_list]

    vle = retrieve_vle_for_kij(smiles_list=smiles_list)

    if vle is None:
        return "No experimental data found for kij optimization"

    x1s = vle[:, 0]
    pressures = vle[:, 1]
    temperatures = vle[:, 2]
    try:
        # Optimize
        res = least_squares(
            fun=_loss_fn_bubble_point,
            x0=[initial_kij],
            kwargs={
                "params": parameters,
                "x1": x1s,
                "temperature": temperatures,
                "pressure": pressures,
            },
            jac="2-point",
            method="lm",
            ftol=1e-8,
            xtol=1e-8,
        )
        return res.x[0].item()
    except RuntimeError:
        logger.exception("Failed kij optimization")
        return (
            "Experimental data found for kij optimization "
            "but kij optimization failed, try another initial_kij"
        )


def optimize_binary_kij_for_vle(
    smiles_list: List[str],
    initial_kij: float,
    vle: NDArray[float64],
) -> float:
    """
    Optimize the kij interaction parameter for a binary mixture with VLE
        experimental data if available.

    Args:
        smiles_list (List[str]): List of SMILES strings [SMILE_1, SMILES_2] for the components.
        initial_kij (float): Initial guess for the kij interaction parameter.
        vle: Array of target VLE pressures.
    """

    parameters = [predict_pcsaft_parameters(smiles) for smiles in smiles_list]

    x1s = vle[:, 0]
    pressures = vle[:, 1]
    temperatures = vle[:, 2]
    try:
        # Optimize
        res = least_squares(
            fun=_loss_fn_bubble_point,
            x0=[initial_kij],
            kwargs={
                "params": parameters,
                "x1": x1s,
                "temperature": temperatures,
                "pressure": pressures,
            },
            jac="2-point",
            method="lm",
            ftol=1e-8,
            xtol=1e-8,
        )
        return res.x[0].item()

    except RuntimeError:
        return 0.0
