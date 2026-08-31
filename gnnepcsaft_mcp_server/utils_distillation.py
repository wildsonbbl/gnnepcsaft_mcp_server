"""Utilities for thermodynamic calculations in fluid mixtures.

This module provides shared data structures and helpers for distillation
calculations using PC-SAFT models.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class McCabeThieleParams:
    """Inputs for a binary McCabe-Thiele distillation calculation.

    Parameters
    ----------
    equilibrium_x : List[float]
        Liquid composition values x of the equilibrium curve.
    equilibrium_y : List[float]
        Vapor composition values y corresponding to each x in the equilibrium curve.
    feed_composition : float
        Feed mole fraction of the light component, x_F, in the feed stream.
    distillate_composition : float
        Target distillate mole fraction of the light component, x_D.
    bottoms_composition : float
        Target bottoms mole fraction of the light component, x_B.
    reflux_ratio : float
        External reflux ratio R = L/D, strictly positive for the rectifying section.
    feed_quality : float, optional
        Feed quality q parameter. q = 1 for saturated liquid, q = 0 for saturated
        vapor, q > 1 for subcooled liquid, and q < 0 for superheated vapor.
    max_stages : int, optional
        Maximum number of stages allowed before stopping with an error.
    """

    equilibrium_x: List[float]
    equilibrium_y: List[float]
    feed_composition: float
    distillate_composition: float
    bottoms_composition: float
    reflux_ratio: float
    feed_quality: float = 1.0
    max_stages: int = 100


def mccabe_thiele(
    params: McCabeThieleParams,
) -> Dict[str, object]:
    """Calculate a binary distillation stage count with the McCabe-Thiele method.

    This function builds the equilibrium curve, the rectifying and stripping operating
    lines, and then steps horizontally/vertically between them to estimate the number
    of ideal stages needed to achieve the specified distillate and bottoms compositions.

    Parameters
    ----------
    params : McCabeThieleParams
        Input data for the column:
        - equilibrium_x, equilibrium_y: binary VLE curve for the light key
        - feed_composition: feed light-key mole fraction x_F
        - distillate_composition: distillate light-key mole fraction x_D
        - bottoms_composition: bottoms light-key mole fraction x_B
        - reflux_ratio: external reflux ratio R
        - feed_quality: feed q parameter
        - max_stages: maximum number of stages allowed

    Returns
    -------
    Dict[str, object]
        A dictionary with the computed stage geometry and summary data. It contains:
        - "number_of_stages": total number of ideal stages counted
        - "feed_stage": stage number where the feed stage is crossed, if any
        - "stage_x": x coordinates of the stepped path used for plotting
        - "stage_y": y coordinates of the stepped path used for plotting
        - "rectifying_line": [slope, intercept] of the rectifying operating line
        - "stripping_line": [slope, intercept] of the stripping operating line
        - "q_line": [slope, intercept] of the feed line, or [None, x_F] for q = 1
        - "feed_intersection": [x_int, y_int] intersection of the operating lines

    Notes
    -----
    The path is constructed as a staircase on the x-y diagram. Each horizontal leg is
    an equilibrium step and each vertical leg moves along the operating line.
    """
    x_eq = np.asarray(params.equilibrium_x, dtype=float)
    y_eq = np.asarray(params.equilibrium_y, dtype=float)
    if x_eq.ndim != 1 or y_eq.ndim != 1 or len(x_eq) != len(y_eq) or len(x_eq) < 2:
        raise ValueError(
            "Equilibrium x and y must be equally sized 1-D arrays with at least two points"
        )
    if not np.all(np.isfinite(x_eq)) or not np.all(np.isfinite(y_eq)):
        raise ValueError("Equilibrium x and y must contain only finite values")
    if np.any(np.diff(x_eq) <= 0) or np.any(np.diff(y_eq) <= 0):
        raise ValueError("Equilibrium x and y must be strictly increasing")
    if x_eq[0] > 0.0 or x_eq[-1] < 1.0 or y_eq[0] > 0.0 or y_eq[-1] < 1.0:
        raise ValueError("Equilibrium data must cover the composition range 0 to 1")

    x_f = params.feed_composition
    x_d = params.distillate_composition
    x_b = params.bottoms_composition
    q = params.feed_quality
    reflux = params.reflux_ratio
    if not all(np.isfinite(value) for value in (x_f, x_d, x_b)):
        raise ValueError("Feed, distillate, and bottoms compositions must be finite")
    if not 0.0 <= x_b < x_f < x_d <= 1.0:
        raise ValueError("Require 0 <= bottoms < feed < distillate <= 1")
    if reflux <= 0.0 or not np.isfinite(reflux):
        raise ValueError("Reflux ratio must be finite and positive")
    if not np.isfinite(q):
        raise ValueError("Feed quality must be finite")
    if params.max_stages < 1:
        raise ValueError("max_stages must be at least 1")

    rectifying_slope = reflux / (reflux + 1.0)
    rectifying_intercept = x_d / (reflux + 1.0)

    if np.isclose(q, 1.0):
        feed_intersection_x = x_f
        q_line = [None, x_f]
    else:
        q_slope = q / (q - 1.0)
        q_intercept = -x_f / (q - 1.0)
        line_denominator = rectifying_slope - q_slope
        if np.isclose(line_denominator, 0.0):
            raise ValueError("Rectifying and q-lines are parallel")
        feed_intersection_x = (q_intercept - rectifying_intercept) / (line_denominator)
        q_line = [q_slope, q_intercept]
    feed_intersection_y = rectifying_slope * feed_intersection_x + rectifying_intercept
    if not x_b < feed_intersection_x <= 1.0 or not 0.0 <= feed_intersection_y <= 1.0:
        raise ValueError(
            "Operating lines do not intersect within the composition range"
        )

    stripping_slope = (feed_intersection_y - x_b) / (feed_intersection_x - x_b)
    stripping_intercept = x_b * (1.0 - stripping_slope)

    def operating_line(x_value: float) -> float:
        if x_value >= feed_intersection_x:
            return rectifying_slope * x_value + rectifying_intercept
        return stripping_slope * x_value + stripping_intercept

    stage_x = [x_d]
    stage_y = [x_d]
    feed_stage = None
    current_y = x_d
    stages = 0
    while stages < params.max_stages:
        current_x = float(np.interp(current_y, y_eq, x_eq))
        stage_x.extend([current_x, current_x])
        stage_y.extend([current_y, operating_line(current_x)])
        stages += 1
        if feed_stage is None and current_x <= feed_intersection_x:
            feed_stage = stages
        if current_x <= x_b:
            break
        current_y = stage_y[-1]
    else:
        raise RuntimeError(
            "Maximum number of stages reached before reaching bottoms composition"
        )

    return {
        "number_of_stages": stages,
        "feed_stage": feed_stage,
        "stage_x": stage_x,
        "stage_y": stage_y,
        "rectifying_line": [rectifying_slope, rectifying_intercept],
        "stripping_line": [stripping_slope, stripping_intercept],
        "q_line": q_line,
        "feed_intersection": [feed_intersection_x, feed_intersection_y],
    }


def distillation_column(
    equilibrium_x: List[float],
    equilibrium_y: List[float],
    feed_composition: float,
    distillate_composition: float,
    bottoms_composition: float,
    reflux_ratio: float,
    feed_quality: float = 1.0,
    max_stages: int = 100,
) -> Dict[str, object]:
    """Calculates the binary McCabe-Thiele distillation calculation.

    Parameters
    ----------
    equilibrium_x : List[float]
        Liquid compositions x of the VLE curve, typically from ``mix_vle(...)["x0"]``.
    equilibrium_y : List[float]
        Vapor compositions y of the VLE curve, typically from ``mix_vle(...)["y0"]``.
    feed_composition : float
        Feed light-key mole fraction x_F.
    distillate_composition : float
        Distillate light-key mole fraction x_D.
    bottoms_composition : float
        Bottoms light-key mole fraction x_B.
    reflux_ratio : float
        Reflux ratio R = L/D.
    feed_quality : float, optional
        Feed quality parameter q. Defaults to 1.0 for saturated liquid feed.
    max_stages : int, optional
        Maximum number of ideal stages to permit before failing. Defaults to 100.

    Returns
    -------
    Dict[str, object]
        A dictionary with the computed stage geometry and summary data. It contains:
        - "number_of_stages": total number of ideal stages counted
        - "feed_stage": stage number where the feed stage is crossed, if any
        - "stage_x": x coordinates of the stepped path used for plotting
        - "stage_y": y coordinates of the stepped path used for plotting
        - "rectifying_line": [slope, intercept] of the rectifying operating line
        - "stripping_line": [slope, intercept] of the stripping operating line
        - "q_line": [slope, intercept] of the feed line, or [None, x_F] for q = 1
        - "feed_intersection": [x_int, y_int] intersection of the operating lines

    Examples
    --------
    >>> result = distillation_column(
    ...     equilibrium_x=[0.0, 0.2, 0.5, 0.8, 1.0],
    ...     equilibrium_y=[0.0, 0.35, 0.7, 0.9, 1.0],
    ...     feed_composition=0.45,
    ...     distillate_composition=0.9,
    ...     bottoms_composition=0.05,
    ...     reflux_ratio=2.0,
    ... )
    >>> result["number_of_stages"]
    14
    """
    return mccabe_thiele(
        McCabeThieleParams(
            equilibrium_x=equilibrium_x,
            equilibrium_y=equilibrium_y,
            feed_composition=feed_composition,
            distillate_composition=distillate_composition,
            bottoms_composition=bottoms_composition,
            reflux_ratio=reflux_ratio,
            feed_quality=feed_quality,
            max_stages=max_stages,
        )
    )
