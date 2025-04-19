"Utils for MCP Server"

from json import loads
from pathlib import Path
from typing import List, Literal, Tuple
from urllib.parse import quote
from urllib.request import HTTPError, urlopen

import numpy as np
import onnxruntime as ort
from gnnepcsaft.data.ogb_utils import smiles2graph
from gnnepcsaft.data.rdkit_util import assoc_number, inchitosmiles, smilestoinchi
from gnnepcsaft.epcsaft.epcsaft_feos import mix_den_feos, mix_vp_feos

file_dir = Path(__file__).parent
model_dir = file_dir / "models"
ort.set_default_logger_severity(3)

msigmae_onnx = ort.InferenceSession(model_dir / "msigmae_7.onnx")
assoc_onnx = ort.InferenceSession(model_dir / "assoc_8.onnx")


def prediction(
    smiles: str,
) -> List[float]:
    """Predict ePC-SAFT parameters
    `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`

    Args:
      smiles (str): SMILES of the molecule.
    """
    lower_bounds = np.asarray([1.0, 1.9, 50.0, 0.0, 0.0, 0, 0, 0])
    upper_bounds = np.asarray([25.0, 4.5, 550.0, 0.9, 5000.0, np.inf, np.inf, np.inf])

    inchi = smilestoinchi(smiles)

    graph = smiles2graph(smiles)
    na, nb = assoc_number(inchi)
    x, edge_index, edge_attr = (
        graph["node_feat"],
        graph["edge_index"],
        graph["edge_feat"],
    )

    assoc = 10 ** (
        assoc_onnx.run(
            None,
            {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
            },
        )[0][0]
        * np.asarray([-1.0, 1.0])
    )
    if na == 0 and nb == 0:
        assoc *= 0
    msigmae = msigmae_onnx.run(
        None,
        {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        },
    )[0][0]
    munanb = np.asarray([0.0, na, nb])
    pred = np.hstack([msigmae, assoc, munanb], dtype=np.float64)
    np.clip(pred, lower_bounds, upper_bounds, out=pred)

    return pred.tolist()  # type: ignore


def pure_phase(
    vapor_pressure: float, system_pressure: float
) -> Literal["liquid", "vapor"]:
    """
    Given the vapor pressure and system pressure, return the phase of the molecule.
    Both pressures must be in the same unit.

    Args:
        vapor_pressure (float): The calculated vapor pressure of the pure component.
        system_pressure (float): The actual system pressure.

    """
    assert isinstance(vapor_pressure, (int, float)), "vapor_pressure must be a number"
    assert isinstance(system_pressure, (int, float)), "system_pressure must be a number"
    assert vapor_pressure > 0, "vapor_pressure must be positive"
    assert system_pressure > 0, "system_pressure must be positive"

    return "liquid" if vapor_pressure < system_pressure else "vapor"


def mixture_phase(
    bubble_point: float,
    dew_point: float,
    system_pressure: float,
) -> Literal["liquid", "vapor", "two-phase"]:
    """
    Given the bubble/dew point of the mixture and the system pressure,
    return the phase of the mixture.
    All pressures must be in the same unit.

    Args:
        bubble_point (float): The calculated bubble point of the mixture.
        dew_point (float): The calculated dew point of the mixture.
        system_pressure (float): The actual system pressure.
    """
    assert isinstance(bubble_point, (int, float)), "bubble_point must be a number"
    assert isinstance(dew_point, (int, float)), "dew_point must be a number"
    assert isinstance(system_pressure, (int, float)), "system_pressure must be a number"
    assert bubble_point > 0, "bubble_point must be positive"
    assert dew_point > 0, "dew_point must be positive"
    assert system_pressure > 0, "system_pressure must be positive"
    return (
        "liquid"
        if bubble_point < system_pressure
        else ("two-phase" if dew_point <= system_pressure else "vapor")
    )


def pubchem_description(inchi: str) -> str:
    """
    Look for information on PubChem for the InChI.

    Args:
        inchi (str): The InChI of the molecule.
    """
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/description/json?inchi="
        + quote(inchi, safe="")
    )
    try:
        with urlopen(url) as ans:
            ans = loads(ans.read().decode("utf8").strip())
    except (TypeError, HTTPError, ValueError):
        ans = "no data available on this molecule in PubChem."
    return ans


def mixture_density(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: List[List[float]],
) -> float:
    """Calculates mixture liquid density (mol/m³) with ePC-SAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """

    return mix_den_feos(parameters, state, kij_matrix)


def mixture_vapor_pressure(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: List[List[float]],
) -> Tuple[float, float]:
    """Calculates mixture `(Bubble point (Pa), Dew point (Pa))` with ePC-SAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, molefractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """

    return mix_vp_feos(parameters, state, kij_matrix)


def smiles_to_inchi(smiles: str) -> str:
    """Transform SMILES to InChI.

    Args:
        smiles (str): SMILES
    """
    return smilestoinchi(smiles)


def inchi_to_smiles(inchi: str) -> str:
    """Transform InChI to SMILES.

    Args:
        inchi (str): InChI
    """
    return inchitosmiles(inchi)
