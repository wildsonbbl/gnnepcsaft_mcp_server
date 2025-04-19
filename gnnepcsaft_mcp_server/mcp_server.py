"GNNePCSAFT MCP Server"

from typing import Any, Callable, List

from gnnepcsaft.data.rdkit_util import mw
from gnnepcsaft.epcsaft.epcsaft_feos import (
    critical_points_feos,
    pure_den_feos,
    pure_h_lv_feos,
    pure_vp_feos,
)
from mcp.server.fastmcp import FastMCP

from .utils import (
    inchi_to_smiles,
    mixture_density,
    mixture_phase,
    mixture_vapor_pressure,
    prediction,
    pubchem_description,
    pure_phase,
    smiles_to_inchi,
)

mcp = FastMCP("gnnepcsaft")
fn_list: List[Callable[..., Any]] = [
    pure_vp_feos,
    pure_den_feos,
    mixture_density,
    mixture_vapor_pressure,
    pure_phase,
    mixture_phase,
    pubchem_description,
    mw,
    smiles_to_inchi,
    prediction,
    inchi_to_smiles,
    pure_h_lv_feos,
    critical_points_feos,
]
for fn in fn_list:
    mcp.add_tool(fn)


def run():
    "run stdio"
    mcp.run("stdio")
