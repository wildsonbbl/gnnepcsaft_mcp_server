"GNNPCSAFT MCP Server"

from typing import Any, Callable, List

from mcp.server.fastmcp import FastMCP

from .utils import (
    batch_critical_points,
    batch_inchi_to_smiles,
    batch_molecular_weights,
    batch_predict_pcsaft_parameters,
    batch_pure_density,
    batch_pure_h_lv,
    batch_pure_vapor_pressure,
    batch_smiles_to_inchi,
    mixture_density,
    mixture_vapor_pressure,
)

mcp = FastMCP("gnnpcsaft")
fn_list: List[Callable[..., Any]] = [
    batch_predict_pcsaft_parameters,
    batch_critical_points,
    batch_inchi_to_smiles,
    batch_smiles_to_inchi,
    batch_molecular_weights,
    batch_pure_density,
    batch_pure_h_lv,
    batch_pure_vapor_pressure,
    mixture_density,
    mixture_vapor_pressure,
]
for fn in fn_list:
    mcp.add_tool(fn)


def run():
    "run stdio"
    mcp.run("stdio")
