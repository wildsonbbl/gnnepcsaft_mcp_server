"GNNePCSAFT MCP Server"

from typing import Any, Callable, List

from mcp.server.fastmcp import FastMCP

from .utils import (
    batch_convert_pure_density_to_kg_per_m3,
    batch_critical_points,
    batch_inchi_to_smiles,
    batch_molecular_weights,
    batch_pa_to_bar,
    batch_predict_epcsaft_parameters,
    batch_pure_density,
    batch_pure_h_lv,
    batch_pure_vapor_pressure,
    batch_smiles_to_inchi,
    mixture_density,
    mixture_phase,
    mixture_vapor_pressure,
    pubchem_description,
    pure_phase,
)

mcp = FastMCP("gnnepcsaft")
fn_list: List[Callable[..., Any]] = [
    batch_convert_pure_density_to_kg_per_m3,
    batch_critical_points,
    batch_inchi_to_smiles,
    batch_molecular_weights,
    batch_pa_to_bar,
    batch_predict_epcsaft_parameters,
    batch_pure_density,
    batch_pure_h_lv,
    batch_pure_vapor_pressure,
    batch_smiles_to_inchi,
    mixture_density,
    mixture_phase,
    mixture_vapor_pressure,
    pubchem_description,
    pure_phase,
]
for fn in fn_list:
    mcp.add_tool(fn)


def run():
    "run stdio"
    mcp.run("stdio")
