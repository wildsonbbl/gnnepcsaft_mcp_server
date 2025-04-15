"mcp server"

from typing import Any, Callable, List

from gnnepcsaft.data.rdkit_util import inchitosmiles, mw, smilestoinchi
from gnnepcsaft.epcsaft.epcsaft_feos import (
    mix_den_feos,
    mix_vp_feos,
    pure_den_feos,
    pure_h_lv_feos,
    pure_vp_feos,
)
from mcp.server.fastmcp import FastMCP

from .utils import mixture_phase, prediction, pubchem_description, pure_phase

mcp = FastMCP("gnnepcsaft")
fn_list: List[Callable[..., Any]] = [
    pure_vp_feos,
    pure_den_feos,
    mix_den_feos,
    mix_vp_feos,
    pure_phase,
    mixture_phase,
    pubchem_description,
    mw,
    smilestoinchi,
    prediction,
    inchitosmiles,
    pure_h_lv_feos,
]
for fn in fn_list:
    mcp.add_tool(fn)


def run():
    "run stdio"
    mcp.run("stdio")
