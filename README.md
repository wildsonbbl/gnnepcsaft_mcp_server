# GNNEPCSAFT MCP Server

## Overview

This is the Model Context Protocol ([MCP](https://modelcontextprotocol.io/introduction)) server implementation for [GNNePCSAFT](https://github.com/wildsonbbl/gnnepcsaft) tools. The server handles communication and context management between LLM models and clients using the MCP protocol. GNNePCSAFT is a Graph Neural Network ([GNN](https://en.wikipedia.org/wiki/Graph_neural_network)) that can estimate [ePC-SAFT](https://en.wikipedia.org/wiki/PC-SAFT) parameters. This allows thermodynamic calculations like density and vapor pressure without experimental data for any molecule. [FeOS](https://github.com/feos-org/feos) is used for the PC-SAFT calculations.

## Features

- Calculates density, vapor pressure, enthalpy of vaporization and critical points
- Estimates ePC-SAFT parameters
- Thermodynamic calculations for pure components and mixtures
- Collects information from PubChem, if any, for any molecule
- Allows thermodynamics-aware LLMs

## Installation

You're gonna need [uvx](https://docs.astral.sh/uv/).

## Usage

The command to start the server would be:

```bash
uvx --from gnnepcsaft-mcp-server gnnepcsaftmcp
```

[Claude Desktop](https://claude.ai/download) config to start the MCP server:

```json
{
  "mcpServers": {
    "gnnepcsaft": {
      "command": "uvx",
      "args": ["--from", "gnnepcsaft-mcp-server", "gnnepcsaftmcp"]
    }
  }
}
```

## License

GNU General Public License v3.0
