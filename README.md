# GNNPCSAFT MCP Server

GNNPCSAFT MCP Server is an implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) for [GNNPCSAFT](https://github.com/wildsonbbl/gnnepcsaft) tools. GNNPCSAFT leverages Graph Neural Networks (GNNs) to estimate [PC-SAFT](https://en.wikipedia.org/wiki/PC-SAFT) pure-component parameters, allowing property predictions such as density and vapor pressure for any molecule or mixture. [FeOs](https://github.com/feos-org/feos) is used for PC-SAFT calculations.

Other implementations with GNNPCSAFT:

- [GNNPCSAFT CLI](https://github.com/wildsonbbl/gnnepcsaftcli)
- [GNNPCSAFT APP](https://github.com/wildsonbbl/gnnpcsaftapp)
- [GNNPCSAFT Webapp](https://github.com/wildsonbbl/gnnepcsaftwebapp)
- [GNNPCSAFT Chat](https://github.com/wildsonbbl/gnnpcsaftchat)

## How to Use

### Installation

You need [uvx](https://docs.astral.sh/uv/) installed.

### Starting the Server

```bash
uvx --from gnnepcsaft-mcp-server gnnpcsaftmcp
```

### Example: Claude Desktop Configuration

```json
{
  "mcpServers": {
    "gnnpcsaft": {
      "command": "uvx",
      "args": ["--from", "gnnepcsaft-mcp-server", "gnnpcsaftmcp"]
    }
  }
}
```

---

## License

GNU General Public License v3.0
