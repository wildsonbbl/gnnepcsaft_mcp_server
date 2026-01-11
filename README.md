# GNNEPCSAFT MCP Server

GNNEPCSAFT MCP Server is an implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) for [GNNePCSAFT](https://github.com/wildsonbbl/gnnepcsaft) tools. GNNePCSAFT leverages Graph Neural Networks (GNNs) to estimate [PC-SAFT](https://en.wikipedia.org/wiki/PC-SAFT) pure-component parameters, allowing property predictions such as density and vapor pressure for any molecule or mixture. [FeOs](https://github.com/feos-org/feos) is used for PC-SAFT calculations.

## How to Use

### Installation

You need [uvx](https://docs.astral.sh/uv/) installed.

### Starting the Server

```bash
uvx --from gnnepcsaft-mcp-server gnnepcsaftmcp
```

### Example: Claude Desktop Configuration

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

---

## License

GNU General Public License v3.0
