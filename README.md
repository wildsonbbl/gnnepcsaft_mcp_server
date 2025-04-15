# GNNEPCSAFT MCP Server

## Overview

This is the Model Context Protocol ([MCP](https://modelcontextprotocol.io/introduction)) server implementation for [GNNePCSAFT](https://github.com/wildsonbbl/gnnepcsaft) tools. The server handles communication and context management between models and clients using the MCP protocol.

## Features

- Model context management
- Protocol handling for model communication
- Client request processing
- Context state synchronization

## Installation

You're gonna need [pipx](https://pipx.pypa.io/latest/installation/) for this.

```bash
pipx install gnnepcsaft-mcp-server
```

## Usage

stdio command to start the MCP server:

```json
{
  "mcpServers": {
    "gnnepcsaft": {
      "command": "gnnepcsaftmcp",
      "args": []
    }
  }
}
```

## License

GNU General Public License v3.0
