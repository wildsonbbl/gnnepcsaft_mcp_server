# GNNEPCSAFT MCP Server

## Overview

This is the Model Context Protocol (MCP) server implementation for [GNNePCSAFT](https://github.com/wildsonbbl/gnnepcsaft) tools. The server handles communication and context management between models and clients using the MCP protocol.

## Features

- Model context management
- Protocol handling for model communication
- Client request processing
- Context state synchronization

## Installation

```bash
pipx install gnnepcsaft_mcp_server
```

## Usage

stdio command to start the MCP server:

```json
{
  "mcpServers": {
    "gnnepcsaft": {
      "command": "gnnepcsaft_mcp_server",
      "args": []
    }
  }
}
```

## License

GNU General Public License v3.0
