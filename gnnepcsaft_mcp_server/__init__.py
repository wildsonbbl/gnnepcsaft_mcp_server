"GNNePCSAFT MCP Server"

from .mcp_server import mcp

if __name__ == "__main__":
    mcp.run("stdio")
