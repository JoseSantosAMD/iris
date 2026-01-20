# MCP Tools for Iris

This directory contains MCP (Model Context Protocol) servers that expose various tools and utilities for AI assistants to interact with the Iris framework and AMD ROCm ecosystem.

## What is MCP?

The Model Context Protocol (MCP) is an open protocol that standardizes how AI assistants connect to external tools and data sources. MCP servers expose tools that AI assistants can discover and invoke to perform actions or retrieve information.

## Available Tools

### rocminfo Server

The `rocminfo_server.py` provides access to the AMD ROCm system information utility through MCP.

**Tool:** `rocminfo`
- **Description:** Run the rocminfo command to get AMD ROCm system information
- **Parameters:** None
- **Returns:** Complete rocminfo output including:
  - HSA System Attributes
  - HSA Agents (CPUs and GPUs)
  - GPU properties (compute units, memory, ISA)
  - Cache hierarchy
  - Memory pools

## Installation

### Prerequisites

1. **ROCm**: Ensure ROCm is installed and `rocminfo` is available
   ```bash
   which rocminfo
   # Should return: /opt/rocm/bin/rocminfo or similar
   ```

2. **Python Dependencies**: Install the MCP Python SDK
   ```bash
   pip install mcp
   ```

### Verify Installation

Test the server directly:
```bash
cd /path/to/iris
python mcp/rocminfo_server.py
```

The server will start and wait for MCP client connections via stdio.

## Usage

### With Claude Desktop

Add the server to your Claude Desktop configuration file:

**macOS/Linux:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "iris-rocminfo": {
      "command": "python",
      "args": ["/absolute/path/to/iris/mcp/rocminfo_server.py"]
    }
  }
}
```

After restarting Claude Desktop, you can ask Claude to use the `rocminfo` tool:

```
Can you run rocminfo and tell me what GPUs are available?
```

### With Other MCP Clients

Any MCP-compatible client can connect to the server. The server uses stdio transport by default, which is the most widely supported method.

Example using an MCP client library:
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["/path/to/iris/mcp/rocminfo_server.py"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        print(tools)
        
        # Call the rocminfo tool
        result = await session.call_tool("rocminfo", {})
        print(result)
```

### Direct Testing

You can also test the server directly from Python:
```bash
python -c "from mcp.rocminfo_server import run_rocminfo; print(run_rocminfo())"
```

## Configuration

The `rocminfo_server.py` automatically searches for the `rocminfo` executable in:
1. System PATH
2. `/opt/rocm/bin/rocminfo`
3. `/opt/rocm-7.1.0/bin/rocminfo`
4. `/usr/bin/rocminfo`

If your `rocminfo` is in a different location, you can modify the `common_paths` list in the `find_rocminfo()` function.

## Security Considerations

⚠️ **Important Security Notes:**

1. **Command Execution**: This server executes shell commands (`rocminfo`). Only use it in trusted environments.
2. **No Input Validation**: The `rocminfo` tool takes no user input, making it relatively safe. However, be cautious when adding tools that accept parameters.
3. **Timeout Protection**: Commands are limited to 30 seconds to prevent hanging.
4. **Access Control**: Ensure only authorized users can access the MCP server. In Claude Desktop, this means securing your configuration file.

## Troubleshooting

### "rocminfo not found" Error

If you see this error:
```
Error: rocminfo not found on this system.
```

**Solutions:**
1. Verify ROCm is installed: `ls /opt/rocm/bin/rocminfo`
2. Add rocminfo to PATH: `export PATH=$PATH:/opt/rocm/bin`
3. Check ROCm version: `rocminfo --version` (if found)

### "mcp package not installed" Error

Install the MCP Python SDK:
```bash
pip install mcp
```

### Server Not Responding

1. Check the server starts without errors: `python mcp/rocminfo_server.py`
2. Verify the configuration file path is correct (use absolute paths)
3. Check Claude Desktop logs (Help > Show Logs)

## Development

### Adding New Tools

To add more MCP tools to this directory:

1. Create a new Python file (e.g., `new_tool_server.py`)
2. Follow the pattern in `rocminfo_server.py`:
   ```python
   from mcp.server.fastmcp import FastMCP
   
   mcp = FastMCP(name="tool-name", version="1.0.0")
   
   @mcp.tool()
   def my_tool(param: str) -> str:
       """Tool description."""
       # Tool implementation
       return result
   
   if __name__ == "__main__":
       mcp.run()
   ```
3. Document it in this README
4. Update the Iris main README to reference the new tool

### Testing

Test tools locally before deploying:
```bash
# Test the server starts
python mcp/rocminfo_server.py

# Test the function directly
python -c "from mcp.rocminfo_server import run_rocminfo; print(run_rocminfo())"
```

## Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [rocminfo Documentation](https://rocm.docs.amd.com/projects/rocminfo/en/latest/)

## License

SPDX-License-Identifier: MIT  
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
