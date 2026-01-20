#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
MCP server that exposes rocminfo as a tool.

This server provides AI assistants with access to AMD ROCm system information
through the rocminfo utility. It can be used to query GPU capabilities,
HSA agents, and system configuration.
"""

import subprocess
import shutil
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: The 'mcp' package is not installed.")
    print("Please install it with: pip install mcp")
    raise


def find_rocminfo() -> Optional[str]:
    """
    Find the rocminfo executable in the system.
    
    Returns:
        Path to rocminfo executable, or None if not found.
    """
    # First try to find it in PATH
    rocminfo_path = shutil.which("rocminfo")
    if rocminfo_path:
        return rocminfo_path
    
    # Try common ROCm installation paths
    common_paths = [
        "/opt/rocm/bin/rocminfo",
        "/opt/rocm-7.1.0/bin/rocminfo",
        "/usr/bin/rocminfo",
    ]
    
    for path in common_paths:
        if shutil.os.path.exists(path):
            return path
    
    return None


def run_rocminfo() -> str:
    """
    Execute rocminfo and return its output.
    
    Returns:
        String containing rocminfo output or error message.
    """
    rocminfo_path = find_rocminfo()
    
    if not rocminfo_path:
        return (
            "Error: rocminfo not found on this system.\n"
            "Please ensure ROCm is installed and rocminfo is in your PATH.\n"
            "Common locations: /opt/rocm/bin/rocminfo"
        )
    
    try:
        result = subprocess.run(
            [rocminfo_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=30  # 30 second timeout
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: rocminfo command timed out after 30 seconds."
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running rocminfo (exit code {e.returncode}):\n"
        if e.stderr:
            error_msg += f"stderr: {e.stderr}\n"
        if e.stdout:
            error_msg += f"stdout: {e.stdout}\n"
        return error_msg
    except Exception as e:
        return f"Unexpected error running rocminfo: {type(e).__name__}: {str(e)}"


# Create MCP server
mcp = FastMCP(
    name="rocminfo-server",
    version="1.0.0",
    description="MCP server exposing AMD ROCm rocminfo utility for querying GPU and HSA system information"
)


@mcp.tool()
def rocminfo() -> str:
    """
    Run the rocminfo command to get AMD ROCm system information.
    
    Returns detailed information about:
    - HSA System Attributes
    - HSA Agents (CPUs and GPUs)
    - GPU properties (compute units, memory, ISA)
    - Cache hierarchy
    - Memory pools
    
    Returns:
        String containing the complete rocminfo output or an error message.
    """
    return run_rocminfo()


if __name__ == "__main__":
    # Run the MCP server using stdio transport (default)
    mcp.run()
