"""Configuration for ParaView Agent."""

from pathlib import Path
from typing import Any, List

from pydantic import Field

from ..mcp._config import McpAgentConfig
from ...tools.mcp import NamedMcpServerParams


class ParaViewAgentConfig(McpAgentConfig):
    """
    Configuration for ParaView Agent that combines MCP agent capabilities
    with Docker container management for ParaView GUI.

    Inherits all MCP agent configuration plus adds ParaView-specific settings.
    """

    # ParaView Docker settings
    paraview_image: str = "paraview-novnc:latest"
    novnc_port: int = 6080
    vnc_port: int = 5900
    pvserver_port: int = 11111
    data_dir: Path | None = None
    auto_gui_connect: bool = True
    inside_docker: bool = False  # Default to False for running outside Docker
    network_name: str = "my-network"  # Docker network name (must match across all containers)
    width: int = 1920  # VNC display width for better fit in Live View
    height: int = 1080  # VNC display height for better fit in Live View
    gui_connect_wait_time: int = 6  # Seconds to wait for GUI to connect before MCP (adjust for machine speed)

    # MCP server settings for ParaView (will be set programmatically)
    paraview_mcp_python_path: str = "/Users/kuangshiai/miniconda3/envs/paraview_mcp/bin/python"
    paraview_mcp_script_path: str = "src/paraview_mcp/paraview_mcp_server.py"

    # Override default description
    description: str = "The ParaView Agent can control ParaView for 3D visualization and scientific data analysis."

    # Default system message for ParaView agent
    system_message: str = Field(
        default="""You are the ParaView Agent, a specialized assistant for 3D scientific visualization using ParaView.

Your capabilities include:
- Loading and visualizing scientific data (VTK, EXODUS, CSV, etc.)
- Creating isosurfaces, slices, and clips
- Volume rendering with custom color maps and opacity
- Streamline visualization for vector fields
- Camera control and view manipulation
- Exporting visualizations and data

When working with ParaView:
1. Start by loading data files
2. Explore available data arrays to understand the dataset
3. Create appropriate visualizations based on the data type
4. **ALWAYS take screenshots after creating or modifying visualizations**
5. Adjust camera angles for better views
6. Take another screenshot after camera adjustments

IMPORTANT Screenshot Guidelines:
- Call get_screenshot() after EVERY visualization operation (load, filter, color change, etc.)
- Screenshots are displayed in the "Actions" tab for the user to review your progress
- When you call get_screenshot(), the image is automatically shown to the user
- DO NOT include base64 image data in your response text
- Simply mention that you've captured a screenshot

The ParaView GUI is available through a web browser at the noVNC URL for live viewing."""
    )

    # Default MCP servers list (will be populated with ParaView MCP server)
    mcp_servers: List[NamedMcpServerParams] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set up ParaView MCP server if not already configured."""
        super().model_post_init(__context)

        # If no MCP servers configured, add the default ParaView MCP server
        if not self.mcp_servers:
            from autogen_ext.tools.mcp import StdioServerParams

            self.mcp_servers = [
                NamedMcpServerParams(
                    server_name="ParaView",
                    server_params=StdioServerParams(
                        command=self.paraview_mcp_python_path,
                        args=[
                            self.paraview_mcp_script_path,
                            "--server",
                            "localhost",
                            "--port",
                            str(self.pvserver_port),
                        ],
                    ),
                )
            ]
