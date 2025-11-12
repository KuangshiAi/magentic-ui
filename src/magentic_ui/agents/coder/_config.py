"""Configuration for Coder Agent with PVPython and RAG capabilities."""

from pathlib import Path
from typing import Any

from autogen_core import ComponentModel
from pydantic import BaseModel, Field


class CoderAgentConfig(BaseModel):
    """
    Configuration for Coder Agent that combines general code execution
    with ParaView/PVPython capabilities using RAG.

    This agent:
    - Can write and execute general Python code
    - Uses RAG to find relevant ParaView operations from operations.json
    - Generates pvpython code based on user requests
    - Executes code in Docker container with access to pvserver
    - Iteratively refines code based on execution errors
    """

    name: str
    model_client: ComponentModel

    # General coder settings
    description: str = Field(
        default="""An agent that can write and execute code to solve tasks or use its language skills to summarize, write, solve math and logic problems.
It understands images and can use them to help it complete the task.
It can access files if given the path and manipulate them using python code. Use the coder if you want to manipulate a file or read a csv or excel files.
In a single step when you ask the agent to do something: it can write code, and then immediately execute the code. If there are errors it can debug the code and try again.

Additionally, this agent has ParaView/PVPython capabilities:
- Generate pvpython visualization scripts based on natural language requests
- Use a knowledge base of ParaView operations to create accurate code
- Handle complex visualization tasks including data loading, filters, volume rendering, color mapping, camera positioning, and screenshot generation"""
    )

    max_debug_rounds: int = 3
    summarize_output: bool = False
    model_context_token_limit: int = 128000

    # ParaView/PVPython RAG settings
    enable_pvpython_rag: bool = True  # Whether to enable PVPython RAG capabilities
    operations_json_path: Path | None = None  # Path to operations.json database for RAG
    top_k_operations: int = 5  # Number of similar operations to retrieve

    # PVServer connection settings
    pvserver_host: str = "paraview-server"  # Docker container hostname
    pvserver_port: int = 11111

    # Python environment settings
    python_env_image: str = "ghcr.io/microsoft/magentic-ui-python-env:latest"
    workspace_dir: Path | None = None  # Workspace directory to mount
    inside_docker: bool = True  # Whether backend is running in Docker
    network_name: str = "my-network"  # Docker network name

    # Debug settings
    debug_dir: Path | None = None  # Directory to save generated code for debugging

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization setup."""
        super().model_post_init(__context)

        # Set default operations path if not provided and PVPython RAG is enabled
        if self.enable_pvpython_rag and self.operations_json_path is None:
            self.operations_json_path = Path(__file__).parent / "operations.json"

        # Validate operations.json exists if RAG is enabled
        if self.enable_pvpython_rag and self.operations_json_path:
            if not self.operations_json_path.exists():
                raise FileNotFoundError(
                    f"Operations JSON not found at {self.operations_json_path}. "
                    "Please ensure the operations.json file exists in the agent directory, "
                    "or set enable_pvpython_rag=False to disable PVPython RAG capabilities."
                )
