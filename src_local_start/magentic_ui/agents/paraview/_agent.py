"""ParaView Agent implementation."""

from typing import Any, AsyncGenerator, List, Sequence

from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    BaseTextChatMessage,
    TextMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
)
from autogen_core import CancellationToken
from autogen_core.model_context import TokenLimitedChatCompletionContext
from autogen_core.models import ChatCompletionClient

from ..mcp._agent import McpAgent
from ...tools.mcp import AggregateMcpWorkbench, NamedMcpServerParams
from ...tools.paraview_docker import ParaViewDockerManager
from ...tools.paraview_local import ParaViewLocalManager
from ._config import ParaViewAgentConfig


class ParaViewAgent(McpAgent):
    """
    A specialized agent for ParaView 3D visualization that combines MCP tool access
    with Docker container management for ParaView GUI.

    This agent:
    - Extends McpAgent to provide ParaView tool access via MCP protocol
    - Manages a ParaView Docker container with GUI access through noVNC
    - Automatically starts/stops the container when the agent is used
    - Provides visual feedback through a web-accessible ParaView interface

    Args:
        name (str): Name of the agent
        model_client (ChatCompletionClient): LLM client for the agent
        mcp_server_params (List[NamedMcpServerParams] | None): MCP server configurations
        paraview_docker_manager (ParaViewDockerManager | None): Docker manager for ParaView container
        model_context_token_limit (int | None): Token limit for model context
        **kwargs: Additional arguments passed to McpAgent

    Properties:
        novnc_port (int | None): Port for noVNC web interface
        pvserver_port (int | None): Port for ParaView server
        is_container_running (bool): Whether the ParaView container is running

    Example:
        ```python
        from pathlib import Path
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        model_client = OpenAIChatCompletionClient(model="gpt-4o")

        agent = ParaViewAgent(
            name="paraview_agent",
            model_client=model_client,
            data_dir=Path("./data"),
        )

        # Agent will automatically start ParaView container on first use
        # Access GUI at http://localhost:{agent.novnc_port}/vnc.html
        ```
    """

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        *,
        mcp_server_params: List[NamedMcpServerParams] | None = None,
        paraview_docker_manager: ParaViewDockerManager | None = None,
        paraview_local_manager: ParaViewLocalManager | None = None,
        model_context_token_limit: int | None = None,
        **kwargs: Any,
    ):
        # Set _model_client before calling super().__init__ because McpAgent tries to access it
        self._model_client = model_client

        self._paraview_docker_manager = paraview_docker_manager
        self._paraview_local_manager = paraview_local_manager
        self._did_lazy_init = False
        self._paraview_just_initialized = False
        self.novnc_port: int | None = None
        self.pvserver_port: int | None = None

        super().__init__(
            name,
            model_client,
            mcp_server_params=mcp_server_params,
            model_context_token_limit=model_context_token_limit,
            **kwargs,
        )

    @property
    def is_container_running(self) -> bool:
        """Check if the ParaView Docker container is running."""
        if self._paraview_docker_manager is None:
            return False
        return self._paraview_docker_manager.is_running

    async def lazy_init(self) -> None:
        """
        Initialize the ParaView server on first use.

        This method:
        - Starts the ParaView server (Docker or local)
        - Waits for pvserver to be ready
        - Starts ParaView GUI if in local mode
        - Sets up the MCP connection to ParaView
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info("ParaView lazy_init() called")

        if self._did_lazy_init:
            logger.info("ParaView already initialized, skipping")
            return

        # Use local manager if available (preferred), otherwise Docker
        if self._paraview_local_manager is not None:
            logger.info("Starting ParaView locally...")
            # Start the local ParaView server and GUI
            await self._paraview_local_manager.start()

            # Capture port information
            self.pvserver_port = self._paraview_local_manager.pvserver_port
            self.novnc_port = None  # No noVNC for local mode

            logger.info(f"ParaView started locally. pvserver port: {self.pvserver_port}")

            # Mark that we just initialized
            self._paraview_just_initialized = True

        elif self._paraview_docker_manager is not None:
            logger.info("Starting ParaView Docker container...")
            # Start the ParaView container
            await self._paraview_docker_manager.start()

            # Capture port information
            self.novnc_port = self._paraview_docker_manager.novnc_port
            self.pvserver_port = self._paraview_docker_manager.pvserver_port

            logger.info(f"ParaView container started. noVNC port: {self.novnc_port}, pvserver port: {self.pvserver_port}")

            # Mark that we just initialized so we can send the browser address
            self._paraview_just_initialized = True
        else:
            logger.warning("No ParaView manager configured - cannot start ParaView")

        self._did_lazy_init = True

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Process messages and ensure ParaView container is initialized.

        This method:
        - Performs lazy initialization of ParaView container on first call
        - Sends browser address to UI after initialization
        - Forwards messages to the parent MCP agent implementation
        - Marks events for UI display
        """
        # Lazy initialization on first message
        await self.lazy_init()

        # Send initialization message after ParaView is started
        if self._paraview_just_initialized:
            if self._paraview_local_manager is not None:
                # Local mode - ParaView GUI should already be open
                yield TextMessage(
                    source="system",
                    content=f"ParaView started locally. GUI window should be open on your desktop.",
                    metadata={
                        "internal": "no",
                        "type": "info",
                        "pvserver_port": str(self.pvserver_port) if self.pvserver_port else "",
                    },
                )
            elif self._paraview_docker_manager is not None and self.novnc_port is not None and self.novnc_port > 0:
                # Docker mode - provide noVNC link
                yield TextMessage(
                    source="system",
                    content=f"ParaView noVNC interface available at http://localhost:{self.novnc_port}/vnc.html",
                    metadata={
                        "internal": "no",
                        "type": "browser_address",
                        "novnc_port": str(self.novnc_port),
                        "pvserver_port": str(self.pvserver_port) if self.pvserver_port else "",
                    },
                )
            # Reset the flag so we don't send the message again
            self._paraview_just_initialized = False

        # Process messages through parent MCP agent
        async for event in super().on_messages_stream(messages, cancellation_token):
            # Display messages to the UI (inherited from McpAgent)
            yield event

    async def close(self) -> None:
        """
        Close the agent and stop ParaView.

        This method:
        - Stops the ParaView server (local or Docker)
        - Cleans up resources
        """
        if self._paraview_local_manager is not None:
            await self._paraview_local_manager.stop()
        elif self._paraview_docker_manager is not None:
            await self._paraview_docker_manager.stop()

    @classmethod
    def _from_config(cls, config: Any):
        """
        Create a ParaViewAgent instance from configuration.

        Args:
            config: ParaViewAgentConfig or compatible configuration object

        Returns:
            ParaViewAgent: Configured agent instance
        """
        if isinstance(config, ParaViewAgentConfig):
            from pathlib import Path
            from autogen_core.models import ChatCompletionClient

            paraview_docker_manager = None
            paraview_local_manager = None

            # Choose between local and Docker ParaView based on configuration
            if config.use_local_paraview:
                # Create local ParaView manager
                paraview_local_manager = ParaViewLocalManager(
                    conda_env=config.conda_env,
                    pvserver_port=config.pvserver_port,
                    data_dir=Path(config.data_dir) if config.data_dir else None,
                    auto_gui_connect=config.auto_gui_connect,
                    gui_connect_wait_time=config.gui_connect_wait_time,
                )
            else:
                # Create Docker ParaView manager (legacy mode)
                paraview_docker_manager = ParaViewDockerManager(
                    image=config.paraview_image,
                    novnc_port=config.novnc_port,
                    vnc_port=config.vnc_port,
                    pvserver_port=config.pvserver_port,
                    data_dir=Path(config.data_dir) if config.data_dir else None,
                    inside_docker=config.inside_docker,
                    network_name=config.network_name,
                    auto_gui_connect=config.auto_gui_connect,
                    width=config.width,
                    height=config.height,
                    gui_connect_wait_time=config.gui_connect_wait_time,
                )

            # Load the model client
            model_client = ChatCompletionClient.load_component(config.model_client)

            # Build the MCP workbench
            workbench = AggregateMcpWorkbench(named_server_params=config.mcp_servers)

            # Create the agent
            return cls(
                name=config.name,
                model_client=model_client,
                mcp_server_params=config.mcp_servers,
                paraview_docker_manager=paraview_docker_manager,
                paraview_local_manager=paraview_local_manager,
                model_context_token_limit=config.model_context_token_limit,
                description=config.description,
                system_message=config.system_message,
            )

        # Fallback to parent implementation
        return super()._from_config(config)
