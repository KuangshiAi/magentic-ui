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
        model_context_token_limit: int | None = None,
        session_id: int | None = None,  # NEW: Session ID for ParaView service lookup
        **kwargs: Any,
    ):
        self._paraview_docker_manager = paraview_docker_manager  # Kept for backward compatibility but not used
        self._did_lazy_init = False
        self._paraview_just_initialized = False
        self.novnc_port: int | None = None
        self.pvserver_port: int | None = None
        self._session_id = session_id  # NEW: Store session ID

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
        Connect MCP to existing ParaView server.

        NEW BEHAVIOR (Session-level ParaView service):
        - The ParaView server is already running (started when session was created)
        - This method only retrieves connection info from the ParaView service
        - MCP connection happens automatically through the parent McpAgent class

        OLD BEHAVIOR (kept for backward compatibility):
        - If docker_manager is provided, use old behavior
        - Start ParaView Docker container
        - Wait for pvserver to be ready
        - Capture port information
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info("ParaView lazy_init() called")

        if self._did_lazy_init:
            logger.info("ParaView already initialized, skipping")
            return

        # NEW: Check if we have a session ID (new flow)
        if self._session_id is not None:
            logger.info(f"Using session-level ParaView service for session {self._session_id}")

            # Import here to avoid circular dependency
            from ...backend.paraview_service_manager import ParaViewServiceManager

            # Get the ParaView service for this session
            paraview_service = await ParaViewServiceManager.get(self._session_id)

            if paraview_service and paraview_service.is_running():
                # Capture port information from the running service
                self.novnc_port = paraview_service.novnc_port
                self.pvserver_port = paraview_service.pvserver_port

                logger.info(
                    f"Connected to session ParaView service. "
                    f"noVNC port: {self.novnc_port}, pvserver port: {self.pvserver_port}"
                )

                # Mark that we just initialized so we can send the browser address
                self._paraview_just_initialized = True
            else:
                logger.warning(
                    f"ParaView service for session {self._session_id} is not running. "
                    "ParaView tools will not be available."
                )

        # OLD: Backward compatibility - use Docker manager if provided
        elif self._paraview_docker_manager is not None:
            # Start the ParaView Docker container (includes GUI auto-connect wait)
            logger.info("Starting ParaView Docker container (legacy mode)...")
            await self._paraview_docker_manager.start()

            # Capture port information
            self.novnc_port = self._paraview_docker_manager.novnc_port
            self.pvserver_port = self._paraview_docker_manager.pvserver_port

            logger.info(
                f"ParaView Docker started. "
                f"noVNC port: {self.novnc_port}, pvserver port: {self.pvserver_port}"
            )

            # Mark that we just initialized so we can send the browser address
            self._paraview_just_initialized = True
        else:
            logger.warning("No ParaView service or Docker manager available")

        self._did_lazy_init = True

    def _prune_old_screenshots(self, messages: Sequence[BaseChatMessage], keep_last_n: int = 3) -> List[BaseChatMessage]:
        """
        Prune old screenshot images from message history to manage context length.

        Keep only the last N screenshots and replace older ones with text descriptions.

        Args:
            messages: The message history
            keep_last_n: Number of most recent screenshots to keep (default: 3)

        Returns:
            Pruned list of messages
        """
        from autogen_agentchat.messages import ToolCallExecutionEvent
        import copy

        # Find all screenshot tool calls
        screenshot_indices = []
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolCallExecutionEvent):
                # Check if this is a screenshot tool call
                if hasattr(msg, 'content') and isinstance(msg.content, list):
                    for item in msg.content:
                        if hasattr(item, 'type') and item.type == 'image':
                            screenshot_indices.append(i)
                            break

        # If we have more screenshots than we want to keep, prune the older ones
        if len(screenshot_indices) > keep_last_n:
            indices_to_prune = screenshot_indices[:-keep_last_n]

            # Create a new list with pruned messages
            pruned_messages = []
            for i, msg in enumerate(messages):
                if i in indices_to_prune:
                    # Replace image with text description
                    new_msg = copy.deepcopy(msg)
                    if hasattr(new_msg, 'content') and isinstance(new_msg.content, list):
                        new_content = []
                        for item in new_msg.content:
                            if hasattr(item, 'type') and item.type == 'image':
                                # Replace with text
                                from autogen_agentchat.messages import FunctionExecutionResultMessage
                                new_content.append(FunctionExecutionResultMessage(
                                    content="[Screenshot removed to save context - view available in ParaView GUI]",
                                    call_id=getattr(item, 'call_id', '')
                                ))
                            else:
                                new_content.append(item)
                        new_msg.content = new_content
                    pruned_messages.append(new_msg)
                else:
                    pruned_messages.append(msg)

            return pruned_messages

        return list(messages)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Process messages and ensure ParaView container is initialized.

        This method:
        - Performs lazy initialization of ParaView container on first call
        - Sends browser address to UI after initialization
        - Prunes old screenshots to manage context length
        - Forwards messages to the parent MCP agent implementation
        - Marks events for UI display
        """
        # Lazy initialization on first message
        await self.lazy_init()

        # Send browser address message after ParaView is initialized
        if (
            self._paraview_just_initialized
            and self._paraview_docker_manager is not None
            and self.novnc_port is not None
            and self.novnc_port > 0
        ):
            # Send ParaView noVNC address message after container is initialized
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

        # Prune old screenshots from messages to manage context length
        pruned_messages = self._prune_old_screenshots(messages, keep_last_n=3)

        # Track the last tool call to capture screenshots automatically
        last_tool_call_name = None

        # Process messages through parent MCP agent
        async for event in super().on_messages_stream(pruned_messages, cancellation_token):
            # Track tool call requests
            if isinstance(event, ToolCallRequestEvent):
                # Store the tool call name(s) for later screenshot capture
                if hasattr(event, 'content') and event.content:
                    for call in event.content:
                        if hasattr(call, 'name'):
                            last_tool_call_name = call.name

            # Fix metadata for manually called get_screenshot AFTER parent processing
            # This must be done AFTER super().on_messages_stream() since parent overwrites metadata
            if isinstance(event, ToolCallExecutionEvent):
                # Check if this is a get_screenshot call
                if last_tool_call_name and last_tool_call_name.endswith('get_screenshot'):
                    # Change metadata from "progress_message" to "paraview_screenshot"
                    metadata = getattr(event, "metadata", {})
                    metadata = {
                        **metadata,
                        "type": "paraview_screenshot"
                    }
                    setattr(event, "metadata", metadata)

            # Yield the event (with corrected metadata if needed)
            yield event

            # Automatically capture screenshot after each tool execution (except get_screenshot itself)
            if isinstance(event, ToolCallExecutionEvent):
                # Only capture screenshot if the last tool wasn't get_screenshot itself
                if last_tool_call_name and not last_tool_call_name.endswith('get_screenshot'):
                    try:

                        # Call get_screenshot through the MCP tools
                        from autogen_agentchat.messages import FunctionCall

                        # Create a tool call request for get_screenshot
                        screenshot_call = FunctionCall(
                            id=f"auto_screenshot_{last_tool_call_name}",
                            name="ParaView-get_screenshot",
                            arguments="{}"
                        )

                        # Yield a tool call request
                        yield ToolCallRequestEvent(
                            source=self.name,
                            content=[screenshot_call]
                        )

                        # Execute the screenshot tool through the workbench
                        if hasattr(self, '_workbench') and self._workbench:
                            # Call the tool directly through the workbench
                            result = await self._workbench.call_tool(
                                "ParaView-get_screenshot",
                                arguments={},
                                cancellation_token=cancellation_token
                            )

                            # Extract Image from ToolResult
                            from autogen_core import Image
                            actual_result = None
                            if hasattr(result, 'result') and isinstance(result.result, list) and len(result.result) > 0:
                                first_result = result.result[0]
                                if hasattr(first_result, 'content'):
                                    actual_result = first_result.content

                            if actual_result is None:
                                actual_result = result

                            # Convert Image to base64 format expected by backend
                            if isinstance(actual_result, Image):
                                base64_data = actual_result.to_base64()
                                content_str = f"[Image: {base64_data}]"
                            else:
                                content_str = str(actual_result)

                            # Yield the screenshot result
                            from autogen_agentchat.messages import FunctionExecutionResult
                            screenshot_event = ToolCallExecutionEvent(
                                source=self.name,
                                content=[FunctionExecutionResult(
                                    call_id=screenshot_call.id,
                                    name="ParaView-get_screenshot",
                                    content=content_str
                                )],
                                metadata={"type": "paraview_screenshot"}
                            )
                            yield screenshot_event
                    except Exception as e:
                        print(f"✗✗✗ Failed to auto-capture screenshot: {e}")
                        import traceback
                        traceback.print_exc()

                # Reset the tool call name after processing
                last_tool_call_name = None

    async def close(self) -> None:
        """
        Close the agent and stop the ParaView Docker container.

        This method:
        - Stops and removes the ParaView Docker container
        - Cleans up resources
        """
        if self._paraview_docker_manager is not None:
            await self._paraview_docker_manager.stop()

    @classmethod
    def _from_config(cls, config: Any, session_id: int | None = None):
        """
        Create a ParaViewAgent instance from configuration.

        Args:
            config: ParaViewAgentConfig or compatible configuration object
            session_id: Optional session ID for session-level ParaView service lookup

        Returns:
            ParaViewAgent: Configured agent instance
        """
        if isinstance(config, ParaViewAgentConfig):
            # Create the ParaView Docker manager (kept for backward compatibility)
            # NEW: When session_id is provided, this won't be used - the session-level
            # ParaView service will be used instead
            from pathlib import Path
            from autogen_core.models import ChatCompletionClient

            paraview_docker_manager = None
            if session_id is None:
                # Legacy mode: create Docker manager
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
                model_context_token_limit=config.model_context_token_limit,
                description=config.description,
                system_message=config.system_message,
                session_id=session_id,  # NEW: Pass session_id
            )

        # Fallback to parent implementation
        return super()._from_config(config)
