"""Coder Agent with ParaView/PVPython capabilities and RAG support."""

import asyncio
import json
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, List, Mapping, Optional, Sequence

import numpy as np
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    MessageFactory,
    TextMessage,
)
from autogen_agentchat.state import BaseState
from autogen_core import CancellationToken, Component
from autogen_core.code_executor import CodeBlock, CodeExecutor, CodeResult
from autogen_core.model_context import (
    ChatCompletionContext,
    TokenLimitedChatCompletionContext,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from loguru import logger
from pydantic import Field
from sentence_transformers import SentenceTransformer
from typing_extensions import Self

from ._config import CoderAgentConfig
from .._utils import exec_command_umask_patched
from ...approval_guard import BaseApprovalGuard
from ...guarded_action import ApprovalDeniedError, TrivialGuardedAction
from ...utils import thread_to_context

# Patch Docker executor with umask fix
DockerCommandLineCodeExecutor._execute_command = exec_command_umask_patched  # type: ignore


def _extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    """Extract code blocks from markdown text."""
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


async def _invoke_action_guard(
    thread: Sequence[BaseChatMessage | BaseAgentEvent],
    delta: Sequence[BaseChatMessage | BaseAgentEvent],
    code_message: TextMessage,
    agent_name: str,
    model_client: ChatCompletionClient,
    approval_guard: BaseApprovalGuard | None,
) -> None:
    """Invoke action guard for code execution approval."""
    guarded_action = TrivialGuardedAction("coding", baseline_override="maybe")
    assert delta[-1] == code_message
    thread = list(thread) + list(delta)
    context = thread_to_context(
        thread,
        agent_name,
        is_multimodal=model_client.model_info["vision"],
    )
    action_description_for_user = TextMessage(
        content="Do you want to execute the code above?",
        source=agent_name,
    )
    await guarded_action.invoke_with_approval(
        {}, code_message, context, approval_guard, action_description_for_user
    )


async def _coding_and_debug(
    system_prompt: str,
    thread: Sequence[BaseChatMessage],
    agent_name: str,
    model_client: ChatCompletionClient,
    code_executor: CodeExecutor,
    max_debug_rounds: int,
    cancellation_token: CancellationToken,
    model_context: ChatCompletionContext,
    approval_guard: BaseApprovalGuard | None,
) -> AsyncGenerator[TextMessage | bool, None]:
    """Write and debug code using the model and executor."""
    delta: Sequence[BaseChatMessage | BaseAgentEvent] = []
    executed_code = False

    for i in range(max_debug_rounds):
        # Add system prompt as the last message before generation
        current_thread = (
            list(thread)
            + list(delta)
            + [TextMessage(source="user", content=system_prompt)]
        )

        # Create LLM context from system message, global chat history, and inner messages
        context = [SystemMessage(content=system_prompt)] + thread_to_context(
            current_thread,
            agent_name,
            is_multimodal=model_client.model_info["vision"],
        )

        # Re-initialize model context to meet token limit quota
        try:
            await model_context.clear()
            for msg in context:
                await model_context.add_message(msg)
            token_limited_context = await model_context.get_messages()
        except Exception:
            token_limited_context = context

        # Generate code using the model
        create_result = await model_client.create(
            messages=token_limited_context, cancellation_token=cancellation_token
        )
        assert isinstance(create_result.content, str)
        code_msg = TextMessage(
            source=agent_name + "-llm",
            metadata={"internal": "no", "type": "potential_code"},
            content=create_result.content,
        )
        delta.append(code_msg)
        yield code_msg

        # Extract code blocks from the LLM's response
        code_block_list = _extract_markdown_code_blocks(create_result.content)
        if len(code_block_list) == 0:
            break

        # Get approval for code execution if guard is enabled
        if approval_guard is not None:
            await _invoke_action_guard(
                thread=thread,
                delta=delta,
                code_message=code_msg,
                agent_name=agent_name,
                model_client=model_client,
                approval_guard=approval_guard,
            )

        code_output_list: List[str] = []
        exit_code_list: List[int] = []
        executed_code = True

        try:
            for cb in code_block_list:
                exit_code: int = 1
                encountered_exception: bool = False
                code_output: str = ""
                result: CodeResult | None = None
                try:
                    result = await code_executor.execute_code_blocks(
                        [cb], cancellation_token
                    )
                    exit_code = result.exit_code or 0
                    code_output = result.output
                except Exception as e:
                    code_output = str(e)
                    encountered_exception = True

                if encountered_exception or result is None:
                    code_output = f"An exception occurred while executing the code block: {code_output}"
                elif code_output.strip() == "":
                    code_output = f"The script ran but produced no output to console. The POSIX exit code was: {result.exit_code}. If you were expecting output, consider revising the script to ensure content is printed to stdout."
                elif exit_code != 0:
                    code_output = f"The script ran, then exited with an error (POSIX exit code: {result.exit_code})\nIts output was:\n{result.output}"

                code_output_list.append(code_output)
                code_output_msg = TextMessage(
                    source=agent_name + "-executor",
                    metadata={"internal": "no", "type": "code_execution"},
                    content=f"Execution result of code block {i + 1}:\n```console\n{code_output}\n```",
                )
                exit_code_list.append(exit_code)
                yield code_output_msg

            final_code_output = ""
            for i, code_output in enumerate(code_output_list):
                final_code_output += f"\n\nExecution Result of Code Block {i + 1}:\n```console\n{code_output}\n```"

            executor_msg = TextMessage(
                source=agent_name + "-executor",
                metadata={"internal": "yes"},
                content=final_code_output,
            )
            delta.append(executor_msg)
            yield executor_msg

            # Break if the code execution was successful
            if all([code_output == 0 for code_output in exit_code_list]):
                break
        except asyncio.TimeoutError:
            executor_msg = TextMessage(
                source=agent_name + "-executor",
                metadata={"internal": "yes"},
                content="Code execution timed out.",
            )
            delta.append(executor_msg)
            yield executor_msg

    yield executed_code


async def _summarize_coding(
    agent_name: str,
    model_client: ChatCompletionClient,
    thread: Sequence[BaseChatMessage | BaseAgentEvent],
    cancellation_token: CancellationToken,
    model_context: ChatCompletionContext,
) -> TextMessage:
    """Create a summary from the inner messages using an extra LLM call."""
    input_messages = (
        [SystemMessage(content="You are an agent that can write and debug code")]
        + thread_to_context(
            list(thread), agent_name, is_multimodal=model_client.model_info["vision"]
        )
        + [
            UserMessage(
                content="""The above is a transcript of your previous messages and a request that was given to you in the beginning.
You need to summarize them to answer the request given to you. Generate a summary of everything that happened.
If there was code that was executed, please copy the final code that was executed without errors.
Don't mention that this is a summary, just give the summary.""",
                source="user",
            )
        ]
    )

    # Re-initialize model context to meet token limit quota
    try:
        await model_context.clear()
        for msg in input_messages:
            await model_context.add_message(msg)
        token_limited_input_messages = await model_context.get_messages()
    except Exception:
        token_limited_input_messages = input_messages

    summary_result = await model_client.create(
        messages=token_limited_input_messages, cancellation_token=cancellation_token
    )
    assert isinstance(summary_result.content, str)
    return TextMessage(
        source=agent_name,
        metadata={"internal": "yes"},
        content=summary_result.content,
    )


class CoderAgentState(BaseState):
    """State for the Coder Agent."""
    chat_history: List[BaseChatMessage] = Field(default_factory=list)
    type: str = Field(default="CoderAgentState")


class CoderAgent(Component[CoderAgentConfig]):
    """An agent capable of writing, executing, and debugging code with ParaView/PVPython support.

    This agent combines:
    - General Python code execution (via Docker or local executor)
    - ParaView/PVPython code generation using RAG
    - Iterative debugging and code refinement
    - Access to ParaView operations knowledge base

    The agent uses either a local or Docker-based code executor to run the generated code
    in a controlled environment. It maintains a chat history and can be paused/resumed
    during execution.
    """

    component_type = "agent"
    component_config_schema = CoderAgentConfig
    component_provider_override = "magentic_ui.agents.CoderAgent"

    DEFAULT_DESCRIPTION = """An agent that can write and execute code to solve tasks or use its language skills to summarize, write, solve math and logic problems.
It understands images and can use them to help it complete the task.
It can access files if given the path and manipulate them using python code. Use the coder if you want to manipulate a file or read a csv or excel files.
In a single step when you ask the agent to do something: it can write code, and then immediately execute the code. If there are errors it can debug the code and try again.

Additionally, this agent has ParaView/PVPython capabilities:
- Generate pvpython visualization scripts based on natural language requests
- Use a knowledge base of ParaView operations to create accurate code
- Handle complex visualization tasks including data loading, filters, volume rendering, color mapping, camera positioning, and screenshot generation"""

    system_prompt_coder_template = """You are a helpful assistant.
In addition to responding with text you can write code and execute code that you generate.
The date today is: {date_today}

Rules to follow for Code:
- Generate py or sh code blocks in the order you'd like your code to be executed.
- Code block must indicate language type. Do not try to predict the answer of execution. Code blocks will be automatically executed for you.
- If you want to stop executing code, make sure to not write any code in your message and your turn will be over.
- Do not generate code that relies on API keys that you don't have access to. Try different approaches.

Tips:
- You don't have to generate code if the task is not related to code, for instance writing a poem, paraphrasing a text, etc.
- If you are asked to solve math or logical problems, first try to answer them without code and then if needed try to use python to solve them.
- You have access to the standard Python libraries in addition to numpy, pandas, scikit-learn, matplotlib, pillow, requests, beautifulsoup4.
- If you need to use an external library, write first a shell script that installs the library first using pip install, then add code blocks to use the library.
- Always use print statements to output your work and partial results.
- For showing plots or other visualizations that are not just text, make sure to save them to file with the right extension for them to be displayed.

ParaView/PVPython Capabilities:
- You have access to ParaView Python (pvpython) for 3D scientific visualization
- When working with ParaView tasks, use the search_paraview_operations tool to find relevant operations
- For pvpython code:
  * Always start with: from paraview.simple import *
  * Connect to pvserver using: Connect('paraview-server', 11111)
  * Use proper error handling and validation
  * Include comments explaining key steps
  * After creating visualizations, remember to Update() and Render()
  * Save screenshots using SaveScreenshot() when requested

VERY IMPORTANT: If you intend to write code to be executed, do not end your response without a code block. If you want to write code you must provide a code block in the current generation."""

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        *,
        model_context_token_limit: int = 128000,
        description: str = DEFAULT_DESCRIPTION,
        max_debug_rounds: int = 3,
        summarize_output: bool = False,
        code_executor: Optional[CodeExecutor] = None,
        work_dir: Path | str | None = None,
        bind_dir: Path | str | None = None,
        use_local_executor: bool = False,
        approval_guard: BaseApprovalGuard | None = None,
        network_name: str = "my-network",
        # PVPython RAG settings
        enable_pvpython_rag: bool = True,
        operations_json_path: Path | None = None,
        top_k_operations: int = 5,
        pvserver_host: str = "paraview-server",
        pvserver_port: int = 11111,
        debug_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the CoderAgent with optional PVPython RAG capabilities.

        Args:
            name: The name of the agent
            model_client: The language model client to use
            model_context_token_limit: Token limit for model context
            description: Description of the agent's capabilities
            max_debug_rounds: Maximum number of code debugging iterations
            summarize_output: Whether to summarize code execution results
            code_executor: Custom code executor to use
            work_dir: Working directory for code execution
            bind_dir: Directory to bind for Docker executor
            use_local_executor: Whether to use local instead of Docker executor
            approval_guard: Guard for code execution approval
            network_name: Docker network name for container communication
            enable_pvpython_rag: Whether to enable PVPython RAG capabilities
            operations_json_path: Path to operations.json database for RAG
            top_k_operations: Number of similar operations to retrieve
            pvserver_host: PVServer hostname (Docker container name)
            pvserver_port: PVServer port
            debug_dir: Directory to save generated code for debugging
            **kwargs: Additional arguments
        """
        super().__init__()
        self.name = name
        self.description = description
        self._model_client = model_client
        self._model_context = TokenLimitedChatCompletionContext(
            model_client, token_limit=model_context_token_limit
        )
        self._chat_history: List[BaseChatMessage] = []
        self._max_debug_rounds = max_debug_rounds
        self._summarize_output = summarize_output
        self.is_paused = False
        self._paused = asyncio.Event()
        self._approval_guard = approval_guard
        self._did_lazy_init = False
        self._network_name = network_name

        # PVPython RAG settings
        self._enable_pvpython_rag = enable_pvpython_rag
        self._top_k_operations = top_k_operations
        self._pvserver_host = pvserver_host
        self._pvserver_port = pvserver_port
        self._debug_dir = debug_dir
        self._operations: List[dict] = []
        self._operation_embeddings: Optional[np.ndarray] = None
        self._model: Optional[SentenceTransformer] = None

        # Initialize RAG components if enabled
        if self._enable_pvpython_rag:
            if operations_json_path is None:
                operations_json_path = Path(__file__).parent / "operations.json"
            self._operations_json_path = operations_json_path

            if self._operations_json_path.exists():
                self._operations = self._load_operations()
                self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self._operation_embeddings = self._compute_operation_embeddings()
                logger.info(f"[CoderAgent] Loaded {len(self._operations)} ParaView operations for RAG")
            else:
                logger.warning(f"[CoderAgent] Operations JSON not found at {self._operations_json_path}, PVPython RAG disabled")
                self._enable_pvpython_rag = False

        # Create debug directory if specified
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        # Set up work directory
        if work_dir is None:
            self._work_dir = Path(tempfile.mkdtemp())
            self._cleanup_work_dir = True
        else:
            self._work_dir = Path(work_dir)
            self._cleanup_work_dir = False

        # Set up code executor
        if code_executor:
            self._code_executor = code_executor
        elif use_local_executor:
            self._code_executor = LocalCommandLineCodeExecutor(work_dir=self._work_dir)
        else:
            from ..._docker import PYTHON_IMAGE, patch_docker_executor_with_network

            container_name = f"{name}-{uuid.uuid4()}"
            self._code_executor = DockerCommandLineCodeExecutor(
                container_name=container_name,
                image=PYTHON_IMAGE,
                work_dir=self._work_dir,
                bind_dir=bind_dir,
                delete_tmp_files=True,
            )
            # Patch the executor to use the specified Docker network
            patch_docker_executor_with_network(self._code_executor, self._network_name)

    def _load_operations(self) -> List[dict]:
        """Load operations from JSON file."""
        with open(self._operations_json_path, 'r') as f:
            operations = json.load(f)
        return operations

    def _compute_operation_embeddings(self) -> np.ndarray:
        """Precompute embeddings for all operations."""
        texts = []
        for op in self._operations:
            # Combine name, description, and code for better matching
            text = f"{op['name']} {op['description']} {op['code_snippet']}"
            texts.append(text)

        # Generate embeddings
        embeddings = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        return embeddings

    def _search_similar_operations(self, query: str, top_k: int) -> List[dict]:
        """Search for similar operations using semantic similarity."""
        if not self._enable_pvpython_rag or self._model is None or self._operation_embeddings is None:
            return []

        # Generate query embedding
        query_embedding = self._model.encode(query, convert_to_numpy=True).astype(np.float32)

        # Compute cosine similarity
        similarities = np.dot(self._operation_embeddings, query_embedding) / (
            np.linalg.norm(self._operation_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_k = min(top_k, len(self._operations))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return matching operations
        results = []
        for idx in top_indices:
            op = self._operations[idx].copy()
            op['similarity_score'] = float(similarities[idx])
            results.append(op)

        return results

    def search_paraview_operations(self, query: str, top_k: int | None = None) -> str:
        """Search for relevant ParaView operations from the knowledge base.

        This is exposed as a method that can be called during code generation
        to provide relevant ParaView operations context.

        Args:
            query: Description of what visualization operations you need
            top_k: Number of similar operations to return (default: 5)

        Returns:
            JSON string containing relevant operations with their code snippets
        """
        if not self._enable_pvpython_rag:
            return json.dumps({"error": "PVPython RAG is not enabled"})

        if top_k is None:
            top_k = self._top_k_operations

        results = self._search_similar_operations(query, top_k)
        return json.dumps(results, indent=2)

    async def lazy_init(self) -> None:
        """Initialize the code executor if it has a start method."""
        if self._did_lazy_init:
            return
        if self._code_executor:
            if hasattr(self._code_executor, "start"):
                await self._code_executor.start()  # type: ignore
        self._did_lazy_init = True

    async def close(self) -> None:
        """Clean up resources used by the agent."""
        logger.info("Closing Coder...")
        await self._code_executor.stop()
        # Remove the work directory if it was created
        if self._cleanup_work_dir and self._work_dir.exists():
            await asyncio.to_thread(shutil.rmtree, self._work_dir)
        # Close the model client
        await self._model_client.close()

    async def pause(self) -> None:
        """Pause the agent by setting the paused state."""
        self.is_paused = True
        self._paused.set()

    async def resume(self) -> None:
        """Resume the agent by clearing the paused state."""
        self.is_paused = False
        self._paused.clear()

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """Get the types of messages produced by the agent."""
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        """Handle incoming messages and return a single response."""
        response: Response | None = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                response = message
        assert response is not None
        return response

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """Handle incoming messages and yield responses as a stream."""
        await self.lazy_init()

        if self.is_paused:
            yield Response(
                chat_message=TextMessage(
                    content="The Coder is paused.",
                    source=self.name,
                    metadata={"internal": "yes"},
                )
            )
            return

        self._chat_history.extend(messages)
        last_message_received: BaseChatMessage = messages[-1]
        inner_messages: List[BaseChatMessage] = []

        # Set up the cancellation token for the code execution
        code_execution_token = CancellationToken()
        cancellation_token.add_callback(lambda: code_execution_token.cancel())

        # Monitor pause event and cancel code execution if paused
        async def monitor_pause() -> None:
            await self._paused.wait()
            code_execution_token.cancel()

        monitor_pause_task = asyncio.create_task(monitor_pause())

        # Build system prompt
        system_prompt_coder = self.system_prompt_coder_template.format(
            date_today=datetime.now().strftime("%Y-%m-%d")
        )

        # Add ParaView operations context if RAG is enabled and query seems ParaView-related
        if self._enable_pvpython_rag:
            message_content = last_message_received.content if hasattr(last_message_received, 'content') else ""
            if any(keyword in str(message_content).lower() for keyword in [
                'paraview', 'pvpython', 'visualization', 'render', 'volume', 'contour',
                'slice', 'clip', 'filter', 'vtk', 'isosurface', 'streamline'
            ]):
                # Search for relevant operations
                relevant_ops = self._search_similar_operations(str(message_content), self._top_k_operations)
                if relevant_ops:
                    ops_context = "\n\nRelevant ParaView Operations:\n"
                    for i, op in enumerate(relevant_ops, 1):
                        ops_context += f"\n{i}. {op['name']}\n"
                        ops_context += f"   Description: {op['description']}\n"
                        ops_context += f"   Code snippet:\n```python\n{op['code_snippet']}\n```\n"
                    system_prompt_coder += ops_context

        try:
            executed_code = False
            # Run the code execution and debugging process
            async for msg in _coding_and_debug(
                system_prompt=system_prompt_coder,
                thread=self._chat_history,
                agent_name=self.name,
                model_client=self._model_client,
                code_executor=self._code_executor,
                max_debug_rounds=self._max_debug_rounds,
                cancellation_token=code_execution_token,
                model_context=self._model_context,
                approval_guard=self._approval_guard,
            ):
                if isinstance(msg, bool):
                    executed_code = msg
                    break
                inner_messages.append(msg)
                self._chat_history.append(msg)
                yield msg

            # Summarize if configured
            if self._summarize_output and executed_code:
                summary_msg = await _summarize_coding(
                    agent_name=self.name,
                    model_client=self._model_client,
                    thread=[last_message_received] + inner_messages,
                    cancellation_token=code_execution_token,
                    model_context=self._model_context,
                )
                self._chat_history.append(summary_msg)
                yield Response(chat_message=summary_msg, inner_messages=inner_messages)
            else:
                # Return transcript of all code and execution steps
                combined_output = ""
                for txt_msg in inner_messages:
                    assert isinstance(txt_msg, TextMessage)
                    combined_output += f"{txt_msg.content}\n"
                final_response_msg = TextMessage(
                    source=self.name,
                    metadata={"internal": "yes"},
                    content=combined_output or "No output.",
                )
                yield Response(
                    chat_message=final_response_msg, inner_messages=inner_messages
                )
        except ApprovalDeniedError:
            yield Response(
                chat_message=TextMessage(
                    content="The user did not approve the code execution.",
                    source=self.name,
                    metadata={"internal": "no"},
                ),
                inner_messages=inner_messages,
            )
        except asyncio.CancelledError:
            yield Response(
                chat_message=TextMessage(
                    content="The task was cancelled by the user.",
                    source=self.name,
                    metadata={"internal": "yes"},
                ),
                inner_messages=inner_messages,
            )
        except Exception as e:
            logger.error(f"Error in CoderAgent: {e}")
            self._chat_history.append(
                TextMessage(
                    content=f"An error occurred while executing the code: {e}",
                    source=self.name,
                )
            )
            yield Response(
                chat_message=TextMessage(
                    content=f"An error occurred in the coder agent: {e}",
                    source=self.name,
                    metadata={"internal": "no"},
                ),
                inner_messages=inner_messages,
            )
        finally:
            # Cancel the monitor task
            try:
                monitor_pause_task.cancel()
                await monitor_pause_task
            except asyncio.CancelledError:
                pass

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Clear the chat history."""
        self._chat_history.clear()

    def _to_config(self) -> CoderAgentConfig:
        """Convert the agent's state to a configuration object."""
        return CoderAgentConfig(
            name=self.name,
            model_client=self._model_client.dump_component(),
            description=self.description,
            max_debug_rounds=self._max_debug_rounds,
            summarize_output=self._summarize_output,
            model_context_token_limit=self._model_context._token_limit,
            enable_pvpython_rag=self._enable_pvpython_rag,
            operations_json_path=self._operations_json_path if self._enable_pvpython_rag else None,
            top_k_operations=self._top_k_operations,
            pvserver_host=self._pvserver_host,
            pvserver_port=self._pvserver_port,
            network_name=self._network_name,
            debug_dir=self._debug_dir,
        )

    @classmethod
    def _from_config(cls, config: CoderAgentConfig) -> Self:
        """Create an agent instance from a configuration object."""
        return cls(
            name=config.name,
            model_client=ChatCompletionClient.load_component(config.model_client),
            description=config.description,
            max_debug_rounds=config.max_debug_rounds,
            summarize_output=config.summarize_output,
            model_context_token_limit=config.model_context_token_limit,
            enable_pvpython_rag=config.enable_pvpython_rag,
            operations_json_path=config.operations_json_path,
            top_k_operations=config.top_k_operations,
            pvserver_host=config.pvserver_host,
            pvserver_port=config.pvserver_port,
            network_name=config.network_name,
            debug_dir=config.debug_dir,
        )

    async def save_state(self) -> Mapping[str, Any]:
        """Save the state of the agent."""
        return {
            "chat_history": [msg.dump() for msg in self._chat_history],
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load the state of the agent."""
        message_factory = MessageFactory()
        self._chat_history = []
        for msg_data in state["chat_history"]:
            msg = message_factory.create(msg_data)
            assert isinstance(msg, BaseChatMessage)
            self._chat_history.append(msg)
