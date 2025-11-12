"""Docker manager for Python environment container."""

from __future__ import annotations

import asyncio
import logging
import socket
from pathlib import Path
from typing import Optional

import docker
from autogen_core import Component
from docker.models.containers import Container
from pydantic import BaseModel

logger = logging.getLogger(__name__)

PYTHON_ENV_IMAGE_DEFAULT = "ghcr.io/microsoft/magentic-ui-python-env:latest"


class PythonEnvDockerManagerConfig(BaseModel):
    """Configuration for Python Environment Docker Manager."""

    image: str = PYTHON_ENV_IMAGE_DEFAULT
    workspace_dir: Path | None = None
    inside_docker: bool = True
    network_name: str = "my-network"
    pvserver_host: str = "paraview-server"
    pvserver_port: int = 11111


class PythonEnvDockerManager(Component[PythonEnvDockerManagerConfig]):
    """
    Manages a Python Environment Docker container for executing pvpython code.

    This manager handles:
    - Starting/stopping Python environment containers
    - Mounting workspace directory for code execution
    - Network configuration to connect to ParaView server
    - Container lifecycle management

    Args:
        image (str): Docker image name. Default: "ghcr.io/microsoft/magentic-ui-python-env:latest"
        workspace_dir (Path | None): Directory to mount as /workspace
        inside_docker (bool): Whether the client is running inside Docker
        network_name (str): Docker network name for container communication
        pvserver_host (str): Hostname of the ParaView server container
        pvserver_port (int): Port for ParaView server connection

    Properties:
        is_running (bool): Whether the container is currently running
        container_name (str): Name of the Docker container

    Example:
        ```python
        manager = PythonEnvDockerManager(
            workspace_dir=Path("./workspace"),
            network_name="my-network",
            pvserver_host="paraview-server"
        )
        await manager.start()
        # Execute Python code in the container
        result = await manager.execute_code("print('Hello')")
        await manager.stop()
        ```
    """

    component_config_schema = PythonEnvDockerManagerConfig
    component_type = "other"

    def __init__(
        self,
        *,
        image: str = PYTHON_ENV_IMAGE_DEFAULT,
        workspace_dir: Path | None = None,
        inside_docker: bool = True,
        network_name: str = "my-network",
        pvserver_host: str = "paraview-server",
        pvserver_port: int = 11111,
    ):
        super().__init__()
        self._image = image
        self._workspace_dir = workspace_dir
        self._inside_docker = inside_docker
        self._network_name = network_name
        self._pvserver_host = pvserver_host
        self._pvserver_port = pvserver_port

        self._container: Optional[Container] = None
        self._docker_name = "python-client"  # Match the run.sh script name

    @property
    def container_name(self) -> str:
        """Get the container name."""
        return self._docker_name

    @property
    def pvserver_host(self) -> str:
        """Get the ParaView server hostname for connection."""
        return self._pvserver_host

    @property
    def pvserver_port(self) -> int:
        """Get the ParaView server port."""
        return self._pvserver_port

    @property
    def is_running(self) -> bool:
        """Check if the container is running."""
        if self._container is None:
            return False
        try:
            self._container.reload()
            return self._container.status == "running"
        except Exception:
            return False

    async def _create_container(self) -> Container:
        """Create and configure the Python Environment Docker container."""
        logger.info(f"Creating Python environment container: {self._docker_name}")

        client = docker.from_env()

        # Prepare volume mounts
        volumes = {}
        if self._workspace_dir:
            workspace_path = self._workspace_dir.resolve()

            # Create workspace directory if it doesn't exist
            if not workspace_path.exists():
                try:
                    workspace_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created workspace directory at: {workspace_path}")
                except Exception as e:
                    logger.error(f"Failed to create workspace directory {workspace_path}: {e}")
                    logger.warning("Continuing without workspace volume mount")
                    workspace_path = None
            else:
                logger.info(f"Using existing workspace directory at: {workspace_path}")

            if workspace_path:
                volumes[str(workspace_path)] = {
                    "bind": "/workspace",
                    "mode": "rw"
                }

        # Prepare environment variables
        environment = {
            "PVSERVER_HOST": self._pvserver_host,
            "PVSERVER_PORT": str(self._pvserver_port),
        }

        # Determine platform based on host architecture
        import platform
        host_arch = platform.machine().lower()
        container_platform = "linux/arm64" if "arm" in host_arch or "aarch64" in host_arch else "linux/amd64"

        return await asyncio.to_thread(
            client.containers.create,
            name=self._docker_name,
            image=self._image,
            detach=True,
            auto_remove=False,  # Keep container running for code execution
            network=self._network_name,  # Always use network for container-to-container communication
            platform=container_platform,
            volumes=volumes if volumes else None,
            environment=environment,
            command="tail -f /dev/null",  # Keep container alive
            working_dir="/workspace",
        )

    async def start(self) -> None:
        """Start the Python Environment Docker container."""
        if self.is_running:
            logger.info("Python environment container is already running")
            return

        try:
            # Create docker network if it doesn't exist (for container communication)
            # Always create network for container-to-container communication
            client = docker.from_env()
            try:
                await asyncio.to_thread(client.networks.get, self._network_name)
                logger.info(f"Docker network '{self._network_name}' already exists")
            except docker.errors.NotFound:
                await asyncio.to_thread(client.networks.create, self._network_name, driver="bridge")
                logger.info(f"Created Docker network: {self._network_name}")

            # Check if a container with this name already exists
            try:
                existing_container = await asyncio.to_thread(
                    client.containers.get, self._docker_name
                )
                logger.info(f"Found existing container: {self._docker_name}")

                # Check if it's running
                await asyncio.to_thread(existing_container.reload)
                if existing_container.status == "running":
                    logger.info("Existing container is running, reusing it")
                    self._container = existing_container
                    return
                else:
                    logger.info("Existing container is not running, removing it")
                    await asyncio.to_thread(existing_container.remove, force=True)
            except docker.errors.NotFound:
                # No existing container, proceed with creation
                logger.info(f"No existing container found with name: {self._docker_name}")
                pass

            # Create and start the container
            self._container = await self._create_container()
            await asyncio.to_thread(self._container.start)

            logger.info(f"Python environment container started: {self._docker_name}")
            logger.info(f"Can connect to ParaView at: {self._pvserver_host}:{self._pvserver_port}")

        except Exception as e:
            logger.error(f"Failed to start Python environment container: {e}")
            self._container = None
            raise

    async def stop(self) -> None:
        """Stop and remove the Python Environment Docker container."""
        if self._container is None:
            logger.info("No Python environment container to stop")
            return

        try:
            logger.info(f"Stopping Python environment container: {self._docker_name}")
            await asyncio.to_thread(self._container.stop, timeout=5)
            await asyncio.to_thread(self._container.remove)
            logger.info("Python environment container stopped and removed")
        except Exception as e:
            logger.error(f"Error stopping Python environment container: {e}")
        finally:
            self._container = None

    async def execute_code(self, code: str, timeout: int = 60) -> tuple[int, str]:
        """
        Execute Python code in the container.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds

        Returns:
            Tuple of (exit_code, output)
        """
        if not self.is_running:
            raise RuntimeError("Container is not running. Call start() first.")

        try:
            # Write code to a temporary file and execute it
            exec_command = f'python -c "{code.replace(chr(34), chr(92)+chr(34))}"'

            exit_code, output = await asyncio.to_thread(
                self._container.exec_run,
                cmd=["bash", "-c", exec_command],
                demux=True,
                workdir="/workspace"
            )

            # Combine stdout and stderr
            stdout_output = output[0].decode() if output[0] else ""
            stderr_output = output[1].decode() if output[1] else ""
            combined_output = stdout_output + stderr_output

            return exit_code, combined_output

        except Exception as e:
            logger.error(f"Error executing code in container: {e}")
            raise

    async def execute_pvpython(self, script_path: str, timeout: int = 60) -> tuple[int, str]:
        """
        Execute a pvpython script in the container.

        Args:
            script_path: Path to the script file (relative to /workspace)
            timeout: Timeout in seconds

        Returns:
            Tuple of (exit_code, output)
        """
        if not self.is_running:
            raise RuntimeError("Container is not running. Call start() first.")

        try:
            exit_code, output = await asyncio.to_thread(
                self._container.exec_run,
                cmd=["pvpython", script_path],
                demux=True,
                workdir="/workspace"
            )

            # Combine stdout and stderr
            stdout_output = output[0].decode() if output[0] else ""
            stderr_output = output[1].decode() if output[1] else ""
            combined_output = stdout_output + stderr_output

            return exit_code, combined_output

        except Exception as e:
            logger.error(f"Error executing pvpython script in container: {e}")
            raise
