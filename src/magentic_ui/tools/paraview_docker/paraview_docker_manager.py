"""Docker manager for ParaView container."""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from pathlib import Path
from typing import Optional

import docker
from autogen_core import Component
from docker.models.containers import Container
from pydantic import BaseModel

logger = logging.getLogger(__name__)

PARAVIEW_IMAGE_DEFAULT = "paraview-novnc:latest"


class ParaViewDockerManagerConfig(BaseModel):
    """Configuration for ParaView Docker Manager."""

    image: str = PARAVIEW_IMAGE_DEFAULT
    novnc_port: int = 6080
    vnc_port: int = 5900
    pvserver_port: int = 11111
    data_dir: Path | None = None
    inside_docker: bool = True
    network_name: str = "my-network"
    auto_gui_connect: bool = True
    width: int = 1920  # VNC display width (increased for better fit in Live View)
    height: int = 1080  # VNC display height (increased for better fit in Live View)
    gui_connect_wait_time: int = 10  # Seconds to wait for GUI to connect before MCP (adjustable per machine)


class ParaViewDockerManager(Component[ParaViewDockerManagerConfig]):
    """
    Manages a ParaView Docker container with GUI access via noVNC.

    This manager handles:
    - Starting/stopping ParaView containers
    - Port management for noVNC, VNC, and pvserver
    - Volume mounting for data access
    - Container lifecycle management

    Args:
        image (str): Docker image name for ParaView. Default: "paraview-novnc:latest"
        novnc_port (int): Port for noVNC web interface. Default: 6080
        vnc_port (int): Port for VNC server. Default: 5900
        pvserver_port (int): Port for ParaView server. Default: 11111
        data_dir (Path | None): Directory to mount for data access
        inside_docker (bool): Whether the client is running inside Docker
        network_name (str): Docker network name for container communication
        auto_gui_connect (bool): Whether to automatically start ParaView GUI

    Properties:
        novnc_address (str): HTTP URL for noVNC web interface
        pvserver_address (str): Address for ParaView server connection
        is_running (bool): Whether the container is currently running

    Example:
        ```python
        manager = ParaViewDockerManager(
            data_dir=Path("./data"),
            novnc_port=6080,
            pvserver_port=11111
        )
        await manager.start()
        # Access ParaView GUI at manager.novnc_address
        # Connect MCP server to manager.pvserver_address
        await manager.stop()
        ```
    """

    component_config_schema = ParaViewDockerManagerConfig
    component_type = "other"

    def __init__(
        self,
        *,
        image: str = PARAVIEW_IMAGE_DEFAULT,
        novnc_port: int = 6080,
        vnc_port: int = 5900,
        pvserver_port: int = 11111,
        data_dir: Path | None = None,
        inside_docker: bool = True,
        network_name: str = "my-network",
        auto_gui_connect: bool = True,
        width: int = 1920,
        height: int = 1080,
        gui_connect_wait_time: int = 10,
    ):
        super().__init__()
        self._image = image
        self._novnc_port = novnc_port
        self._vnc_port = vnc_port
        self._pvserver_port = pvserver_port
        self._data_dir = data_dir
        self._inside_docker = inside_docker
        self._network_name = network_name
        self._auto_gui_connect = auto_gui_connect
        self._width = width
        self._height = height
        self._gui_connect_wait_time = gui_connect_wait_time

        self._container: Optional[Container] = None
        # Use static hostname for Docker network resolution
        self._docker_name = "paraview-server"
        self._hostname = self._docker_name  # Same for consistency

    def _get_available_port(self) -> tuple[int, socket.socket]:
        """Get an available port on the local machine."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        return port, s

    def _generate_new_ports(self) -> None:
        """Generate new available ports for all services."""
        self._novnc_port, novnc_sock = self._get_available_port()
        self._vnc_port, vnc_sock = self._get_available_port()
        self._pvserver_port, pvserver_sock = self._get_available_port()

        # Keep static container name and hostname
        # self._docker_name and self._hostname are already set in __init__

        novnc_sock.close()
        vnc_sock.close()
        pvserver_sock.close()

    @property
    def novnc_address(self) -> str:
        """Get the HTTP address for noVNC web interface."""
        return f"http://{self._hostname}:{self._novnc_port}/vnc.html"

    @property
    def pvserver_address(self) -> str:
        """Get the address for ParaView server connection."""
        return f"{self._hostname}:{self._pvserver_port}"

    @property
    def novnc_port(self) -> int:
        """Get the noVNC port."""
        return self._novnc_port

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
        """Create and configure the ParaView Docker container."""
        logger.info(
            f"Creating ParaView container with noVNC on port {self._novnc_port} "
            f"and pvserver on port {self._pvserver_port}..."
        )

        client = docker.from_env()

        # Prepare volume mounts
        volumes = {}
        if self._data_dir:
            # Ensure the data directory exists on the host before Docker tries to mount it
            data_dir_path = self._data_dir.resolve()

            # Check if directory exists, if not create it
            if not data_dir_path.exists():
                try:
                    data_dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created data directory at: {data_dir_path}")
                except Exception as e:
                    logger.error(f"Failed to create data directory {data_dir_path}: {e}")
                    logger.warning("Continuing without data volume mount")
                    data_dir_path = None
            else:
                logger.info(f"Using existing data directory at: {data_dir_path}")

            if data_dir_path:
                volumes[str(data_dir_path)] = {
                    "bind": "/home/MCPagent/data",
                    "mode": "rw"
                }

        # Prepare environment variables
        environment = {
            "AUTO_GUI_CONNECT": "1" if self._auto_gui_connect else "0",
            "PV_PORT": str(self._pvserver_port),
            "WIDTH": str(self._width),
            "HEIGHT": str(self._height),
        }

        return await asyncio.to_thread(
            client.containers.create,
            name=self._docker_name,
            image=self._image,
            detach=True,
            auto_remove=True,
            network=self._network_name,  # Always use network for container-to-container communication
            platform="linux/amd64",  # ParaView container is built for amd64
            ports={
                f"{self._novnc_port}/tcp": self._novnc_port,
                f"{self._vnc_port}/tcp": self._vnc_port,
                f"{self._pvserver_port}/tcp": self._pvserver_port,
            },
            volumes=volumes if volumes else None,
            environment=environment,
        )

    async def start(self) -> None:
        """Start the ParaView Docker container."""
        if self.is_running:
            logger.info("ParaView container is already running")
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

            logger.info(f"ParaView container started: {self._docker_name}")
            logger.info(f"noVNC available at: {self.novnc_address}")
            logger.info(f"pvserver available at: {self.pvserver_address}")

            # Wait for pvserver to be ready
            await self._wait_for_pvserver()

        except Exception as e:
            logger.error(f"Failed to start ParaView container: {e}")
            raise

    async def _wait_for_pvserver(self, timeout: int = 60) -> None:
        """Wait for pvserver to be ready to accept connections."""
        logger.info("Waiting for pvserver to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Try to connect to the port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", self._pvserver_port))
                sock.close()

                if result == 0:
                    logger.info("pvserver is ready")

                    # If AUTO_GUI_CONNECT is enabled, wait longer for GUI to connect FIRST
                    # This is critical: GUI must connect before MCP to avoid crashes
                    if self._auto_gui_connect:
                        logger.info("Waiting for ParaView GUI to auto-connect to pvserver...")
                        logger.info("(GUI must connect FIRST before MCP connection to prevent crashes)")
                        logger.info(f"Waiting {self._gui_connect_wait_time} seconds for GUI connection...")
                        await asyncio.sleep(self._gui_connect_wait_time)
                        logger.info("GUI connection window complete, proceeding with MCP initialization")

                    return
            except Exception:
                pass

            await asyncio.sleep(0.5)

        logger.warning(f"pvserver did not become ready within {timeout} seconds")

    async def stop(self) -> None:
        """Stop and remove the ParaView Docker container."""
        if self._container is None:
            return

        try:
            logger.info(f"Stopping ParaView container: {self._docker_name}")
            await asyncio.to_thread(self._container.stop, timeout=5)
            self._container = None
            logger.info("ParaView container stopped")
        except Exception as e:
            logger.error(f"Failed to stop ParaView container: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def _to_config(self) -> ParaViewDockerManagerConfig:
        """Convert to configuration object."""
        return ParaViewDockerManagerConfig(
            image=self._image,
            novnc_port=self._novnc_port,
            vnc_port=self._vnc_port,
            pvserver_port=self._pvserver_port,
            data_dir=self._data_dir,
            inside_docker=self._inside_docker,
            network_name=self._network_name,
            auto_gui_connect=self._auto_gui_connect,
        )

    @classmethod
    def _from_config(cls, config: ParaViewDockerManagerConfig) -> ParaViewDockerManager:
        """Create instance from configuration object."""
        return cls(
            image=config.image,
            novnc_port=config.novnc_port,
            vnc_port=config.vnc_port,
            pvserver_port=config.pvserver_port,
            data_dir=config.data_dir,
            inside_docker=config.inside_docker,
            network_name=config.network_name,
            auto_gui_connect=config.auto_gui_connect,
        )
