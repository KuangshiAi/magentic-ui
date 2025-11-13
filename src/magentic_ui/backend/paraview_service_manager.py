"""
ParaView Service Manager

Manages ParaView server lifecycle at the session level, independent of agent initialization.
This allows the ParaView UI to be available before any conversation starts.
"""

from typing import Optional, Dict
from loguru import logger
import asyncio

from ..tools.paraview_docker.paraview_docker_manager import ParaViewDockerManager


class ParaViewServiceManager:
    """
    Manages ParaView server instances at the session level.

    Responsibilities:
    - Start/stop ParaView Docker containers
    - Track noVNC port for UI access
    - Provide pvserver connection info for MCP agents
    - Clean up resources on session deletion
    """

    # Class-level registry of active ParaView services by session ID
    _active_services: Dict[int, "ParaViewServiceManager"] = {}
    _lock = asyncio.Lock()

    def __init__(
        self,
        session_id: int,
        docker_manager: ParaViewDockerManager,
    ):
        """
        Initialize a ParaView service for a specific session.

        Args:
            session_id: The session ID this service is associated with
            docker_manager: Configured ParaViewDockerManager instance
        """
        self.session_id = session_id
        self.docker_manager = docker_manager
        self.novnc_port: Optional[int] = None
        self.pvserver_port: Optional[int] = None
        self._is_running = False

    @classmethod
    async def get_or_create(
        cls,
        session_id: int,
        docker_manager: Optional[ParaViewDockerManager] = None,
    ) -> "ParaViewServiceManager":
        """
        Get existing ParaView service for a session or create a new one.

        Args:
            session_id: The session ID
            docker_manager: ParaViewDockerManager to use (required for new services)

        Returns:
            ParaViewServiceManager instance
        """
        async with cls._lock:
            if session_id in cls._active_services:
                logger.info(f"Reusing existing ParaView service for session {session_id}")
                return cls._active_services[session_id]

            if docker_manager is None:
                raise ValueError("docker_manager is required when creating a new ParaView service")

            logger.info(f"Creating new ParaView service for session {session_id}")
            service = cls(session_id, docker_manager)
            cls._active_services[session_id] = service
            return service

    @classmethod
    async def get(cls, session_id: int) -> Optional["ParaViewServiceManager"]:
        """
        Get existing ParaView service for a session.

        Args:
            session_id: The session ID

        Returns:
            ParaViewServiceManager instance or None if not found
        """
        async with cls._lock:
            return cls._active_services.get(session_id)

    @classmethod
    async def remove(cls, session_id: int) -> None:
        """
        Remove a ParaView service from the registry.

        Args:
            session_id: The session ID
        """
        async with cls._lock:
            if session_id in cls._active_services:
                logger.info(f"Removing ParaView service for session {session_id} from registry")
                del cls._active_services[session_id]

    async def start(self) -> Dict[str, any]:
        """
        Start the ParaView Docker container and wait for it to be ready.

        Returns:
            Dict with port information and status
        """
        if self._is_running:
            logger.info(f"ParaView service for session {self.session_id} is already running")
            return self._get_connection_info()

        try:
            logger.info(f"Starting ParaView Docker container for session {self.session_id}")

            # Start the Docker container (includes pvserver startup and GUI auto-connect wait)
            await self.docker_manager.start()

            # Capture port information
            self.novnc_port = self.docker_manager.novnc_port
            self.pvserver_port = self.docker_manager.pvserver_port

            self._is_running = True

            logger.info(
                f"ParaView service started for session {self.session_id} - "
                f"noVNC: {self.novnc_port}, pvserver: {self.pvserver_port}"
            )

            return self._get_connection_info()

        except Exception as e:
            logger.error(f"Failed to start ParaView service for session {self.session_id}: {e}")
            raise

    async def stop(self) -> None:
        """
        Stop the ParaView Docker container and clean up resources.
        """
        if not self._is_running:
            logger.info(f"ParaView service for session {self.session_id} is not running")
            return

        try:
            logger.info(f"Stopping ParaView Docker container for session {self.session_id}")
            await self.docker_manager.stop()
            self._is_running = False
            logger.info(f"ParaView service stopped for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error stopping ParaView service for session {self.session_id}: {e}")
            raise

    def _get_connection_info(self) -> Dict[str, any]:
        """Get connection information for this ParaView service."""
        return {
            "status": "running" if self._is_running else "stopped",
            "session_id": self.session_id,
            "novnc_port": self.novnc_port,
            "pvserver_port": self.pvserver_port,
            "novnc_url": f"http://localhost:{self.novnc_port}/vnc.html" if self.novnc_port else None,
        }

    def get_pvserver_connection_params(self) -> Dict[str, any]:
        """
        Get pvserver connection parameters for MCP agent initialization.

        Returns:
            Dict with host and port for pvserver connection
        """
        if not self._is_running:
            raise RuntimeError(f"ParaView service for session {self.session_id} is not running")

        # Use container name as host (for Docker network communication)
        return {
            "host": "paraview-server",  # Static container name
            "port": self.pvserver_port or 11111,
        }

    def is_running(self) -> bool:
        """Check if the ParaView service is running."""
        return self._is_running

    def get_novnc_url(self) -> Optional[str]:
        """Get the noVNC URL for browser access."""
        if self._is_running and self.novnc_port:
            return f"http://localhost:{self.novnc_port}/vnc.html"
        return None
