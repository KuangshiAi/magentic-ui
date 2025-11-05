"""Local ParaView manager for running ParaView natively without Docker."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

from autogen_core import Component
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ParaViewLocalManagerConfig(BaseModel):
    """Configuration for ParaView Local Manager."""

    conda_env: str = "paraview_mcp"
    pvserver_port: int = 11111
    data_dir: Path | None = None
    auto_gui_connect: bool = True
    gui_connect_wait_time: int = 10  # Seconds to wait for GUI to connect


class ParaViewLocalManager(Component[ParaViewLocalManagerConfig]):
    """
    Manages a local ParaView installation with GUI.

    This manager handles:
    - Starting pvserver from conda environment
    - Launching ParaView GUI with auto-connect
    - Port management for pvserver
    - Process lifecycle management

    Args:
        conda_env (str): Name of conda environment with ParaView installed
        pvserver_port (int): Port for ParaView server. Default: 11111
        data_dir (Path | None): Directory to access for data
        auto_gui_connect (bool): Whether to automatically start ParaView GUI
        gui_connect_wait_time (int): Seconds to wait for GUI connection

    Properties:
        pvserver_address (str): Address for ParaView server connection
        is_running (bool): Whether pvserver is currently running

    Example:
        ```python
        manager = ParaViewLocalManager(
            conda_env="paraview_mcp",
            data_dir=Path("./data"),
            pvserver_port=11111
        )
        await manager.start()
        # ParaView GUI will open automatically
        # Connect MCP server to manager.pvserver_address
        await manager.stop()
        ```
    """

    component_config_schema = ParaViewLocalManagerConfig
    component_type = "other"

    def __init__(
        self,
        *,
        conda_env: str = "paraview_mcp",
        pvserver_port: int = 11111,
        data_dir: Path | None = None,
        auto_gui_connect: bool = True,
        gui_connect_wait_time: int = 10,
    ):
        super().__init__()
        self._conda_env = conda_env
        self._pvserver_port = pvserver_port
        self._data_dir = data_dir
        self._auto_gui_connect = auto_gui_connect
        self._gui_connect_wait_time = gui_connect_wait_time

        self._pvserver_process: Optional[subprocess.Popen] = None
        self._gui_process: Optional[subprocess.Popen] = None
        self._hostname = "localhost"

    @property
    def pvserver_address(self) -> str:
        """Get the address for ParaView server connection."""
        return f"{self._hostname}:{self._pvserver_port}"

    @property
    def pvserver_port(self) -> int:
        """Get the ParaView server port."""
        return self._pvserver_port

    @property
    def is_running(self) -> bool:
        """Check if pvserver is running."""
        if self._pvserver_process is None:
            return False
        return self._pvserver_process.poll() is None

    async def start(self) -> None:
        """Start the local ParaView server and GUI."""
        if self.is_running:
            logger.info("ParaView server is already running")
            return

        try:
            # Start pvserver
            await self._start_pvserver()

            # Wait for pvserver to be ready
            await self._wait_for_pvserver()

            # Start ParaView GUI with auto-connect
            if self._auto_gui_connect:
                await self._start_gui()

            logger.info(f"ParaView started. pvserver available at: {self.pvserver_address}")

        except Exception as e:
            logger.error(f"Failed to start ParaView: {e}")
            await self.stop()
            raise

    async def _start_pvserver(self) -> None:
        """Start pvserver process."""
        logger.info(f"Starting pvserver on port {self._pvserver_port}...")

        # Get conda environment paths
        conda_base = Path.home() / "miniconda3"
        env_path = conda_base / "envs" / self._conda_env
        pvserver_bin = env_path / "bin" / "pvserver"

        if not pvserver_bin.exists():
            raise FileNotFoundError(f"pvserver not found at {pvserver_bin}")

        # Start pvserver with multi-client support
        cmd = [
            str(pvserver_bin),
            f"--server-port={self._pvserver_port}",
            "--multi-clients",
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        self._pvserver_process = await asyncio.to_thread(
            subprocess.Popen,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._get_conda_env(),
        )

        logger.info(f"pvserver started with PID {self._pvserver_process.pid}")

    async def _start_gui(self) -> None:
        """Start ParaView GUI with auto-connect to pvserver."""
        logger.info("Starting ParaView GUI with auto-connect...")

        # Get paths
        conda_base = Path.home() / "miniconda3"
        env_path = conda_base / "envs" / self._conda_env
        paraview_bin = env_path / "bin" / "paraview"

        if not paraview_bin.exists():
            raise FileNotFoundError(f"paraview not found at {paraview_bin}")

        # CRITICAL FIX: Use ParaView's built-in --server-url option instead of a script
        # This ensures proper client registration for multi-client sync
        # Format: --server-url=cs://hostname:port
        server_url = f"cs://localhost:{self._pvserver_port}"

        cmd = [
            str(paraview_bin),
            f"--server-url={server_url}",
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        logger.info(f"GUI will connect to: {server_url}")

        self._gui_process = await asyncio.to_thread(
            subprocess.Popen,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._get_conda_env(),
        )

        logger.info(f"ParaView GUI started with PID {self._gui_process.pid}")

        # Wait for GUI to connect and create its view
        logger.info(f"Waiting {self._gui_connect_wait_time}s for GUI to connect...")
        await asyncio.sleep(self._gui_connect_wait_time)
        logger.info("GUI connection window complete, ready for MCP connection")

    def _create_auto_connect_script(self) -> str:
        """Create auto-connect Python script for ParaView GUI."""
        import tempfile

        script_content = f"""#!/usr/bin/env python3
'''Auto-connect script for ParaView GUI to connect to pvserver.'''

import os
import sys
import time

from paraview import servermanager
from paraview.simple import *

def auto_connect_to_server():
    '''Connect ParaView GUI to the pvserver with retry logic.'''
    pv_host = 'localhost'
    pv_port = {self._pvserver_port}

    max_retries = 20
    for attempt in range(max_retries):
        try:
            # Connect to pvserver
            connection = servermanager.Connect(pv_host, pv_port)

            if connection:
                # Give the connection a moment to fully establish
                time.sleep(0.5)

                # Create or get the render view
                try:
                    renderView1 = CreateRenderView() if GetActiveView() is None else GetActiveView()
                    renderView1.ResetCamera()

                    # Set white background
                    LoadPalette(paletteName='WhiteBackground')
                    Render(renderView1)

                    time.sleep(1.0)
                except:
                    pass

                print(f"Successfully connected to pvserver at {{pv_host}}:{{pv_port}}")
                return True
            else:
                time.sleep(0.5)

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                print(f"Failed to connect after {{max_retries}} attempts: {{e}}")
                return False

    return False

# Execute auto-connect
try:
    auto_connect_to_server()
except Exception as e:
    print(f"Auto-connect error: {{e}}")
"""

        # Write to temporary file
        fd, path = tempfile.mkstemp(suffix=".py", prefix="paraview_autoconnect_")
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)

        logger.info(f"Created auto-connect script at {path}")
        return path

    def _get_conda_env(self) -> dict:
        """Get environment variables for conda environment."""
        conda_base = Path.home() / "miniconda3"
        env_path = conda_base / "envs" / self._conda_env

        env = os.environ.copy()
        env["PATH"] = f"{env_path / 'bin'}:{env.get('PATH', '')}"
        env["CONDA_DEFAULT_ENV"] = self._conda_env
        env["CONDA_PREFIX"] = str(env_path)

        return env

    async def _wait_for_pvserver(self, timeout: int = 60) -> None:
        """Wait for pvserver to be ready to accept connections."""
        logger.info("Waiting for pvserver to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", self._pvserver_port))
                sock.close()

                if result == 0:
                    logger.info("pvserver is ready and accepting connections")
                    return
            except Exception:
                pass

            await asyncio.sleep(0.5)

        raise TimeoutError(f"pvserver did not become ready within {timeout} seconds")

    async def stop(self) -> None:
        """Stop the ParaView server and GUI."""
        logger.info("Stopping ParaView...")

        # Stop GUI process
        if self._gui_process is not None:
            try:
                logger.info(f"Stopping ParaView GUI (PID: {self._gui_process.pid})")
                self._gui_process.terminate()
                await asyncio.to_thread(self._gui_process.wait, timeout=5)
                self._gui_process = None
                logger.info("ParaView GUI stopped")
            except Exception as e:
                logger.warning(f"Error stopping GUI: {e}")
                try:
                    self._gui_process.kill()
                except:
                    pass

        # Stop pvserver process
        if self._pvserver_process is not None:
            try:
                logger.info(f"Stopping pvserver (PID: {self._pvserver_process.pid})")
                self._pvserver_process.terminate()
                await asyncio.to_thread(self._pvserver_process.wait, timeout=5)
                self._pvserver_process = None
                logger.info("pvserver stopped")
            except Exception as e:
                logger.warning(f"Error stopping pvserver: {e}")
                try:
                    self._pvserver_process.kill()
                except:
                    pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def _to_config(self) -> ParaViewLocalManagerConfig:
        """Convert to configuration object."""
        return ParaViewLocalManagerConfig(
            conda_env=self._conda_env,
            pvserver_port=self._pvserver_port,
            data_dir=self._data_dir,
            auto_gui_connect=self._auto_gui_connect,
            gui_connect_wait_time=self._gui_connect_wait_time,
        )

    @classmethod
    def _from_config(cls, config: ParaViewLocalManagerConfig) -> ParaViewLocalManager:
        """Create instance from configuration object."""
        return cls(
            conda_env=config.conda_env,
            pvserver_port=config.pvserver_port,
            data_dir=config.data_dir,
            auto_gui_connect=config.auto_gui_connect,
            gui_connect_wait_time=config.gui_connect_wait_time,
        )
