import logging
import os
import asyncio
from typing import Optional

import docker
from docker.errors import DockerException, ImageNotFound

BROWSER_IMAGE_ENV_VAR = "MAGENTIC_UI_BROWSER_IMAGE"
PYTHON_IMAGE_ENV_VAR = "MAGENTIC_UI_PYTHON_IMAGE"

DOCKER_REGISTRY = "ghcr.io/microsoft"
BROWSER_IMAGE = os.getenv(
    BROWSER_IMAGE_ENV_VAR, f"{DOCKER_REGISTRY}/magentic-ui-browser:0.0.1"
)
PYTHON_IMAGE = os.getenv(
    PYTHON_IMAGE_ENV_VAR, f"{DOCKER_REGISTRY}/magentic-ui-python-env:0.0.1"
)


def check_docker_running() -> bool:
    try:
        client = docker.from_env()
        client.ping()  # type: ignore
        return True
    except (DockerException, ConnectionError):
        return False


def check_docker_image(image_name: str, client: docker.DockerClient) -> bool:
    try:
        client.images.get(image_name)
        return True
    except ImageNotFound:
        return False


def split_docker_repository_and_tag(image_name: str):
    if ":" in image_name:
        return image_name.rsplit(":", 1)
    return image_name, "latest"


def pull_browser_image(client: docker.DockerClient | None = None) -> None:
    client = client or docker.from_env()
    repo, tag = split_docker_repository_and_tag(BROWSER_IMAGE)
    client.images.pull(repo, tag)


def pull_python_image(client: docker.DockerClient | None = None) -> None:
    client = client or docker.from_env()
    repo, tag = split_docker_repository_and_tag(PYTHON_IMAGE)
    client.images.pull(repo, tag)


def check_docker_access():
    try:
        client = docker.from_env()
        client.ping()  # type: ignore
        return True
    except DockerException as e:
        logging.error(
            f"Error {e}: Cannot access Docker. Please refer to the TROUBLESHOOTING.md document for possible solutions."
        )
        return False


def check_browser_image(client: docker.DockerClient | None = None) -> bool:
    if not check_docker_access():
        return False
    client = client or docker.from_env()
    return check_docker_image(BROWSER_IMAGE, client)


def check_python_image(client: docker.DockerClient | None = None) -> bool:
    if not check_docker_access():
        return False
    client = client or docker.from_env()
    return check_docker_image(PYTHON_IMAGE, client)


def patch_docker_executor_with_network(executor: "DockerCommandLineCodeExecutor", network_name: str) -> None:
    """
    Monkey-patch a DockerCommandLineCodeExecutor to use a specific Docker network.

    This patches the start() method to connect the container to the specified network
    after it's created.

    Args:
        executor: The DockerCommandLineCodeExecutor instance to patch
        network_name: Name of the Docker network to connect to
    """
    from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

    original_start = executor.start

    async def patched_start() -> None:
        """Patched start method that connects container to network."""
        # Call the original start method
        await original_start()

        # Connect the container to the network
        if executor._container and network_name:
            client = docker.from_env()
            try:
                # Get or create the network
                try:
                    network = await asyncio.to_thread(client.networks.get, network_name)
                    logging.info(f"Using existing Docker network: {network_name}")
                except docker.errors.NotFound:
                    network = await asyncio.to_thread(client.networks.create, network_name, driver="bridge")
                    logging.info(f"Created Docker network: {network_name}")

                # Connect the container to the network
                await asyncio.to_thread(network.connect, executor._container)
                logging.info(f"Connected container {executor.container_name} to network {network_name}")

            except Exception as e:
                logging.error(f"Failed to connect container to network {network_name}: {e}")
                raise

    # Replace the start method
    executor.start = patched_start
