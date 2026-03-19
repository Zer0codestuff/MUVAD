import atexit
import time
import os
import subprocess
import requests
import ollama
import getpass

# Get logger
from helpers.logger import getLogger
logger = getLogger("ollama")

# Define global dict of ollama servers
ollama_servers: dict[str, subprocess.Popen] = {}

def restart_ollama_server(host: str, device: str | None = None, timeout: float = 1) -> None:
    """Restart ollama server"""

    global ollama_servers

    # Stop ollama server if exists
    stop_ollama_server(host)

    # Environment variables
    env = os.environ.copy()
    env["OLLAMA_HOST"] = host
    if device and device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device[5:]

    # Start ollama server
    logger.debug(f"Starting Ollama server on host {host}...")
    ollama_server = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait a timeout
    time.sleep(timeout)

    # Check if ollama server raise an error
    if ollama_server.returncode is None:
        ollama_servers[host] = ollama_server
        logger.info(f"Ollama server started on host {host}")
    else:
        logger.warning(f"Ollama server already running on host {host}")

    atexit.register(stop_ollama_server, host)


def stop_ollama_server(host: str) -> None:
    """Stop ollama server"""

    global ollama_servers

    # If Ollama server is started, kill it
    if host in ollama_servers:
        logger.debug(f"Stopping Ollama server on host {host}...")
        ollama_servers[host].kill()
        ollama_servers.pop(host)
        try:
            username = getpass.getuser()
            subprocess.run(
                ["pkill", "-u", username, "ollama"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
        logger.warning(f"Ollama server on host {host} stopped")


class OllamaModel:
    """Ollama model"""

    def __init__(self,
            model_name: str,
            parameters: dict = None,
            device: str | None = None,
            host: str | None = None
        ) -> None:
        if parameters is None:
            parameters = {}
        """Initialize an Ollama model with parameters"""

        # Initialize object variables
        self.model_name = model_name
        self.device = device
        self.host = host
        self.max_token = parameters.pop("max_token", None)
        self.parameters = parameters
        self.new_model_name = f"{model_name}_VAD"

        # Start ollama server
        self.restart_ollama()


    def restart_ollama(self) -> None:
        """Restart Ollama"""

        # Start ollama server
        if self.host and ("localhost" in self.host or "127.0.0.1" in self.host or "0.0.0.0" in self.host):
            restart_ollama_server(self.host, self.device)

        # Until is not started...
        status = False
        while not status:
            try:
                # Connect to ollama
                logger.debug("Connecting to Ollama...")
                self.client = ollama.Client(self.host)
                status = True

            except requests.ConnectionError as err:
                logger.error("Error on Ollama connection, retrying...")
                logger.error(f"{err}")

        logger.info("Ollama client started")

        if self.parameters:
            logger.debug("Creating custom model with parameters...")
            self.client.create(self.new_model_name, from_=self.model_name, parameters=self.parameters)
            logger.debug("Custom model with parameters created")


    def generate(self, prompt: str, images: list = None) -> str:
        if images is None:
            images = []
        """Generate new response"""

        # Until the response is generated...
        status = False
        while not status:
            try:
                # Generate response
                logger.debug("Generating response...")
                response = self.client.generate(
                    model=self.new_model_name, 
                    prompt=prompt,
                    images=images if images else None,
                    stream=True,
                )
                ret = ""
                for token in response:
                    ret += token.response
                    if self.max_token and len(ret) >= self.max_token:
                        break

                status = True

            except Exception as err:
                logger.error("Error on Ollama client, restarting Ollama and retrying...")
                logger.error(f"{err}")
                self.restart_ollama()

        logger.debug("Response generated")
        return ret
