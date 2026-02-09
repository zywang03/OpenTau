# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebSocket server for OpenTau policy inference.

This server wraps an OpenTau policy using the OpenPI BasePolicy interface,
allowing it to be accessed via OpenPI's websocket client.

Usage:
    python src/opentau/scripts/websocket/server.py --config_path=/path/to/config.json \\
        --host=0.0.0.0 --port=8765
"""

import asyncio
import http
import logging
import time
import traceback
from dataclasses import asdict
from pprint import pformat

import torch

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.factory import get_policy_class
from opentau.scripts.websocket.policy_adapter import OpenTauPolicyAdapter
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    attempt_torch_compile,
    auto_torch_device,
    init_logging,
)

logger = logging.getLogger(__name__)


class OpenTauWebsocketPolicyServer:
    """WebSocket server for serving OpenTau policy via OpenPI protocol.

    This server wraps an OpenTau policy and serves it using the OpenPI websocket
    protocol, allowing OpenPI clients to connect and request inference.
    """

    def __init__(
        self,
        policy_adapter: OpenTauPolicyAdapter,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Initialize the server.

        Args:
            policy_adapter: The OpenTau policy adapter implementing OpenPI BasePolicy.
            host: Host address to bind to.
            port: Port to bind to.
            metadata: Optional metadata dictionary to send to clients on connection.
        """
        self._policy_adapter = policy_adapter
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """Start the server and run forever."""
        asyncio.run(self.run())

    async def run(self):
        """Run the async server."""
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info(f"WebSocket server started on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """Handle a websocket connection.

        Args:
            websocket: The websocket connection.
        """
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send metadata to client on connection
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy_adapter.infer(obs)
                infer_time = time.monotonic() - infer_time

                # Add server timing information
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                error_msg = traceback.format_exc()
                logger.exception("Error during inference")
                await websocket.send(error_msg)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """Handle health check requests.

    Args:
        connection: The server connection.
        request: The HTTP request.

    Returns:
        HTTP response if path is /healthz, None otherwise.
    """
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def create_server(
    cfg: TrainPipelineConfig,
    host: str = "0.0.0.0",
    port: int | None = None,
    metadata: dict | None = None,
) -> OpenTauWebsocketPolicyServer:
    """Create and initialize a websocket policy server.

    Args:
        cfg: Training pipeline configuration including policy settings.
        host: Host address to bind to.
        port: Port to bind to.
        metadata: Optional metadata dictionary.

    Returns:
        Initialized OpenTauWebsocketPolicyServer instance.
    """
    device = auto_torch_device()
    dtype = torch.bfloat16

    logger.info(f"Initializing policy on device: {device}")

    # Load the policy model
    logger.info(f"Loading policy from: {cfg.policy.pretrained_path}")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.to(device=device, dtype=dtype)
    policy.eval()
    policy = attempt_torch_compile(policy, device_hint=device)
    policy.reset()
    logger.info("Policy loaded successfully")

    # Create adapter
    policy_adapter = OpenTauPolicyAdapter(policy, cfg, device, dtype)

    # Create server metadata
    if metadata is None:
        metadata = {
            "policy_type": cfg.policy.type,
            "device": str(device),
            "dtype": str(dtype),
            "num_cams": cfg.num_cams,
            "resolution": cfg.resolution,
            "max_state_dim": cfg.max_state_dim,
        }

    server = OpenTauWebsocketPolicyServer(policy_adapter, host=host, port=port, metadata=metadata)
    return server


@parser.wrap()
def server_main(cfg: TrainPipelineConfig):
    """Main entry point for the websocket server.

    Args:
        cfg: Training pipeline configuration parsed from CLI/config file.
    
    Command line arguments:
        --host: Host address to bind to (default: 0.0.0.0)
        --port: Port to bind to (default: 8765)
    """
    logging.info(pformat(asdict(cfg)))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Get server settings from command line, config, or use defaults
    host = parser.parse_arg("host") or getattr(cfg, "websocket_host", "0.0.0.0")
    port_str = parser.parse_arg("port")
    if port_str:
        port = int(port_str)
    else:
        port = getattr(cfg, "websocket_port", 8765)

    server = create_server(cfg, host=host, port=port)
    logger.info(f"Starting WebSocket server on {host}:{port}")
    logger.info(f"Policy: {cfg.policy.type}")
    server.serve_forever()


if __name__ == "__main__":
    init_logging()
    server_main()
