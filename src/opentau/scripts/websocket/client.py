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

"""Simple WebSocket client for testing OpenTau policy server."""

import argparse
import numpy as np
from openpi_client import websocket_client_policy


def create_fake_obs(num_cams=2, resolution=(224, 224), state_dim=8):
    """Create fake observation."""
    images = [
        np.random.randint(0, 256, size=(*resolution, 3), dtype=np.uint8)
        for _ in range(num_cams)
    ]
    return {
        "images": images,
        "state": np.random.randn(state_dim).astype(np.float32),
        "prompt": "pick up the red block",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    for _ in range(10):
        obs = create_fake_obs()
        action = policy.infer(obs)
        print(f"Action shape: {action['action'].shape}")


if __name__ == "__main__":
    main()
