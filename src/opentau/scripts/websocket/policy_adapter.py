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

"""Adapter to wrap OpenTau policy as OpenPI BasePolicy interface."""

import logging
from typing import Dict

import numpy as np
import torch

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)


class OpenTauPolicyAdapter(_base_policy.BasePolicy):
    """Adapter that wraps an OpenTau policy to implement the OpenPI BasePolicy interface.

    This adapter converts between OpenPI's dictionary-based observation/action format
    and OpenTau's tensor-based batch format.
    """

    def __init__(
        self,
        opentau_policy,
        config,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the adapter.

        Args:
            opentau_policy: The OpenTau policy instance (e.g., PI0Policy, PI05Policy).
            config: TrainPipelineConfig containing policy configuration.
            device: Torch device to use for tensors.
            dtype: Torch dtype to use for tensors.
        """
        self._policy = opentau_policy
        self._config = config
        self._device = device
        self._dtype = dtype

    def _decode_image(self, image: np.ndarray) -> torch.Tensor:
        """Decode an image from numpy array to torch tensor.

        Args:
            image: Numpy array of shape (H, W, C) with values in [0, 1] or [0, 255].

        Returns:
            Tensor of shape (1, C, H, W) normalized to [0, 1].
        """
        from PIL import Image

        # Handle different input formats
        if image.dtype == np.uint8:
            # Image is in [0, 255] range
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            # Image might be in [0, 255] range
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # Ensure shape is (H, W, C)
        if image.ndim == 2:
            # Grayscale, add channel dimension
            image = np.expand_dims(image, axis=-1)
        if image.ndim == 3 and image.shape[2] == 1:
            # Single channel, convert to RGB
            image = np.repeat(image, 3, axis=2)

        # Resize if needed
        # if image.shape[:2] != self._config.resolution:
        #     image_pil = Image.fromarray((image * 255).astype(np.uint8))
        #     image_pil = image_pil.resize(self._config.resolution[::-1])  # PIL uses (W, H)
        #     image = np.array(image_pil, dtype=np.float32) / 255.0

        # Convert to (C, H, W) tensor and add batch dimension: (1, C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(device=self._device, dtype=self._dtype)

    def _convert_obs_to_batch(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """Convert OpenPI observation dict to OpenTau batch format.

        Expected obs format:
        - "images": list of numpy arrays, each of shape (H, W, C) with values in [0, 1] or [0, 255]
        - "state": numpy array of shape (state_dim,)
        - "prompt": string (optional, defaults to "")

        Returns:
            Dictionary with keys: "camera0", "camera1", ..., "img_is_pad", "state", "prompt"
        """
        batch = {}

        # Process camera images - similar to grpc server
        img_is_pad = []
        images = obs.get("images", [])
        
        for i, camera_image in enumerate(images):
            camera_name = f"camera{i}"
            if isinstance(camera_image, np.ndarray):
                batch[camera_name] = self._decode_image(camera_image)
                img_is_pad.append(False)
            else:
                raise ValueError(f"Image {i} must be a numpy array, got {type(camera_image)}")

        # Fill in missing cameras with zeros
        for i in range(len(images), self._config.num_cams):
            batch[f"camera{i}"] = torch.zeros(
                (1, 3, *self._config.resolution),
                dtype=self._dtype,
                device=self._device,
            )
            img_is_pad.append(True)

        batch["img_is_pad"] = torch.tensor([img_is_pad], dtype=torch.bool, device=self._device)

        # Process robot state
        if "state" in obs:
            state = obs["state"]
            if isinstance(state, np.ndarray):
                state = state.flatten().tolist()
            elif isinstance(state, (list, tuple)):
                state = list(state)
            else:
                raise ValueError(f"State must be numpy array, list, or tuple, got {type(state)}")

            # Pad to max_state_dim if needed
            if len(state) < self._config.max_state_dim:
                state.extend([0.0] * (self._config.max_state_dim - len(state)))

            batch["state"] = torch.tensor(
                # [state[: self._config.max_state_dim]],
                [state],
                dtype=self._dtype,
                device=self._device,
            )
        else:
            raise ValueError("Robot state is required but was not provided in observation")

        # Process prompt
        prompt = obs.get("prompt", "")
        batch["prompt"] = [prompt] if prompt else [""]

        return batch

    def _convert_action_to_dict(self, action_tensor) -> Dict:
        """Convert OpenTau action tensor to OpenPI action dict.

        Args:
            action_tensor: Tensor of shape (n_action_steps, batch_size=1, action_dim)
                          or (batch_size=1, action_dim) or (action_dim,)

        Returns:
            Dictionary with "action" key containing numpy array.
        """
        # Convert to numpy
        if isinstance(action_tensor, torch.Tensor):
            action_np = action_tensor.to("cpu", torch.float32).detach().numpy()
        else:
            action_np = np.asarray(action_tensor, dtype=np.float32)

        # Handle different tensor shapes
        if action_np.ndim == 3:
            # (n_action_steps, batch_size=1, action_dim) -> (n_action_steps, action_dim)
            action_np = action_np.squeeze(1)
        elif action_np.ndim == 2:
            # (batch_size=1, action_dim) -> (action_dim,)
            if action_np.shape[0] == 1:
                action_np = action_np.squeeze(0)
        # If ndim == 1, already in correct shape

        return {"action": action_np}

    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations (OpenPI interface).

        Args:
            obs: Observation dictionary with keys:
                 - "images": list of numpy arrays, each of shape (H, W, C) in [0, 1] or [0, 255]
                 - "state": numpy array of robot state
                 - "prompt": string (optional, defaults to "")

        Returns:
            Action dictionary with "action" key containing numpy array.
        """
        # Convert observation to OpenTau batch format
        batch = self._convert_obs_to_batch(obs)

        # Run inference
        with torch.inference_mode():
            action_tensor = self._policy.sample_actions(batch)
            # action_tensor shape: (n_action_steps, batch_size=1, action_dim)

        # Convert action tensor to dictionary
        return self._convert_action_to_dict(action_tensor)

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        self._policy.reset()
