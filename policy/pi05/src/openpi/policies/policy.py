from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            self._profile_fns = None

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, profile: bool = False) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        if profile and not self._is_pytorch_model:
            outputs = self._infer_jax_profile(sample_rng_or_pytorch_device, inputs, observation, sample_kwargs)
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]) if hasattr(x, "shape") and x.shape[:1] == (1,) else x, outputs)
            profile_data = outputs.pop("policy_profile")
            outputs = self._output_transform(outputs)
            outputs["policy_profile"] = profile_data
            return outputs

        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _ensure_jax_profile_fns(self):
        if self._profile_fns is not None:
            return self._profile_fns
        required = ("profile_embed_prefix", "profile_prefill", "profile_denoise_step")
        missing = [name for name in required if not hasattr(self._model, name)]
        if missing:
            raise ValueError(f"Model does not support profiling helpers: {missing}")
        self._profile_fns = {
            "embed_prefix": nnx_utils.module_jit(self._model.profile_embed_prefix),
            "prefill": nnx_utils.module_jit(self._model.profile_prefill),
            "denoise_step": nnx_utils.module_jit(self._model.profile_denoise_step),
        }
        return self._profile_fns

    @staticmethod
    def _sync(value):
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)
        return value

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000

    def _infer_jax_profile(self, rng: at.KeyArrayLike, inputs: dict, observation: _model.Observation, sample_kwargs: dict):
        profile_fns = self._ensure_jax_profile_fns()
        num_steps = int(sample_kwargs.get("num_steps", 10))
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        total_start = time.perf_counter()

        start = time.perf_counter()
        observation, prefix_tokens, prefix_mask, prefix_ar_mask = profile_fns["embed_prefix"](observation)
        self._sync((observation, prefix_tokens, prefix_mask, prefix_ar_mask))
        embed_prefix_ms = self._elapsed_ms(start)

        start = time.perf_counter()
        kv_cache = profile_fns["prefill"](prefix_tokens, prefix_mask, prefix_ar_mask)
        self._sync(kv_cache)
        prefill_ms = self._elapsed_ms(start)

        start = time.perf_counter()
        if "noise" in sample_kwargs:
            x_t = sample_kwargs["noise"]
        else:
            x_t = jax.random.normal(rng, (observation.state.shape[0], self._model.action_horizon, self._model.action_dim))
        self._sync(x_t)
        noise_ms = self._elapsed_ms(start)

        dt = jnp.asarray(-1.0 / num_steps, dtype=x_t.dtype)
        step_time = jnp.asarray(1.0, dtype=x_t.dtype)
        denoise_steps_ms = []
        for _ in range(num_steps):
            start = time.perf_counter()
            x_t, step_time = profile_fns["denoise_step"](
                observation, prefix_tokens, prefix_mask, kv_cache, x_t, step_time, dt
            )
            self._sync((x_t, step_time))
            denoise_steps_ms.append(self._elapsed_ms(start))

        total_ms = self._elapsed_ms(total_start)
        return {
            "state": inputs["state"],
            "actions": x_t,
            "policy_profile": {
                "total_ms": total_ms,
                "vlm_embed_prefix_ms": embed_prefix_ms,
                "vlm_prefill_ms": prefill_ms,
                "noise_init_ms": noise_ms,
                "vlm_decode_action_expert_steps_ms": denoise_steps_ms,
                "action_expert_denoise_steps_ms": denoise_steps_ms,
                "action_expert_denoise_total_ms": float(sum(denoise_steps_ms)),
                "action_expert_denoise_mean_ms": float(sum(denoise_steps_ms) / len(denoise_steps_ms)),
                "num_denoise_steps": num_steps,
            },
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
