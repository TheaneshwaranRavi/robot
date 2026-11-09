"""Remote GR00T N1.7 PolicyServer backend.

Wire format checked against NVIDIA/Isaac-GR00T @ 7aa0bca (gr00t/policy/server_client.py and
gr00t_policy.py); see docs/GROOT_PROTOCOL.md. Summary:

    request  = {"endpoint": "get_action", "data": {"observation": obs, "options": None},
                "api_token": ...}
    obs      = {"video":    {key: uint8   (B, T, H, W, C)},
                "state":    {key: float32 (B, T, D)},
                "language": {key: [[str]]}}           # B lists of T strings
    response = [action, info],  action = {key: float32 (B, T=horizon, D)}

Arrays go out as {"__ndarray_class__": True, "as_npy": <np.save bytes>}, which the server decodes
with allow_pickle=False. Arrays come back in msgpack-numpy's {"nd", "type", "shape", "data"} form.
Both are handled here, so msgpack-numpy is not a dependency.
"""
from __future__ import annotations

import io
import json
from typing import Any

import msgpack
import numpy as np
import zmq

from ..safety import SafetyStop, require_remote_ready
from .base import Policy


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            raise TypeError("refusing to send an object-dtype array")
        buffer = io.BytesIO()
        np.save(buffer, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}
    return obj


def _get(obj: dict, key: str) -> Any:
    return obj.get(key, obj.get(key.encode()))


def _decode(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    if _get(obj, "__ndarray_class__"):
        return np.load(io.BytesIO(_get(obj, "as_npy")), allow_pickle=False)
    if _get(obj, "nd"):
        kind = _get(obj, "kind")
        if kind in (b"O", "O"):
            raise SafetyStop("policy server sent an object-dtype (pickle) array; refusing it")
        dtype = _get(obj, "type")
        dtype = np.dtype(dtype.decode() if isinstance(dtype, bytes) else dtype)
        return np.frombuffer(_get(obj, "data"), dtype=dtype).reshape(_get(obj, "shape")).copy()
    return obj


def pack(message: Any) -> bytes:
    return msgpack.packb(message, default=_encode, use_bin_type=True)


def unpack(payload: bytes) -> Any:
    return msgpack.unpackb(payload, object_hook=_decode, raw=False)


class GrootPolicy(Policy):
    name = "groot"

    def __init__(self, config: dict[str, Any], host: str, port: int, timeout_ms: int,
                 api_token: str | None = None) -> None:
        require_remote_ready(config)  # never talk to a server about a placeholder robot
        self.config = config
        self.address = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.horizon = config["action_horizon"]
        self.video_keys = [c["name"] for c in config["observation"]["video"]]
        self.state_keys = [s["name"] for s in config["observation"]["state"]]
        self.language_key = config["observation"]["language"]
        self.actions = [(a["name"], a["dim"]) for a in config["action"]]
        self.context = zmq.Context()
        self.socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(self.address)

    def call(self, endpoint: str, data: dict | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(pack(request))
            response = unpack(self.socket.recv())
        except zmq.error.Again:
            self._connect()  # a timed-out REQ socket cannot send again
            raise SafetyStop(f"GR00T server at {self.address} did not answer within {self.timeout_ms} ms")
        if isinstance(response, dict) and "error" in response:
            raise SafetyStop(f"GR00T server error: {response['error']}")
        return response

    def check_modality_keys(self) -> None:
        """Refuse to run if the server's modality config names different keys than ours."""
        modality = self.call("get_modality_config")
        if not modality:
            return  # server exposes none; per-call validation still applies
        ours = {"video": set(self.video_keys), "state": set(self.state_keys),
                "language": {self.language_key}, "action": {name for name, _ in self.actions}}
        for group, expected in ours.items():
            entry = modality.get(group) if isinstance(modality, dict) else None
            if isinstance(entry, dict) and "as_json" in entry:
                entry = entry["as_json"]
                entry = json.loads(entry) if isinstance(entry, (str, bytes)) else entry
            keys = set(entry.get("modality_keys", [])) if isinstance(entry, dict) else None
            if keys is not None and keys != expected:
                raise SafetyStop(f"server {group} keys {sorted(keys)} != config keys {sorted(expected)}")

    def reset(self, task: str) -> None:
        super().reset(task)
        self.check_modality_keys()
        self.call("reset", {"options": None})

    def build_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "video": {k: np.asarray(obs[k], dtype=np.uint8)[None, None] for k in self.video_keys},
            "state": {k: np.asarray(obs[k], dtype=np.float32)[None, None] for k in self.state_keys},
            "language": {self.language_key: [[str(obs[self.language_key])]]},
        }

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        response = self.call("get_action", {"observation": self.build_observation(obs), "options": None})
        if not isinstance(response, (list, tuple)) or len(response) != 2 or not isinstance(response[0], dict):
            raise SafetyStop(f"malformed GR00T reply: expected [action, info], got {type(response).__name__}")
        action = response[0]
        parts = []
        for name, dim in self.actions:
            value = action.get(name)
            if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != dim:
                shape = getattr(value, "shape", None)
                raise SafetyStop(f"GR00T action {name!r} has shape {shape}, expected (1, T, {dim})")
            parts.append(value[0].astype(np.float64))
        steps = {p.shape[0] for p in parts}
        if len(steps) != 1:
            raise SafetyStop(f"GR00T action keys disagree on horizon: {sorted(steps)}")
        return np.concatenate(parts, axis=1)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()
