# GR00T N1.7 PolicyServer wire format

This is the wire format that `sim/policies/groot.py` speaks. The module docstring there is the source of truth; this page expands on it.

**Verification status.** The format was read from NVIDIA/Isaac-GR00T at commit `7aa0bca50f1097f8b17f5c274c98264b72a1419f`, in `gr00t/policy/server_client.py` and `gr00t/policy/gr00t_policy.py`. The client has only been tested against an in-process fake server that implements this format (in `tests/`). It has **not** been run against a live GR00T server. Treat the first live run as an integration test, and run it with the robot disabled.

## Transport

ZeroMQ REQ/REP over TCP (default port 5555). Each message is one msgpack map. The client is strictly request, then reply. It never retries: on a timeout (`GROOT_TIMEOUT_MS`, default 5000) it closes the socket and stops the run with a safety stop.

## Request

```
{"endpoint": <str>, "data": <map, optional>, "api_token": <str, optional>}
```

`api_token` is sent only when `GROOT_API_TOKEN` or `--api-token` is set.

| Endpoint | `data` | Reply | Used by this client |
| --- | --- | --- | --- |
| `ping` | none | health reply | not by `sim` (used by `examples/groot_ping.py`) |
| `get_modality_config` | none | modality config per group | yes, once per episode on reset |
| `reset` | `{"options": null}` | server state reset | yes, on every episode reset |
| `get_action` | `{"observation": obs, "options": null}` | `[action, info]` | yes, each time a new chunk is needed |
| `kill` | none | server shuts down | never sent |

If the server answers with a map containing `"error"`, the client raises a safety stop with that message.

### `get_action` observation

All arrays carry a batch axis `B` and a time axis `T`. This client always sends `B = 1`, `T = 1`.

```
obs = {
  "video":    {<video key>: uint8   (B, T, H, W, C)},
  "state":    {<state key>: float32 (B, T, D)},
  "language": {<language key>: [[str]]},      # B lists of T strings
}
```

With `config/embodiment.sim.json` that is `video.front_camera` `(1, 1, 224, 224, 3)`, `state.arm_joint_positions` `(1, 1, 6)`, `state.gripper_position` `(1, 1, 1)` and `language.task_description` `[["pick up the red cube"]]`.

### `get_action` response

A two-element list `[action, info]`.

```
action = {<action key>: float32 (B, T, D)}    # T = action horizon (16)
```

The client checks that every configured action key is present, is a 3-D array with `B = 1` and last dimension equal to the configured `dim`, and that all keys agree on `T`. It concatenates the keys in config order into a `(T, 7)` chunk, which the safety layer then clamps. Anything else is a safety stop. `info` is ignored.

## Array encoding

The two directions use different encodings.

- **Client to server:** each array becomes `{"__ndarray_class__": True, "as_npy": <bytes>}`, where the bytes come from `np.save(..., allow_pickle=False)`. The server decodes with `allow_pickle=False`. The client refuses to send object-dtype arrays.
- **Server to client:** msgpack-numpy form, `{"nd": True, "type": <dtype str>, "kind": ..., "shape": [...], "data": <raw bytes>}`. The client also accepts the `__ndarray_class__` form. It refuses object-dtype (`kind` `"O"`) arrays, because decoding them would mean unpickling data from the network. `msgpack-numpy` itself is not a dependency; both encodings are handled in `sim/policies/groot.py`.

## Modality-key check on reset

On every `reset` the client first calls `get_modality_config`. For each group (`video`, `state`, `language`, `action`) it reads the server's `modality_keys` and compares them as sets with the keys in the embodiment config. Any mismatch is a safety stop before any observation is sent. If the server returns no modality config, or a group without keys, that group is skipped and the per-call shape checks above still apply. Entries wrapped as `{"as_json": ...}` are unwrapped first.

## What must match on the server side

- The server must run a **fine-tuned `NEW_EMBODIMENT` model**. The base `nvidia/GR00T-N1.7-3B` checkpoint does not know this robot.
- Its modality config must use **this repo's key names**: video `front_camera`; state `arm_joint_positions` and `gripper_position`; language `task_description`; action `arm_joint_delta` and `gripper_command`. Change the names in `config/embodiment.sim.json` and the server's modality config together, or not at all.
- Dimensions must match the config: image `224x224x3`, state dims 6 and 1, action dims 6 and 1, horizon 16.
- Action semantics must match: `arm_joint_delta` is a per-step joint delta in radians (limit 0.05), `gripper_command` is 0 (open) to 1 (closed). Out-of-range values are clamped, not rejected.
- If the server uses an API token, pass the same value through `GROOT_API_TOKEN`.
- The server must be reachable on a trusted network only. The wire protocol has no encryption, and the token is sent in clear text.
