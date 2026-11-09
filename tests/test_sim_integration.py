"""MuJoCo and remote-policy integration tests for the virtual robot."""
from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

try:
    import mujoco  # noqa: F401
    import numpy as np
except (ImportError, OSError):
    HAS_SIM_DEPENDENCIES = False
else:
    HAS_SIM_DEPENDENCIES = True

if HAS_SIM_DEPENDENCIES:
    import zmq

    from sim import run as sim_run
    from sim.config import ROOT, SIM_CONFIG, load_config
    from sim.env import SimEnv
    from sim.policies import groot
    from sim.policies.scripted import ScriptedPickPolicy
    from sim.safety import ActionLimits, SafetyStop, Watchdog


_NO_REPLY = object()


class _FakeGrootServer:
    """A small REP server whose replies are all serialized with ``groot.pack``."""

    def __init__(self, responder, expected_requests: int) -> None:
        self.responder = responder
        self.expected_requests = expected_requests
        self.requests: list[dict] = []
        self.errors: list[BaseException] = []
        self.stop = threading.Event()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 50)
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        try:
            while len(self.requests) < self.expected_requests and not self.stop.is_set():
                try:
                    payload = self.socket.recv()
                except zmq.error.Again:
                    continue
                request = groot.unpack(payload)
                self.requests.append(request)
                reply = self.responder(request)
                if reply is _NO_REPLY:
                    self.stop.wait()
                    return
                self.socket.send(groot.pack(reply))
        except BaseException as error:
            self.errors.append(error)
        finally:
            self.socket.close(linger=1000)
            self.context.term()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("fake GR00T server thread did not stop")


@unittest.skipUnless(HAS_SIM_DEPENDENCIES, "requires importable numpy and mujoco")
class SimulationIntegrationTest(unittest.TestCase):
    @staticmethod
    def _config():
        return load_config(SIM_CONFIG)

    @staticmethod
    def _action_reply(arm_shape=(1, 16, 6)):
        return [
            {
                "arm_joint_delta": np.zeros(arm_shape, dtype=np.float32),
                "gripper_command": np.ones((1, 16, 1), dtype=np.float32),
            },
            {},
        ]

    @staticmethod
    def _standard_responder(action_reply):
        def respond(request):
            endpoint = request.get("endpoint")
            if endpoint in ("get_modality_config", "reset"):
                return {}
            if endpoint == "get_action":
                return action_reply
            raise AssertionError(f"unexpected endpoint {endpoint!r}")

        return respond

    def _assert_server_clean(self, server: _FakeGrootServer) -> None:
        self.assertFalse(server.thread.is_alive(), "fake GR00T server did not stop")
        if server.errors:
            raise server.errors[0]

    def _run_scripted(self, seed: int):
        config = self._config()
        env = SimEnv(config)
        limits = ActionLimits.from_config(config)
        policy = ScriptedPickPolicy(env, config["action_horizon"], max(limits.high[:6]))
        try:
            summary = sim_run.run_episode(
                env,
                policy,
                limits,
                Watchdog(1.0, 1.0 / env.control_hz),
                seed=seed,
                steps=300,
                execute_k=8,
                recorder=None,
                render_obs=False,
                hold_steps=0,
            )
            final_qpos = env.arm_qpos()
        finally:
            policy.close()
            env.close()
        return summary, final_qpos

    def test_scripted_policy_lifts_cube_for_seeds_zero_through_four(self) -> None:
        for seed in range(5):
            with self.subTest(seed=seed):
                summary, _ = self._run_scripted(seed)
                self.assertTrue(summary["success"], summary)
                self.assertLessEqual(summary["success_step"], 300)

    def test_scripted_policy_is_deterministic_for_one_seed(self) -> None:
        first_summary, first_qpos = self._run_scripted(seed=3)
        second_summary, second_qpos = self._run_scripted(seed=3)
        self.assertTrue(first_summary["success"], first_summary)
        self.assertTrue(second_summary["success"], second_summary)
        np.testing.assert_array_equal(first_qpos, second_qpos)

    def test_groot_policy_round_trip_and_observation_shapes(self) -> None:
        config = self._config()
        server = _FakeGrootServer(
            self._standard_responder(self._action_reply()), expected_requests=3
        )
        policy = groot.GrootPolicy(config, "127.0.0.1", server.port, timeout_ms=1000)
        task = "pick up the red cube"
        observation = {
            "front_camera": np.zeros((224, 224, 3), dtype=np.uint8),
            "arm_joint_positions": np.arange(6, dtype=np.float64),
            "gripper_position": np.array([0.5], dtype=np.float64),
            "task_description": task,
        }
        try:
            policy.reset(task)
            chunk = policy.act(observation)
        finally:
            policy.close()
            server.close()

        self._assert_server_clean(server)
        self.assertEqual(chunk.shape, (16, 7))
        np.testing.assert_array_equal(chunk[:, :6], np.zeros((16, 6)))
        np.testing.assert_array_equal(chunk[:, 6], np.ones(16))

        action_request = next(r for r in server.requests if r["endpoint"] == "get_action")
        wire_observation = action_request["data"]["observation"]
        video = wire_observation["video"]["front_camera"]
        self.assertEqual(video.shape, (1, 1, 224, 224, 3))
        self.assertEqual(video.dtype, np.dtype(np.uint8))
        for name, dimension in (("arm_joint_positions", 6), ("gripper_position", 1)):
            with self.subTest(state=name):
                state = wire_observation["state"][name]
                self.assertEqual(state.shape, (1, 1, dimension))
                self.assertEqual(state.dtype, np.dtype(np.float32))
        language = wire_observation["language"]["task_description"]
        self.assertEqual(language, [[task]])
        self.assertIsInstance(language[0][0], str)

    def test_groot_server_that_never_replies_is_recorded_as_stop_reason(self) -> None:
        """A server that never answers (even on reset) is recorded as a safety stop."""
        config = self._config()
        env = SimEnv(config)
        limits = ActionLimits.from_config(config)
        server = _FakeGrootServer(lambda _request: _NO_REPLY, expected_requests=1)
        policy = groot.GrootPolicy(config, "127.0.0.1", server.port, timeout_ms=200)
        started = time.monotonic()
        try:
            summary = sim_run.run_episode(
                env,
                policy,
                limits,
                Watchdog(1.0, 1.0 / env.control_hz),
                seed=0,
                steps=1,
                execute_k=8,
                recorder=None,
                render_obs=False,
                hold_steps=0,
            )
            elapsed = time.monotonic() - started
        finally:
            policy.close()
            env.close()
            server.close()

        self._assert_server_clean(server)
        self.assertIsNotNone(summary["stop_reason"])
        self.assertIn("did not answer within 200 ms", summary["stop_reason"])
        self.assertGreaterEqual(elapsed, 0.15)
        self.assertLess(elapsed, 5.0)

    def test_wrong_groot_action_shape_is_recorded_as_stop_reason(self) -> None:
        config = self._config()
        env = SimEnv(config)
        limits = ActionLimits.from_config(config)
        server = _FakeGrootServer(
            self._standard_responder(self._action_reply(arm_shape=(1, 16, 5))),
            expected_requests=3,
        )
        policy = groot.GrootPolicy(config, "127.0.0.1", server.port, timeout_ms=1000)
        try:
            summary = sim_run.run_episode(
                env,
                policy,
                limits,
                Watchdog(1.0, 1.0 / env.control_hz),
                seed=0,
                steps=1,
                execute_k=8,
                recorder=None,
                render_obs=False,
                hold_steps=0,
            )
        finally:
            policy.close()
            env.close()
            server.close()

        self._assert_server_clean(server)
        self.assertIsNotNone(summary["stop_reason"])
        self.assertIn("shape (1, 16, 5), expected (1, T, 6)", summary["stop_reason"])

    def test_groot_policy_refuses_placeholder_before_opening_socket(self) -> None:
        example_config = load_config(ROOT / "config" / "embodiment.example.json")
        with mock.patch.object(groot.zmq, "Context") as context_factory:
            with self.assertRaisesRegex(SafetyStop, "placeholder"):
                groot.GrootPolicy(example_config, "127.0.0.1", 1, timeout_ms=200)
        context_factory.assert_not_called()

    def test_main_scripted_smoke(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as out_dir:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = sim_run.main(["--steps", "300", "--no-record", "--out", out_dir])
        self.assertEqual(result, 0, stdout.getvalue() + stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
