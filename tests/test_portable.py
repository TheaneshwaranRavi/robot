"""Smoke tests for the portable client. They need no GPU, GR00T server, or robot."""
import copy
import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import groot_ping  # noqa: E402
from embodiment_config import ConfigError, load_config, validate_config  # noqa: E402

EXAMPLE_CONFIG = ROOT / "config" / "embodiment.example.json"


def run_ping(*argv: str) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", ["groot_ping.py", *argv]):
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = groot_ping.main()
    return code, stdout.getvalue(), stderr.getvalue()


class EmbodimentConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(EXAMPLE_CONFIG.read_text())

    def test_example_config_is_valid(self) -> None:
        load_config(EXAMPLE_CONFIG)

    def test_rejects_missing_action_limits(self) -> None:
        del self.config["action"][0]["high"]
        with self.assertRaisesRegex(ConfigError, "low and high"):
            validate_config(self.config)

    def test_rejects_inverted_action_limits(self) -> None:
        self.config["action"][0]["low"] = 1.0
        with self.assertRaisesRegex(ConfigError, "less than high"):
            validate_config(self.config)

    def test_rejects_bad_shape(self) -> None:
        self.config["observation"]["video"][0]["height"] = 0
        with self.assertRaisesRegex(ConfigError, "height"):
            validate_config(self.config)

    def test_rejects_duplicate_names(self) -> None:
        self.config["action"].append(copy.deepcopy(self.config["action"][0]))
        with self.assertRaisesRegex(ConfigError, "duplicate"):
            validate_config(self.config)


class PingTest(unittest.TestCase):
    def test_ping_request_includes_token_only_when_set(self) -> None:
        self.assertEqual(groot_ping.build_ping_request(None), {"endpoint": "ping"})
        self.assertEqual(
            groot_ping.build_ping_request("secret"), {"endpoint": "ping", "api_token": "secret"}
        )

    def test_dry_run_with_config(self) -> None:
        code, out, _ = run_ping("--dry-run", "--config", str(EXAMPLE_CONFIG))
        self.assertEqual(code, 0)
        self.assertIn("Action dim: 7", out)
        self.assertIn("placeholder", out)

    def test_invalid_config_fails_before_connecting(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            handle.write("{}")
            handle.flush()
            code, _, err = run_ping("--dry-run", "--config", handle.name)
        self.assertEqual(code, 2)
        self.assertIn("Invalid embodiment config", err)

    def test_live_ping_against_fake_server(self) -> None:
        import msgpack
        import zmq

        context = zmq.Context()
        server = context.socket(zmq.REP)
        port = server.bind_to_random_port("tcp://127.0.0.1")
        received = []

        def serve_once() -> None:
            received.append(msgpack.unpackb(server.recv(), raw=False))
            server.send(msgpack.packb({"status": "ok", "message": "fake"}, use_bin_type=True))

        thread = threading.Thread(target=serve_once)
        thread.start()
        try:
            code, out, _ = run_ping("--host", "127.0.0.1", "--port", str(port), "--timeout-ms", "2000")
        finally:
            thread.join(timeout=5)
            server.close()
            context.term()

        self.assertEqual(code, 0)
        self.assertEqual(received, [{"endpoint": "ping"}])
        self.assertIn("healthy", out)

    def test_unreachable_server_times_out(self) -> None:
        code, _, err = run_ping("--host", "127.0.0.1", "--port", "1", "--timeout-ms", "200")
        self.assertEqual(code, 1)
        self.assertIn("did not respond", err)


if __name__ == "__main__":
    unittest.main()
