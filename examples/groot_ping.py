#!/usr/bin/env python3
import argparse
import os
import sys

from embodiment_config import ConfigError, load_config, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check an NVIDIA Isaac GR00T N1.7 PolicyServer endpoint."
    )
    parser.add_argument("--host", default=os.getenv("GROOT_HOST", "host.docker.internal"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GROOT_PORT", "5555")))
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=int(os.getenv("GROOT_TIMEOUT_MS", "5000")),
    )
    parser.add_argument("--api-token", default=os.getenv("GROOT_API_TOKEN") or None)
    parser.add_argument(
        "--config",
        default=os.getenv("GROOT_EMBODIMENT_CONFIG") or None,
        help="Embodiment JSON to validate before connecting; no observations are sent.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_ping_request(api_token: str | None) -> dict[str, str]:
    request = {"endpoint": "ping"}
    if api_token:
        request["api_token"] = api_token
    return request


def main() -> int:
    args = parse_args()
    if args.config:
        try:
            config = load_config(args.config)
        except ConfigError as error:
            print(f"Invalid embodiment config: {error}", file=sys.stderr)
            return 2
        for line in summarize(config):
            print(line)

    address = f"tcp://{args.host}:{args.port}"
    request = build_ping_request(args.api_token)

    if args.dry_run:
        print(f"GR00T PolicyServer target: {address}")
        print("Request: ping")
        return 0

    import msgpack
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    socket.connect(address)

    try:
        socket.send(msgpack.packb(request, use_bin_type=True))
        response = msgpack.unpackb(socket.recv(), raw=False)
    except zmq.error.Again:
        print(f"GR00T PolicyServer did not respond at {address}", file=sys.stderr)
        return 1
    finally:
        socket.close()
        context.term()

    if not isinstance(response, dict) or response.get("status") != "ok":
        print(f"Unexpected GR00T response: {response!r}", file=sys.stderr)
        return 1

    print(f"GR00T PolicyServer is healthy at {address}")
    print(response.get("message", "Server is running"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
