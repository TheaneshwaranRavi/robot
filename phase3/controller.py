"""Pure, sim-time state machine for inspection and sorting."""
from __future__ import annotations

import math
from enum import Enum
from numbers import Real
from typing import Any, Iterable, Mapping

from .config import validate_config
from .contracts import CameraFrame, CellEvent, InspectionResult, PartObservation


class State(str, Enum):
    BOOT = "BOOT"
    HOME = "HOME"
    WAIT_PART = "WAIT_PART"
    CAPTURE = "CAPTURE"
    LOCALISE = "LOCALISE"
    INSPECT = "INSPECT"
    DECIDE = "DECIDE"
    PICK = "PICK"
    VERIFY_GRASP = "VERIFY_GRASP"
    PLACE_GOOD = "PLACE_GOOD"
    PLACE_REJECT = "PLACE_REJECT"
    VERIFY_DROP = "VERIFY_DROP"
    RECORD = "RECORD"
    RECOVER = "RECOVER"
    SAFE_STOP = "SAFE_STOP"


Message = tuple[str, Any]


class Controller:
    """Advance a deterministic cell mission from typed input messages."""

    MAX_RETRIES = 1

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = validate_config(config)
        self.timeouts = self.config["timeouts_s"]
        self.bins = self.config["bins"]
        self.state = State.BOOT
        self.state_history = [State.BOOT]
        self.good_parts = 0
        self.rejected_parts = 0
        self.failed_parts = 0
        self.stop_reason: str | None = None
        self.part_id: str | None = None
        self.cycle = 0
        self.target_bin: str | None = None
        self._decision: str | None = None
        self._inspection_error: str | None = None
        self._frame: CameraFrame | None = None
        self._observation: PartObservation | None = None
        self._inspection: InspectionResult | None = None
        self._retry_count = 0
        self._retry_state: State | None = None
        self._state_entered_t = 0.0
        self._last_t: float | None = None
        self._failed_counted = False

    @property
    def good_count(self) -> int:
        return self.good_parts

    @property
    def rejected_count(self) -> int:
        return self.rejected_parts

    @property
    def failed_count(self) -> int:
        return self.failed_parts

    @property
    def attempt(self) -> int:
        return self._retry_count + 1

    @property
    def counts(self) -> dict[str, int]:
        return {"good": self.good_parts, "rejected": self.rejected_parts, "failed": self.failed_parts}

    def update(self, t: float, inputs: Iterable[Message | CellEvent] = ()) -> list[Message]:
        """Consume inputs and return commands/events without I/O or wall-clock access."""
        t = self._time(t)
        first = self._last_t is None
        if not first and t < self._last_t:
            raise ValueError("sim time cannot move backwards")
        self._last_t = t
        out: list[Message] = []
        if first:
            self._state_entered_t = t
            self._run_automatic(t, out)
        elif self._timed_out(t, out):
            self._run_automatic(t, out)
            return out

        for topic, payload in self._normalise_inputs(inputs):
            self._handle(topic, payload, t, out)
            if self.state is State.SAFE_STOP:
                break
        self._run_automatic(t, out)
        return out

    @staticmethod
    def _time(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("sim time must be a finite non-negative number")
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("sim time must be a finite non-negative number")
        return result

    @staticmethod
    def _normalise_inputs(inputs: Iterable[Message | CellEvent]) -> list[Message]:
        result: list[Message] = []
        for item in inputs:
            if isinstance(item, CellEvent):
                result.append((item.topic, item.payload))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                topic, payload = item
                if not isinstance(topic, str) or not topic:
                    raise ValueError("input topic must be a non-empty string")
                result.append((topic, payload))
            elif isinstance(item, Mapping) and "topic" in item:
                topic = item["topic"]
                if not isinstance(topic, str) or not topic:
                    raise ValueError("input topic must be a non-empty string")
                result.append((topic, item.get("payload", {})))
            else:
                raise ValueError("each input must be (topic, payload), a topic mapping, or CellEvent")
        return result

    def _transition(self, state: State, t: float, out: list[Message], reason: str = "") -> None:
        previous = self.state
        self.state = state
        self.state_history.append(state)
        self._state_entered_t = t
        out.append(("cell/state", {
            "t": t,
            "from": previous.value,
            "state": state.value,
            "part_id": self.part_id,
            "cycle": self.cycle,
            "attempt": self.attempt,
            "target_bin": self.target_bin,
            "reason": reason,
            "counts": self.counts,
        }))
        self._request(state, t, out, reason)

    def _request(self, state: State, t: float, out: list[Message], reason: str) -> None:
        common = {"t": t, "part_id": self.part_id, "cycle": self.cycle, "attempt": self.attempt}
        if state is State.HOME:
            out.append(("robot/home", {**common, "command_id": f"cycle-{self.cycle}-home"}))
        elif state is State.CAPTURE:
            out.append(("camera/capture", common))
        elif state is State.LOCALISE:
            out.append(("localiser/localise", {**common, "frame": self._frame}))
        elif state is State.INSPECT:
            out.append(("detector/inspect", {**common, "frame": self._frame, "observation": self._observation}))
        elif state is State.PICK:
            out.append(("robot/pick", {
                **common,
                "command_id": f"cycle-{self.cycle}-pick-{self.attempt}",
                "observation": self._observation,
                "target_bin": self.target_bin,
            }))
        elif state is State.VERIFY_GRASP:
            out.append(("robot/verify_grasp", common))
        elif state in (State.PLACE_GOOD, State.PLACE_REJECT):
            action = "place_good" if state is State.PLACE_GOOD else "place_reject"
            out.append((f"robot/{action}", {
                **common,
                "command_id": f"cycle-{self.cycle}-{action}-{self.attempt}",
                "bin": self.target_bin,
            }))
        elif state is State.VERIFY_DROP:
            out.append(("robot/verify_drop", {**common, "bin": self.target_bin}))
        elif state is State.RECOVER:
            out.append(("robot/hold", {
                **common,
                "recovery": True,
                "reason": reason,
                "retry_state": self._retry_state.value if self._retry_state else None,
            }))
        elif state is State.SAFE_STOP:
            out.append(("robot/hold", {**common, "recovery": False, "reason": self.stop_reason}))
            out.append(("cell/safe_stop", {**common, "reason": self.stop_reason, "counts": self.counts}))

    def _run_automatic(self, t: float, out: list[Message]) -> None:
        for _ in range(8):
            if self.state is State.BOOT:
                self._transition(State.HOME, t, out)
            elif self.state is State.DECIDE:
                decision = self._decision or "UNCERTAIN"
                self.target_bin = self.bins["good"] if decision == "GOOD" else self.bins["reject"]
                self._transition(State.PICK, t, out, self._inspection_error or decision)
            elif self.state is State.RECORD:
                if self.target_bin == self.bins["good"]:
                    self.good_parts += 1
                else:
                    self.rejected_parts += 1
                out.append(("record/part", {
                    "t": t,
                    "part_id": self.part_id,
                    "cycle": self.cycle,
                    "decision": self._decision,
                    "target_bin": self.target_bin,
                    "inspection_error": self._inspection_error,
                    "attempts": self.attempt,
                    "counts": self.counts,
                }))
                self._clear_part()
                self._transition(State.WAIT_PART, t, out)
            else:
                return
        raise RuntimeError("automatic state transition loop did not settle")

    def _handle(self, topic: str, payload: Any, t: float, out: list[Message]) -> None:
        if self.state is State.HOME and topic == "robot/done":
            self._robot_done(payload, "home", State.WAIT_PART, State.HOME, t, out)
        elif self.state is State.WAIT_PART and topic == "part/available":
            data = self._mapping(payload, topic)
            part_id = data.get("part_id")
            if not isinstance(part_id, str) or not part_id.strip():
                raise ValueError("part/available needs a non-empty part_id")
            self._start_part(part_id, t, out)
        elif self.state is State.CAPTURE:
            if topic == "camera/frame":
                frame = payload.get("frame") if isinstance(payload, Mapping) else payload
                if not isinstance(frame, CameraFrame):
                    raise ValueError("camera/frame needs a CameraFrame")
                if not frame.fresh:
                    self._stage_failure("camera returned a stale frame", State.CAPTURE, t, out)
                else:
                    self._frame = frame
                    self._transition(State.LOCALISE, t, out)
            elif topic == "camera/error":
                self._stage_failure(self._reason(payload, "camera error"), State.CAPTURE, t, out)
        elif self.state is State.LOCALISE:
            if topic == "localiser/result":
                observation = payload.get("observation") if isinstance(payload, Mapping) else payload
                if not isinstance(observation, PartObservation):
                    raise ValueError("localiser/result needs a PartObservation")
                if (
                    observation.part_id != self.part_id
                    or observation.source_frame_id != self._frame.frame_id
                    or observation.timestamp_s < self._frame.timestamp_s
                ):
                    self._stage_failure("localiser result does not match the active part/frame", State.CAPTURE, t, out)
                else:
                    self._observation = observation
                    self._transition(State.INSPECT, t, out)
            elif topic == "localiser/error":
                self._stage_failure(self._reason(payload, "localiser error"), State.CAPTURE, t, out)
        elif self.state is State.INSPECT:
            if topic == "detector/result":
                result = payload.get("result") if isinstance(payload, Mapping) else payload
                if not isinstance(result, InspectionResult):
                    self._inspection_failed("detector returned a malformed result", t, out)
                elif (
                    result.part_id != self.part_id
                    or result.source_frame_id != self._frame.frame_id
                    or result.timestamp_s < self._observation.timestamp_s
                ):
                    self._inspection_failed("detector result does not match the active part/frame", t, out)
                else:
                    self._inspection = result
                    self._decision = result.decision
                    self._inspection_error = result.error
                    self._transition(State.DECIDE, t, out, result.error or result.decision)
            elif topic == "detector/error":
                self._inspection_failed(self._reason(payload, "detector error"), t, out)
        elif self.state is State.PICK and topic == "robot/done":
            self._robot_done(payload, "pick", State.VERIFY_GRASP, State.PICK, t, out)
        elif self.state is State.VERIFY_GRASP and topic == "robot/grasp":
            ok, reason = self._outcome(payload, "robot/grasp")
            if ok:
                place = State.PLACE_GOOD if self.target_bin == self.bins["good"] else State.PLACE_REJECT
                self._transition(place, t, out)
            else:
                detail = f"grasp verification failed: {reason}" if reason else "grasp verification failed"
                self._stage_failure(detail, State.PICK, t, out)
        elif self.state in (State.PLACE_GOOD, State.PLACE_REJECT) and topic == "robot/done":
            action = "place_good" if self.state is State.PLACE_GOOD else "place_reject"
            self._robot_done(payload, action, State.VERIFY_DROP, self.state, t, out)
        elif self.state is State.VERIFY_DROP and topic == "robot/drop":
            ok, reason = self._outcome(payload, "robot/drop")
            if ok:
                self._transition(State.RECORD, t, out)
            else:
                retry = State.PLACE_GOOD if self.target_bin == self.bins["good"] else State.PLACE_REJECT
                detail = f"drop verification failed: {reason}" if reason else "drop verification failed"
                self._stage_failure(detail, retry, t, out)
        elif self.state is State.RECOVER and topic == "robot/done":
            data = self._mapping(payload, topic)
            if str(data.get("action", "")).lower() not in ("hold", "recover"):
                return
            ok, reason = self._outcome(data, topic)
            if ok and self._retry_state is not None:
                self._transition(self._retry_state, t, out, "bounded retry")
            else:
                self._safe_stop(reason or "recovery failed", t, out)

    def _robot_done(
        self,
        payload: Any,
        action: str,
        success_state: State,
        retry_state: State,
        t: float,
        out: list[Message],
    ) -> None:
        data = self._mapping(payload, "robot/done")
        if str(data.get("action", "")).lower() != action:
            return
        ok, reason = self._outcome(data, "robot/done")
        if ok:
            self._transition(success_state, t, out)
        else:
            self._stage_failure(reason or f"{action} failed", retry_state, t, out)

    @staticmethod
    def _mapping(payload: Any, topic: str) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{topic} payload must be a mapping")
        return payload

    @classmethod
    def _outcome(cls, payload: Any, topic: str) -> tuple[bool, str]:
        data = cls._mapping(payload, topic)
        ok = data.get("ok")
        if not isinstance(ok, bool):
            raise ValueError(f"{topic} needs a bool ok field")
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError(f"{topic} reason must be a string")
        return ok, reason

    @classmethod
    def _reason(cls, payload: Any, default: str) -> str:
        if isinstance(payload, str):
            return payload.strip() or default
        data = cls._mapping(payload, default)
        reason = data.get("reason", default)
        if not isinstance(reason, str) or not reason.strip():
            return default
        return reason

    def _start_part(self, part_id: str, t: float, out: list[Message]) -> None:
        self.cycle += 1
        self.part_id = part_id
        self.target_bin = None
        self._decision = None
        self._inspection_error = None
        self._frame = None
        self._observation = None
        self._inspection = None
        self._retry_count = 0
        self._retry_state = None
        self._failed_counted = False
        self._transition(State.CAPTURE, t, out)

    def _clear_part(self) -> None:
        self.part_id = None
        self.target_bin = None
        self._decision = None
        self._inspection_error = None
        self._frame = None
        self._observation = None
        self._inspection = None
        self._retry_count = 0
        self._retry_state = None
        self._failed_counted = False

    def _inspection_failed(self, reason: str, t: float, out: list[Message]) -> None:
        self._inspection = None
        self._decision = "UNCERTAIN"
        self._inspection_error = reason
        self._transition(State.DECIDE, t, out, reason)

    def _stage_failure(self, reason: str, retry_state: State, t: float, out: list[Message]) -> None:
        if self._retry_count < self.MAX_RETRIES:
            self._retry_count += 1
            self._retry_state = retry_state
            self._transition(State.RECOVER, t, out, reason)
        else:
            self._safe_stop(reason, t, out)

    def _safe_stop(self, reason: str, t: float, out: list[Message]) -> None:
        if self.state is State.SAFE_STOP:
            return
        self.stop_reason = reason.strip() if isinstance(reason, str) and reason.strip() else "unspecified failure"
        if self.part_id is not None and not self._failed_counted:
            self.failed_parts += 1
            self._failed_counted = True
        self._transition(State.SAFE_STOP, t, out, self.stop_reason)

    def _timed_out(self, t: float, out: list[Message]) -> bool:
        limit = self.timeouts[self.state.value]
        if limit is None or t - self._state_entered_t <= limit:
            return False
        reason = f"{self.state.value} timeout after {limit:.3f}s"
        if self.state is State.INSPECT:
            self._inspection_failed(reason, t, out)
        elif self.state is State.RECOVER:
            self._safe_stop(reason, t, out)
        else:
            retry = {
                State.VERIFY_GRASP: State.PICK,
                State.VERIFY_DROP: State.PLACE_GOOD if self.target_bin == self.bins["good"] else State.PLACE_REJECT,
            }.get(self.state, self.state)
            self._stage_failure(reason, retry, t, out)
        return True


CellController = Controller

__all__ = ["State", "Message", "Controller", "CellController"]
