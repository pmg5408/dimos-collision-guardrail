# Copyright 2025-2026 Dimensional Inc.
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

from __future__ import annotations

from collections.abc import Callable, Iterator
import queue
import threading
import time
from typing import Any, TypeVar

import pytest

from dimos.control.safety.guardrail_hysteresis import RiskLevel
from dimos.control.safety.guardrail_policy import (
    OpticalFlowMagnitudeGuardrailPolicy,
    RiskAssessment,
    RiskResult,
    RiskUnavailable,
)
from dimos.control.safety.guardrail_types import GuardrailDecision, GuardrailState
from dimos.control.safety.rgb_collision_guardrail import (
    RGBCollisionGuardrail,
    _InputSnapshot,
)
from dimos.control.safety.test_utils import (
    FakeTransport,
    SequencePolicy,
    _assessment,
    _cmd,
    _textured_gray_image,
)
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image

T = TypeVar("T")


class RaisingPolicy:
    def evaluate(self, previous_image: Image, current_image: Image) -> RiskResult:
        raise RuntimeError("synthetic policy failure")

    def reset(self) -> None:
        pass


class CountingPassPolicy:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def evaluate(self, previous_image: Image, current_image: Image) -> RiskResult:
        with self._lock:
            self._call_count += 1

        return RiskAssessment(level=RiskLevel.CLEAR, score=0.0)

    def reset(self) -> None:
        with self._lock:
            self._call_count = 0


@pytest.fixture
def module() -> Iterator[RGBCollisionGuardrail]:
    guardrail = RGBCollisionGuardrail(
        decision_hz=50.0,
        command_timeout_s=0.05,
        image_timeout_s=0.05,
    )
    yield guardrail
    guardrail._close_module()


def _wait_for_output(
    outputs: queue.Queue[Twist],
    predicate: Callable[[Twist], bool],
    *,
    timeout_s: float = 0.5,
) -> Twist:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            candidate = outputs.get(timeout=max(remaining, 0.01))
        except queue.Empty:
            continue
        if predicate(candidate):
            return candidate
    raise AssertionError("Timed out waiting for matching guardrail output")


def _wait_for_decision(
    guardrail: RGBCollisionGuardrail,
    predicate: Callable[[GuardrailDecision], bool],
    *,
    timeout_s: float = 0.5,
) -> GuardrailDecision:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with guardrail._condition:
            decision = guardrail._runtime_state.last_decision
            if decision is not None and predicate(decision):
                return decision
        time.sleep(0.01)

    raise AssertionError("Timed out waiting for matching guardrail decision")


def _start_threaded_guardrail(
    policy: Any,
    **config_overrides: Any,
) -> tuple[RGBCollisionGuardrail, FakeTransport[Image], FakeTransport[Twist], queue.Queue[Twist]]:
    config: dict[str, Any] = {
        "decision_hz": 50.0,
        "command_timeout_s": 0.3,
        "image_timeout_s": 0.3,
    }
    config.update(config_overrides)

    guardrail = RGBCollisionGuardrail(policy_override=policy, **config)
    image_transport: FakeTransport[Image] = FakeTransport()
    cmd_transport: FakeTransport[Twist] = FakeTransport()
    outputs: queue.Queue[Twist] = queue.Queue()

    guardrail.color_image.transport = image_transport
    guardrail.incoming_cmd_vel.transport = cmd_transport
    guardrail.safe_cmd_vel.subscribe(outputs.put)
    guardrail.start()

    return guardrail, image_transport, cmd_transport, outputs


def _snapshot(
    *,
    previous_image: Image | None = None,
    current_image: Image | None = None,
    incoming_cmd_vel: Twist | None = None,
    image_fresh: bool = True,
    cmd_fresh: bool = True,
    frame_pair_fresh: bool = True,
    has_new_frame_pair: bool = True,
    last_decision: GuardrailDecision | None = None,
) -> _InputSnapshot:
    return _InputSnapshot(
        previous_image=previous_image,
        current_image=current_image,
        incoming_cmd_vel=incoming_cmd_vel,
        image_fresh=image_fresh,
        cmd_fresh=cmd_fresh,
        frame_pair_fresh=frame_pair_fresh,
        has_new_frame_pair=has_new_frame_pair,
        last_decision=last_decision,
    )


def _valid_snapshot(**overrides: Any) -> _InputSnapshot:
    defaults: dict[str, Any] = {
        "previous_image": _textured_gray_image(),
        "current_image": _textured_gray_image(shift_x=2),
        "incoming_cmd_vel": _cmd(0.4),
    }
    defaults.update(overrides)
    return _snapshot(**defaults)


@pytest.mark.parametrize(
    ("snapshot", "expected_state", "expected_reason"),
    [
        pytest.param(
            _snapshot(),
            GuardrailState.INIT,
            "no_command_received",
            id="no_command",
        ),
        pytest.param(
            _snapshot(incoming_cmd_vel=_cmd()),
            GuardrailState.INIT,
            "waiting_for_first_image",
            id="no_image",
        ),
        pytest.param(
            _snapshot(incoming_cmd_vel=_cmd(), current_image=_textured_gray_image()),
            GuardrailState.INIT,
            "waiting_for_frame_pair",
            id="no_frame_pair",
        ),
        pytest.param(
            _valid_snapshot(image_fresh=False),
            GuardrailState.SENSOR_DEGRADED,
            "image_stale",
            id="stale_image",
        ),
        pytest.param(
            _valid_snapshot(frame_pair_fresh=False),
            GuardrailState.SENSOR_DEGRADED,
            "frame_pair_stale",
            id="frame_pair_gap",
        ),
        pytest.param(
            _valid_snapshot(current_image=_textured_gray_image(height=90)),
            GuardrailState.SENSOR_DEGRADED,
            "frame_shape_mismatch",
            id="shape_mismatch",
        ),
    ],
)
def test_invalid_inputs_fail_closed_with_a_named_reason(
    module: RGBCollisionGuardrail,
    snapshot: _InputSnapshot,
    expected_state: GuardrailState,
    expected_reason: str,
) -> None:
    decision = module._input_failure_decision(snapshot)

    assert decision is not None
    assert decision.state == expected_state
    assert decision.reason == expected_reason
    assert decision.cmd_vel == Twist.zero()


def test_valid_inputs_are_not_rejected(module: RGBCollisionGuardrail) -> None:
    assert module._input_failure_decision(_valid_snapshot()) is None


def test_config_builds_the_default_detector() -> None:
    guardrail = RGBCollisionGuardrail()

    try:
        assert isinstance(guardrail._policy, OpticalFlowMagnitudeGuardrailPolicy)
    finally:
        guardrail._close_module()


def test_policy_override_replaces_the_configured_detector() -> None:
    policy = CountingPassPolicy()
    guardrail = RGBCollisionGuardrail(policy_override=policy)

    try:
        assert guardrail._policy is policy
    finally:
        guardrail._close_module()


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(_cmd(0.03, angular_z=0.35), id="below_forward_deadband"),
        pytest.param(_cmd(-0.2, angular_z=0.4), id="reverse_motion"),
        pytest.param(Twist(linear=[0.0, 0.0, 0.0], angular=[0.0, 0.0, 0.6]), id="pure_yaw"),
        pytest.param(None, id="no_command"),
    ],
)
def test_forward_motion_not_commanded(module: RGBCollisionGuardrail, cmd: Twist | None) -> None:
    assert module._forward_motion_commanded(cmd) is False


def test_forward_motion_commanded_above_deadband(module: RGBCollisionGuardrail) -> None:
    assert module._forward_motion_commanded(_cmd(0.4)) is True


def test_pass_state_forwards_the_command_untouched(module: RGBCollisionGuardrail) -> None:
    incoming = _cmd(0.5, angular_z=0.42)

    assert module._command_for_state(GuardrailState.PASS, _valid_snapshot(incoming_cmd_vel=incoming)) == incoming


def test_clamp_state_limits_forward_speed_and_preserves_angular(
    module: RGBCollisionGuardrail,
) -> None:
    snapshot = _valid_snapshot(incoming_cmd_vel=_cmd(0.5, angular_z=0.42))

    cmd_vel = module._command_for_state(GuardrailState.CLAMP, snapshot)

    assert cmd_vel.linear.x == pytest.approx(module.config.clamp_forward_speed_mps)
    assert cmd_vel.angular.z == pytest.approx(0.42)


def test_clamp_state_does_not_raise_a_slower_forward_speed(
    module: RGBCollisionGuardrail,
) -> None:
    slower_than_clamp = module.config.clamp_forward_speed_mps / 2.0
    snapshot = _valid_snapshot(incoming_cmd_vel=_cmd(slower_than_clamp))

    cmd_vel = module._command_for_state(GuardrailState.CLAMP, snapshot)

    assert cmd_vel.linear.x == pytest.approx(slower_than_clamp)


def test_stop_state_zeroes_forward_speed_and_preserves_angular(
    module: RGBCollisionGuardrail,
) -> None:
    snapshot = _valid_snapshot(incoming_cmd_vel=_cmd(0.5, angular_z=0.42))

    cmd_vel = module._command_for_state(GuardrailState.STOP_LATCHED, snapshot)

    assert cmd_vel.linear.x == pytest.approx(0.0)
    assert cmd_vel.angular.z == pytest.approx(0.42)


@pytest.mark.parametrize(
    "state",
    [GuardrailState.INIT, GuardrailState.SENSOR_DEGRADED],
)
def test_non_gating_states_emit_zero(module: RGBCollisionGuardrail, state: GuardrailState) -> None:
    assert module._command_for_state(state, _valid_snapshot()) == Twist.zero()


def test_stale_command_emits_zero_in_every_state(module: RGBCollisionGuardrail) -> None:
    snapshot = _valid_snapshot(cmd_fresh=False)

    for state in GuardrailState:
        assert module._command_for_state(state, snapshot) == Twist.zero()


def test_unavailable_assessment_degrades_with_the_reported_reason(
    module: RGBCollisionGuardrail,
) -> None:
    decision = module._decision_from_risk_assessment(
        RiskUnavailable("current_roi_occluded"),
        _valid_snapshot(),
    )

    assert decision.state == GuardrailState.SENSOR_DEGRADED
    assert decision.reason == "current_roi_occluded"
    assert decision.cmd_vel == Twist.zero()


def test_frozen_stream_degrades_after_repeated_identical_frames(
    module: RGBCollisionGuardrail,
) -> None:
    frame = _textured_gray_image()
    snapshot = _valid_snapshot(previous_image=frame, current_image=frame)

    first = module._frozen_stream_decision(snapshot)
    second = module._frozen_stream_decision(snapshot)
    third = module._frozen_stream_decision(snapshot)

    assert first is None
    assert second is None
    assert third is not None
    assert third.state == GuardrailState.SENSOR_DEGRADED
    assert third.reason == "static_scene"
    assert third.cmd_vel == Twist.zero()


def test_frozen_stream_counter_resets_when_a_frame_changes(
    module: RGBCollisionGuardrail,
) -> None:
    frame = _textured_gray_image()
    frozen = _valid_snapshot(previous_image=frame, current_image=frame)

    module._frozen_stream_decision(frozen)
    module._frozen_stream_decision(frozen)
    module._frozen_stream_decision(_valid_snapshot())

    assert module._static_frame_hits == 0


def test_policy_exception_fail_closes_to_zero() -> None:
    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(
        RaisingPolicy(),
    )

    try:
        cmd_transport.publish(_cmd(0.4, angular_z=0.3))
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))

        observed = _wait_for_output(outputs, lambda twist: twist == Twist.zero())
        assert observed == Twist.zero()

        decision = _wait_for_decision(
            guardrail,
            lambda d: d.state == GuardrailState.SENSOR_DEGRADED
            and d.reason == "policy_evaluation_failed",
            timeout_s=0.5,
        )

        assert decision.state == GuardrailState.SENSOR_DEGRADED
        assert decision.reason == "policy_evaluation_failed"

        with guardrail._condition:
            assert guardrail._runtime_state.state == GuardrailState.SENSOR_DEGRADED
    finally:
        guardrail.stop()


def test_frozen_stream_counter_advances_per_frame_pair_not_per_tick() -> None:
    policy = CountingPassPolicy()
    guardrail, image_transport, cmd_transport, _outputs = _start_threaded_guardrail(
        policy,
        static_scene_frame_count=3,
        image_timeout_s=5.0,
    )

    try:
        cmd_transport.publish(_cmd())
        frame = _textured_gray_image()
        image_transport.publish(frame)
        image_transport.publish(frame)

        time.sleep(0.2)

        assert guardrail._static_frame_hits == 1
        with guardrail._condition:
            assert guardrail._runtime_state.state != GuardrailState.SENSOR_DEGRADED
    finally:
        guardrail.stop()


def test_stop_keeps_the_thread_handle_when_the_join_times_out(mocker) -> None:
    mocker.patch(
        "dimos.control.safety.rgb_collision_guardrail._THREAD_JOIN_TIMEOUT_S",
        0.05,
    )
    release = threading.Event()

    class BlockingPolicy:
        def evaluate(self, previous_image: Image, current_image: Image) -> RiskResult:
            release.wait()
            return RiskAssessment(level=RiskLevel.CLEAR, score=0.0)

        def reset(self) -> None:
            pass

    guardrail, image_transport, cmd_transport, _outputs = _start_threaded_guardrail(
        BlockingPolicy(),
    )

    try:
        cmd_transport.publish(_cmd(0.4))
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))
        time.sleep(0.1)

        guardrail.stop()

        # The worker outlived the join, so a second start() would put two threads on
        # one output stream. It has to refuse.
        assert guardrail._thread is not None
        with pytest.raises(RuntimeError):
            guardrail.start()
    finally:
        release.set()
        guardrail.stop()


def test_command_arriving_during_evaluation_is_not_delayed_to_the_next_tick() -> None:
    class SlowPolicy:
        def evaluate(self, previous_image: Image, current_image: Image) -> RiskResult:
            time.sleep(0.1)
            return RiskAssessment(level=RiskLevel.CLEAR, score=0.0)

        def reset(self) -> None:
            pass

    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(
        SlowPolicy(),
        decision_hz=2.0,
        command_timeout_s=5.0,
        image_timeout_s=5.0,
    )

    try:
        first = _cmd(0.4, angular_z=0.1)
        cmd_transport.publish(first)
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))

        # Land a command while the worker is inside evaluate(). notify() cannot help
        # here -- nobody is waiting to be notified -- so without the wake flag this
        # command would sit out the full 500ms tick.
        time.sleep(0.05)
        second = _cmd(0.4, angular_z=-0.3)
        cmd_transport.publish(second)
        published_at = time.monotonic()

        _wait_for_output(outputs, lambda twist: twist == second, timeout_s=0.4)
        assert time.monotonic() - published_at < 0.4
    finally:
        guardrail.stop()


def test_output_continues_at_the_tick_rate_without_new_input() -> None:
    policy = CountingPassPolicy()
    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(
        policy,
        decision_hz=50.0,
        command_timeout_s=5.0,
        image_timeout_s=5.0,
    )

    try:
        upstream = _cmd(0.4)
        cmd_transport.publish(upstream)
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))

        _wait_for_output(outputs, lambda twist: twist == upstream)

        # Nothing new is published from here on; the gate must keep emitting so
        # silence stays a reliable signal that it has died.
        for _ in range(3):
            assert _wait_for_output(outputs, lambda twist: twist == upstream) == upstream
    finally:
        guardrail.stop()


def test_held_state_tracks_newer_commands_without_a_new_frame_pair() -> None:
    policy = SequencePolicy([_assessment(RiskLevel.CAUTION)])
    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(
        policy,
        hysteresis={"caution_frame_count": 1},
        command_timeout_s=5.0,
        image_timeout_s=5.0,
    )

    try:
        cmd_transport.publish(_cmd(0.4, angular_z=0.2))
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))

        clamped = Twist(linear=[0.1, 0.0, 0.0], angular=[0.0, 0.0, 0.2])
        _wait_for_output(outputs, lambda twist: twist == clamped)

        # No new frames arrive, so the state is held -- but the gate must apply it
        # to the newer command rather than replaying the old output.
        cmd_transport.publish(_cmd(0.4, angular_z=-0.45))

        reclamped = Twist(linear=[0.1, 0.0, 0.0], angular=[0.0, 0.0, -0.45])
        assert _wait_for_output(outputs, lambda twist: twist == reclamped) == reclamped
    finally:
        guardrail.stop()


def test_pass_publishes_latest_upstream_command() -> None:
    upstream_first = _cmd(0.3, angular_z=0.1)
    upstream_second = _cmd(0.45, angular_z=0.35)
    policy = SequencePolicy([_assessment(RiskLevel.CLEAR)])
    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(policy)

    try:
        cmd_transport.publish(upstream_first)
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))

        first_output = _wait_for_output(outputs, lambda twist: twist == upstream_first)
        assert first_output == upstream_first

        cmd_transport.publish(upstream_second)
        second_output = _wait_for_output(outputs, lambda twist: twist == upstream_second)
        assert second_output == upstream_second
    finally:
        guardrail.stop()


@pytest.mark.parametrize(
    ("level", "frame_counts", "guarded_cmd"),
    [
        pytest.param(
            RiskLevel.CAUTION,
            {"caution_frame_count": 1},
            Twist(linear=[0.1, 0.0, 0.0], angular=[0.0, 0.0, 0.4]),
            id="caution_clamps_forward_speed",
        ),
        pytest.param(
            RiskLevel.STOP,
            {"stop_frame_count": 2},
            Twist(linear=[0.0, 0.0, 0.0], angular=[0.0, 0.0, 0.4]),
            id="repeated_stop_zeroes_forward_speed",
        ),
    ],
)
def test_non_pass_states_publish_guarded_output(
    level: RiskLevel,
    frame_counts: dict[str, float],
    guarded_cmd: Twist,
) -> None:
    upstream_cmd = _cmd(0.35, angular_z=0.4)
    policy = SequencePolicy([_assessment(level)])
    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(
        policy,
        hysteresis=frame_counts,
    )

    try:
        cmd_transport.publish(upstream_cmd)
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))
        image_transport.publish(_textured_gray_image(shift_x=4))

        published = _wait_for_output(outputs, lambda twist: twist == guarded_cmd)
        assert published == guarded_cmd
    finally:
        guardrail.stop()


def test_non_pass_heartbeat_republishes_guarded_output() -> None:
    guarded_cmd = Twist(linear=[0.1, 0.0, 0.0], angular=[0.0, 0.0, 0.5])
    policy = SequencePolicy([_assessment(RiskLevel.CAUTION)])
    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(
        policy,
        hysteresis={"caution_frame_count": 1},
    )

    try:
        cmd_transport.publish(_cmd(0.4, angular_z=0.5))
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))

        first = _wait_for_output(outputs, lambda twist: twist == guarded_cmd)
        second = _wait_for_output(outputs, lambda twist: twist == guarded_cmd)

        assert first == guarded_cmd
        assert second == guarded_cmd
    finally:
        guardrail.stop()


def test_non_pass_decision_can_publish_without_new_command() -> None:
    upstream_cmd = _cmd(0.4, angular_z=0.3)
    stop_cmd = Twist(linear=[0.0, 0.0, 0.0], angular=[0.0, 0.0, 0.3])
    policy = SequencePolicy(
        [
            _assessment(RiskLevel.CLEAR),
            _assessment(RiskLevel.STOP),
        ]
    )
    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(
        policy,
        hysteresis={"stop_frame_count": 1},
    )

    try:
        cmd_transport.publish(upstream_cmd)
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))

        first_output = _wait_for_output(outputs, lambda twist: twist == upstream_cmd)
        assert first_output == upstream_cmd

        image_transport.publish(_textured_gray_image(shift_x=4))

        autonomous_stop = _wait_for_output(outputs, lambda twist: twist == stop_cmd)
        assert autonomous_stop == stop_cmd
    finally:
        guardrail.stop()


def test_fast_upstream_commands_reuse_last_risk_decision() -> None:
    policy = CountingPassPolicy()
    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(
        policy,
        decision_hz=2.0,
        command_timeout_s=1.0,
        image_timeout_s=1.0,
    )

    first_cmd = _cmd(0.20, angular_z=0.10)
    second_cmd = _cmd(0.32, angular_z=0.20)
    third_cmd = _cmd(0.44, angular_z=0.30)

    try:
        cmd_transport.publish(first_cmd)
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))

        first_output = _wait_for_output(outputs, lambda twist: twist == first_cmd, timeout_s=0.6)
        assert first_output == first_cmd
        assert policy.call_count == 1

        cmd_transport.publish(second_cmd)
        second_output = _wait_for_output(
            outputs,
            lambda twist: twist == second_cmd,
            timeout_s=0.2,
        )
        assert second_output == second_cmd

        cmd_transport.publish(third_cmd)
        third_output = _wait_for_output(
            outputs,
            lambda twist: twist == third_cmd,
            timeout_s=0.2,
        )
        assert third_output == third_cmd

        assert policy.call_count == 1
        assert policy.call_count < 3
    finally:
        guardrail.stop()


def test_stop_publishes_zero_as_final_command() -> None:
    policy = CountingPassPolicy()
    guardrail, image_transport, cmd_transport, outputs = _start_threaded_guardrail(policy)

    forward_cmd = _cmd(0.4, angular_z=0.3)

    try:
        cmd_transport.publish(forward_cmd)
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))

        _wait_for_output(outputs, lambda twist: twist == forward_cmd)
        for _ in range(5):
            cmd_transport.publish(forward_cmd)
    finally:
        guardrail.stop()

    final_output: Twist | None = None
    while True:
        try:
            final_output = outputs.get_nowait()
        except queue.Empty:
            break

    assert final_output == Twist.zero()


def test_double_start_raises() -> None:
    policy = CountingPassPolicy()
    guardrail, _image_transport, _cmd_transport, _outputs = _start_threaded_guardrail(policy)

    try:
        with pytest.raises(RuntimeError):
            guardrail.start()
    finally:
        guardrail.stop()


def test_restart_resets_runtime_state() -> None:
    policy = SequencePolicy([_assessment(RiskLevel.STOP)])
    guardrail, image_transport, cmd_transport, _outputs = _start_threaded_guardrail(
        policy,
        hysteresis={"stop_frame_count": 1},
    )

    try:
        cmd_transport.publish(_cmd(0.4, angular_z=0.3))
        image_transport.publish(_textured_gray_image())
        image_transport.publish(_textured_gray_image(shift_x=2))
        _wait_for_decision(guardrail, lambda d: d.state == GuardrailState.STOP_LATCHED)
    finally:
        guardrail.stop()

    with guardrail._condition:
        assert guardrail._runtime_state.state == GuardrailState.STOP_LATCHED

    guardrail.start()
    try:
        with guardrail._condition:
            assert guardrail._runtime_state.state == GuardrailState.INIT
    finally:
        guardrail.stop()
