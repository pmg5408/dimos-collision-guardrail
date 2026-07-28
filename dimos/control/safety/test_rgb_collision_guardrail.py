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
    RiskAssessment,
    RiskResult,
    RiskUnavailable,
)
from dimos.control.safety.guardrail_types import GuardrailDecision, GuardrailState
from dimos.control.safety.rgb_collision_guardrail import RGBCollisionGuardrail
from dimos.control.safety.test_utils import (
    FakeTransport,
    SequencePolicy,
    _assessment,
    _cmd,
    _decision,
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
        guarded_output_publish_hz=50.0,
        risk_evaluation_hz=50.0,
        command_timeout_s=0.05,
        image_timeout_s=0.05,
        risk_timeout_s=0.05,
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
    **config_overrides: float,
) -> tuple[RGBCollisionGuardrail, FakeTransport[Image], FakeTransport[Twist], queue.Queue[Twist]]:
    config: dict[str, float] = {
        "guarded_output_publish_hz": 50.0,
        "risk_evaluation_hz": 50.0,
        "command_timeout_s": 0.3,
        "image_timeout_s": 0.3,
        "risk_timeout_s": 0.3,
    }
    config.update(config_overrides)

    guardrail = RGBCollisionGuardrail(**config)
    image_transport: FakeTransport[Image] = FakeTransport()
    cmd_transport: FakeTransport[Twist] = FakeTransport()
    outputs: queue.Queue[Twist] = queue.Queue()

    guardrail.color_image.transport = image_transport
    guardrail.incoming_cmd_vel.transport = cmd_transport
    guardrail.safe_cmd_vel.subscribe(outputs.put)
    guardrail._policy = policy
    guardrail.start()

    return guardrail, image_transport, cmd_transport, outputs


def test_no_command_returns_init_zero(module: RGBCollisionGuardrail) -> None:
    now = time.monotonic()

    with module._condition:
        decision = module._input_failure_decision_locked(now)

    assert decision is not None
    assert decision.state == GuardrailState.INIT
    assert decision.reason == "no_command_received"
    assert decision.cmd_vel == Twist.zero()


def test_waiting_for_first_image_returns_init_zero(module: RGBCollisionGuardrail) -> None:
    now = time.monotonic()

    with module._condition:
        module._runtime_state.latest_cmd_vel = _cmd()
        module._runtime_state.latest_cmd_time = now
        decision = module._input_failure_decision_locked(now)

    assert decision is not None
    assert decision.state == GuardrailState.INIT
    assert decision.reason == "waiting_for_first_image"
    assert decision.cmd_vel == Twist.zero()


def test_no_frame_pair_returns_init_zero(module: RGBCollisionGuardrail) -> None:
    now = time.monotonic()

    with module._condition:
        module._runtime_state.latest_cmd_vel = _cmd()
        module._runtime_state.latest_cmd_time = now
        module._runtime_state.latest_image = _textured_gray_image()
        module._runtime_state.latest_image_time = now
        decision = module._input_failure_decision_locked(now)

    assert decision is not None
    assert decision.state == GuardrailState.INIT
    assert decision.reason == "waiting_for_frame_pair"
    assert decision.cmd_vel == Twist.zero()


def test_waiting_for_first_risk_evaluation_returns_init_zero(
    module: RGBCollisionGuardrail,
) -> None:
    now = time.monotonic()

    with module._condition:
        module._runtime_state.latest_cmd_vel = _cmd()
        module._runtime_state.latest_cmd_time = now
        module._runtime_state.previous_image = _textured_gray_image()
        module._runtime_state.previous_image_time = now
        module._runtime_state.latest_image = _textured_gray_image(shift_x=2)
        module._runtime_state.latest_image_time = now
        module._runtime_state.last_risk_time = None
        decision = module._risk_staleness_decision_locked(now)

    assert decision is not None
    assert decision.state == GuardrailState.INIT
    assert decision.reason == "waiting_for_first_risk_evaluation"
    assert decision.cmd_vel == Twist.zero()


def test_stale_image_returns_sensor_degraded_zero(module: RGBCollisionGuardrail) -> None:
    now = time.monotonic()
    stale_time = now - 0.2

    with module._condition:
        module._runtime_state.latest_cmd_vel = _cmd()
        module._runtime_state.latest_cmd_time = now
        module._runtime_state.previous_image = _textured_gray_image()
        module._runtime_state.latest_image = _textured_gray_image(shift_x=2)
        module._runtime_state.previous_image_time = stale_time
        module._runtime_state.latest_image_time = stale_time
        decision = module._input_failure_decision_locked(now)

    assert decision is not None
    assert decision.state == GuardrailState.SENSOR_DEGRADED
    assert decision.reason == "image_stale"
    assert decision.cmd_vel == Twist.zero()


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(_cmd(0.03, angular_z=0.35), id="below_forward_deadband"),
        pytest.param(_cmd(-0.2, angular_z=0.4), id="reverse_motion"),
        pytest.param(Twist(linear=[0.0, 0.0, 0.0], angular=[0.0, 0.0, 0.6]), id="pure_yaw"),
    ],
)
def test_forward_motion_not_commanded_below_deadband(
    module: RGBCollisionGuardrail,
    cmd: Twist,
) -> None:
    assert module._forward_motion_commanded(cmd) is False


def test_forward_motion_commanded_above_deadband(module: RGBCollisionGuardrail) -> None:
    assert module._forward_motion_commanded(_cmd(0.4)) is True


def test_clamp_limits_forward_speed_and_preserves_angular(
    module: RGBCollisionGuardrail,
) -> None:
    incoming = _cmd(0.5, angular_z=0.42)

    decision = module._build_clamp_decision(incoming, "risk_clamp", 1.0)

    assert decision.state == GuardrailState.CLAMP
    assert decision.cmd_vel.linear.x == pytest.approx(module.config.clamp_forward_speed_mps)
    assert decision.cmd_vel.angular.z == pytest.approx(0.42)


def test_clamp_does_not_raise_a_slower_forward_speed(module: RGBCollisionGuardrail) -> None:
    slower_than_clamp = module.config.clamp_forward_speed_mps / 2.0

    decision = module._build_clamp_decision(_cmd(slower_than_clamp), "risk_clamp", 1.0)

    assert decision.cmd_vel.linear.x == pytest.approx(slower_than_clamp)


def test_stop_zeroes_forward_speed_and_preserves_angular(
    module: RGBCollisionGuardrail,
) -> None:
    decision = module._build_stop_decision(_cmd(0.5, angular_z=0.42), "risk_stop", 1.0)

    assert decision.state == GuardrailState.STOP_LATCHED
    assert decision.cmd_vel.linear.x == pytest.approx(0.0)
    assert decision.cmd_vel.angular.z == pytest.approx(0.42)


def test_unavailable_assessment_degrades_with_the_reported_reason(
    module: RGBCollisionGuardrail,
) -> None:
    decision = module._decision_from_assessment(RiskUnavailable("current_roi_occluded"), _cmd(0.4))

    assert decision.state == GuardrailState.SENSOR_DEGRADED
    assert decision.reason == "current_roi_occluded"
    assert decision.cmd_vel == Twist.zero()


def test_frame_shape_mismatch_returns_sensor_degraded_zero(
    module: RGBCollisionGuardrail,
) -> None:
    now = time.monotonic()

    with module._condition:
        module._runtime_state.latest_cmd_vel = _cmd()
        module._runtime_state.latest_cmd_time = now
        module._runtime_state.previous_image = _textured_gray_image(width=160, height=120)
        module._runtime_state.latest_image = _textured_gray_image(width=160, height=90)
        module._runtime_state.previous_image_time = now
        module._runtime_state.latest_image_time = now
        decision = module._input_failure_decision_locked(now)

    assert decision is not None
    assert decision.state == GuardrailState.SENSOR_DEGRADED
    assert decision.reason == "frame_shape_mismatch"
    assert decision.cmd_vel == Twist.zero()


def test_frozen_stream_degrades_after_repeated_identical_frames(
    module: RGBCollisionGuardrail,
) -> None:
    frame = _textured_gray_image()

    with module._condition:
        module._runtime_state.previous_image = frame
        module._runtime_state.latest_image = frame

        first = module._frozen_stream_decision_locked()
        second = module._frozen_stream_decision_locked()
        third = module._frozen_stream_decision_locked()

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

    with module._condition:
        module._runtime_state.previous_image = frame
        module._runtime_state.latest_image = frame
        module._frozen_stream_decision_locked()
        module._frozen_stream_decision_locked()

        module._runtime_state.latest_image = _textured_gray_image(shift_x=3)
        module._frozen_stream_decision_locked()

        assert module._runtime_state.static_frame_hits == 0


def test_stale_risk_returns_sensor_degraded_zero(module: RGBCollisionGuardrail) -> None:
    now = time.monotonic()
    stale_risk_time = now - 0.2

    with module._condition:
        module._runtime_state.latest_cmd_vel = _cmd()
        module._runtime_state.latest_cmd_time = now
        module._runtime_state.previous_image = _textured_gray_image()
        module._runtime_state.latest_image = _textured_gray_image(shift_x=2)
        module._runtime_state.previous_image_time = now
        module._runtime_state.latest_image_time = now
        module._runtime_state.last_risk_time = stale_risk_time
        decision = module._risk_staleness_decision_locked(now)

    assert decision is not None
    assert decision.state == GuardrailState.SENSOR_DEGRADED
    assert decision.reason == "risk_state_stale"
    assert decision.cmd_vel == Twist.zero()


def test_stale_command_publishes_zero_output(module: RGBCollisionGuardrail) -> None:
    now = time.monotonic()
    stale_cmd_time = now - 0.2

    with module._condition:
        module._runtime_state.latest_cmd_vel = _cmd()
        module._runtime_state.latest_cmd_time = stale_cmd_time
        module._runtime_state.last_decision = _decision(GuardrailState.PASS, _cmd())
        module._runtime_state.pending_cmd_update = True
        cmd_to_publish = module._consume_publish_cmd_locked(now)

    assert cmd_to_publish == Twist.zero()


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
        risk_timeout_s=5.0,
    )

    try:
        cmd_transport.publish(_cmd())
        frame = _textured_gray_image()
        image_transport.publish(frame)
        image_transport.publish(frame)

        time.sleep(0.2)

        with guardrail._condition:
            assert guardrail._runtime_state.static_frame_hits == 1
            assert guardrail._runtime_state.state != GuardrailState.SENSOR_DEGRADED
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
        **frame_counts,
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
        caution_frame_count=1,
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
        stop_frame_count=1,
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
        guarded_output_publish_hz=100.0,
        risk_evaluation_hz=2.0,
        command_timeout_s=1.0,
        image_timeout_s=1.0,
        risk_timeout_s=1.0,
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
        stop_frame_count=1,
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
