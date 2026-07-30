# Copyright 2026 Dimensional Inc.
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

from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
import time
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field
from reactivex.disposable import CompositeDisposable, Disposable

from dimos.control.safety.guardrail_hysteresis import (
    HysteresisConfig,
    RiskHysteresis,
    RiskLevel,
)
from dimos.control.safety.guardrail_policy import (
    GuardrailPolicy,
    RiskAssessment,
    RiskResult,
    RiskUnavailable,
)
from dimos.control.safety.guardrail_types import GuardrailDecision, GuardrailState
from dimos.control.safety.policies import AnyPolicyConfig
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

_THREAD_JOIN_TIMEOUT_S = 2.0


logger = setup_logger()


class RGBCollisionGuardrailConfig(ModuleConfig):
    model_config = ConfigDict(extra="forbid")

    # Scheduling
    min_publish_hz: float = Field(default=10.0, gt=0.0)

    # Freshness and fail-closed behavior
    command_timeout_s: float = Field(default=0.25, gt=0.0)
    image_timeout_s: float = Field(default=0.25, gt=0.0)
    frame_pair_max_gap_s: float = Field(default=0.2, gt=0.0)

    # Motion gating
    forward_motion_deadband_mps: float = Field(default=0.05, ge=0.0)
    clamp_forward_speed_mps: float = Field(default=0.1, ge=0.0)

    static_scene_frame_count: int = Field(default=3, ge=1)

    # Which detector measures risk, selected by its `kind` tag; the module builds
    # whichever config arrives. Required.
    policy: AnyPolicyConfig
    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)


@dataclass
class _GuardrailRuntimeState:
    latest_image: Image | None = None
    previous_image: Image | None = None
    latest_image_time: float | None = None
    previous_image_time: float | None = None
    latest_cmd_vel: Twist | None = None
    latest_cmd_time: float | None = None
    last_decision: GuardrailDecision | None = None
    state: GuardrailState = GuardrailState.INIT
    image_generation: int = 0
    last_evaluated_image_generation: int = -1
    # Set by the callbacks, cleared by the worker.
    wake_requested: bool = False


@dataclass(frozen=True)
class _InputSnapshot:
    """A consistent read of the incoming streams, taken under the lock."""

    previous_image: Image | None
    current_image: Image | None
    incoming_cmd_vel: Twist | None
    image_fresh: bool
    cmd_fresh: bool
    frame_pair_fresh: bool
    has_new_frame_pair: bool
    last_decision: GuardrailDecision | None


class RGBCollisionGuardrail(Module[RGBCollisionGuardrailConfig]):
    """RGB-only motion guardrail for direct Twist control.

    Sits on the command path to the robot and gates it against a monocular camera:
    forward motion is passed, clamped, or stopped according to the collision risk
    the detector reports.

    Input callbacks store the newest value and wake the worker; it also wakes
    at least every 1/min_publish_hz, and publishes every time.
    """

    default_config = RGBCollisionGuardrailConfig

    color_image: In[Image]
    incoming_cmd_vel: In[Twist]
    safe_cmd_vel: Out[Twist]

    _condition: Condition
    _runtime_state: _GuardrailRuntimeState
    _stop_event: Event
    _thread: Thread | None
    _policy: GuardrailPolicy
    _hysteresis: RiskHysteresis
    _static_frame_hits: int
    # Serialises the two writers of safe_cmd_vel: the worker and stop(). A lock of its
    # own rather than _condition, because publish() runs consumer callbacks whose
    # duration this module has no way to know, and holding _condition across them would
    # delay the input callbacks behind whatever a consumer happens to do.
    _output_lock: Lock

    def __init__(self, *, policy_override: GuardrailPolicy | None = None, **kwargs: Any) -> None:
        """Build the guardrail, optionally supplying the detector.
        """
        super().__init__(**kwargs)
        self._condition = Condition()
        self._runtime_state = _GuardrailRuntimeState()
        self._stop_event = Event()
        self._thread = None
        self._policy = policy_override if policy_override is not None else self.config.policy.build()
        self._hysteresis = RiskHysteresis(self.config.hysteresis)
        self._static_frame_hits = 0
        self._output_lock = Lock()

    @rpc
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("RGB guardrail is already running")

        if self._thread is not None:
            logger.warning("RGB guardrail worker thread exited without stop(); restarting")

        super().start()
        self._stop_event.clear()

        self._detach_inputs()
        self._reset_run_state()

        try:
            self._disposables.add(Disposable(self.color_image.subscribe(self._on_color_image)))
            self._disposables.add(
                Disposable(self.incoming_cmd_vel.subscribe(self._on_incoming_cmd_vel))
            )

            thread = Thread(
                target=self._decision_loop,
                name=f"{self.__class__.__name__}-thread",
                daemon=True,
            )
            thread.start()
        except Exception:
            self._detach_inputs()
            self._thread = None
            raise

        self._thread = thread

        logger.info(
            "RGB guardrail started",
            min_publish_hz=self.config.min_publish_hz,
            command_timeout_s=self.config.command_timeout_s,
            image_timeout_s=self.config.image_timeout_s,
            frame_pair_max_gap_s=self.config.frame_pair_max_gap_s,
        )

    @rpc
    def stop(self) -> None:
        was_running = self._thread is not None

        # Ends the worker loop, and closes the output stream with it: every publish
        # tests this flag while holding the output lock, so once it is set no decision
        # can reach the stream, including one already in flight.
        self._stop_event.set()

        self._detach_inputs()

        with self._condition:
            self._condition.notify_all()

        if self._thread is not None:
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    "RGB guardrail worker thread did not stop within timeout",
                    timeout_s=_THREAD_JOIN_TIMEOUT_S,
                )
            else:
                self._thread = None
        if was_running:
            # Under the output lock, so this lands behind any decision that is already
            # publishing rather than interleaving with it. Every later decision finds
            # the stop flag set, which is what leaves this zero as the final word.
            with self._output_lock:
                self.safe_cmd_vel.publish(Twist.zero())

        logger.info("RGB guardrail stopped")
        super().stop()

    def _detach_inputs(self) -> None:
        self._disposables.dispose()
        self._disposables = CompositeDisposable()

    def _reset_run_state(self) -> None:
        """Clear everything the worker accumulates across a run.

        Only the runtime state is written under the lock. The rest is touched by the
        worker alone, and start() has already established that no worker exists.
        """
        with self._condition:
            self._runtime_state = _GuardrailRuntimeState()
        self._hysteresis.reset()
        self._policy.reset()
        self._static_frame_hits = 0

    def _publish_unless_stopping(self, cmd_vel: Twist) -> None:
        """Write a decision to the output stream unless stop() has begun.

        The test and the publish share one critical section. Split apart, stop() could
        set the flag and publish its final zero in the gap between them, and this
        command would land afterwards as the last thing the robot hears.
        """
        with self._output_lock:
            if self._stop_event.is_set():
                return
            self.safe_cmd_vel.publish(cmd_vel)

    def _on_color_image(self, image: Image) -> None:
        now = time.monotonic()
        with self._condition:
            self._runtime_state.previous_image = self._runtime_state.latest_image
            self._runtime_state.previous_image_time = self._runtime_state.latest_image_time
            self._runtime_state.latest_image = image
            self._runtime_state.latest_image_time = now
            self._runtime_state.image_generation += 1
            self._runtime_state.wake_requested = True
            self._condition.notify()

    def _on_incoming_cmd_vel(self, cmd_vel: Twist) -> None:
        now = time.monotonic()
        with self._condition:
            self._runtime_state.latest_cmd_vel = cmd_vel
            self._runtime_state.latest_cmd_time = now
            self._runtime_state.wake_requested = True
            self._condition.notify()

    def _decision_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                # Sleep only when nothing is waiting. notify() alone is not enough:
                # it wakes a thread that is already blocked, so a message arriving
                # while this thread was deciding would otherwise be dropped and wait
                # out a full tick. The flag makes the loop follow the upstream rate
                # instead of a rate configured in advance.
                if not self._runtime_state.wake_requested:
                    self._condition.wait(timeout=self._tick_period_s())
                self._runtime_state.wake_requested = False

                if self._stop_event.is_set():
                    return

                snapshot = self._take_input_snapshot_locked(time.monotonic())

            decision = self._decide(snapshot)

            with self._condition:
                self._store_decision_locked(decision)

            self._publish_unless_stopping(decision.cmd_vel)

    def _tick_period_s(self) -> float:
        return 1.0 / self.config.min_publish_hz

    def _take_input_snapshot_locked(self, now: float) -> _InputSnapshot:
        runtime_state = self._runtime_state

        has_new_frame_pair = self._has_new_frame_pair_locked()
        if has_new_frame_pair:
            runtime_state.last_evaluated_image_generation = runtime_state.image_generation

        return _InputSnapshot(
            previous_image=runtime_state.previous_image,
            current_image=runtime_state.latest_image,
            incoming_cmd_vel=runtime_state.latest_cmd_vel,
            image_fresh=self._is_image_fresh_locked(now),
            cmd_fresh=self._is_cmd_fresh_locked(now),
            frame_pair_fresh=self._is_frame_pair_fresh_locked(),
            has_new_frame_pair=has_new_frame_pair,
            last_decision=runtime_state.last_decision,
        )

    def _has_new_frame_pair_locked(self) -> bool:
        runtime_state = self._runtime_state
        return (
            runtime_state.previous_image is not None
            and runtime_state.image_generation != runtime_state.last_evaluated_image_generation
        )

    def _is_cmd_fresh_locked(self, now: float) -> bool:
        latest_cmd_time = self._runtime_state.latest_cmd_time
        if latest_cmd_time is None:
            return False
        return (now - latest_cmd_time) <= self.config.command_timeout_s

    def _is_image_fresh_locked(self, now: float) -> bool:
        latest_image_time = self._runtime_state.latest_image_time
        if latest_image_time is None:
            return False
        return (now - latest_image_time) <= self.config.image_timeout_s

    def _is_frame_pair_fresh_locked(self) -> bool:
        previous_image_time = self._runtime_state.previous_image_time
        latest_image_time = self._runtime_state.latest_image_time
        if previous_image_time is None or latest_image_time is None:
            return False
        return (latest_image_time - previous_image_time) <= self.config.frame_pair_max_gap_s

    def _decide(self, snapshot: _InputSnapshot) -> GuardrailDecision:
        input_failure = self._input_failure_decision(snapshot)
        if input_failure is not None:
            self._hysteresis.reset()
            return input_failure

        if not self._forward_motion_commanded(snapshot.incoming_cmd_vel):
            return self._build_decision(GuardrailState.PASS, "forward_guard_inactive", snapshot)

        if not snapshot.has_new_frame_pair:
            return self._held_decision(snapshot)

        frozen_stream = self._frozen_stream_decision(snapshot)
        if frozen_stream is not None:
            self._hysteresis.reset()
            return frozen_stream

        previous_image = snapshot.previous_image
        current_image = snapshot.current_image
        if previous_image is None or current_image is None:
            self._hysteresis.reset()
            return self._build_degraded_decision("frame_pair_unavailable", snapshot)

        try:
            assessment = self._policy.evaluate(
                previous_image=previous_image,
                current_image=current_image,
            )
        except Exception:
            last_decision = snapshot.last_decision
            logger.exception(
                "RGB guardrail policy evaluation failed",
                state=(last_decision.state if last_decision else GuardrailState.INIT).value,
            )
            self._hysteresis.reset()
            return self._build_degraded_decision("policy_evaluation_failed", snapshot)

        return self._decision_from_risk_assessment(assessment, snapshot)

    def _held_decision(self, snapshot: _InputSnapshot) -> GuardrailDecision:
        """Re-issue the current state against the latest command.

        Ticks without a new frame pair must still track upstream commands, so the
        command is rebuilt rather than replayed.
        """
        last_decision = snapshot.last_decision
        if last_decision is None:
            return self._build_init_decision("waiting_for_first_decision", snapshot)

        return self._build_decision(last_decision.state, last_decision.reason, snapshot)

    def _input_failure_decision(self, snapshot: _InputSnapshot) -> GuardrailDecision | None:
        if snapshot.incoming_cmd_vel is None:
            return self._build_init_decision("no_command_received", snapshot)

        current_image = snapshot.current_image
        if current_image is None:
            return self._build_init_decision("waiting_for_first_image", snapshot)

        previous_image = snapshot.previous_image
        if previous_image is None:
            return self._build_init_decision("waiting_for_frame_pair", snapshot)

        if not snapshot.image_fresh:
            return self._build_degraded_decision("image_stale", snapshot)

        if not snapshot.frame_pair_fresh:
            return self._build_degraded_decision("frame_pair_stale", snapshot)

        if previous_image.data.shape != current_image.data.shape:
            return self._build_degraded_decision("frame_shape_mismatch", snapshot)

        return None

    def _frozen_stream_decision(self, snapshot: _InputSnapshot) -> GuardrailDecision | None:
        """Detect a camera redelivering identical frames."""
        previous_image = snapshot.previous_image
        current_image = snapshot.current_image
        if previous_image is None or current_image is None:
            return None

        if not np.array_equal(previous_image.data, current_image.data):
            self._static_frame_hits = 0
            return None

        self._static_frame_hits += 1
        if self._static_frame_hits >= self.config.static_scene_frame_count:
            return self._build_degraded_decision("static_scene", snapshot)

        return None

    def _decision_from_risk_assessment(
        self,
        assessment: RiskResult,
        snapshot: _InputSnapshot,
    ) -> GuardrailDecision:
        if isinstance(assessment, RiskUnavailable):
            self._hysteresis.reset()
            return self._build_degraded_decision(assessment.reason, snapshot)

        state = self._hysteresis.observe(assessment.level)
        return self._build_decision(state, self._reason_for_state(state, assessment), snapshot)

    def _reason_for_state(self, state: GuardrailState, assessment: RiskAssessment) -> str:
        if state == GuardrailState.STOP_LATCHED:
            return "risk_stop"

        if state == GuardrailState.CLAMP:
            # A clamp reached from a clear reading means the latch is releasing.
            return "risk_clamp" if assessment.level >= RiskLevel.CAUTION else "risk_recovery"

        return "risk_clear"

    def _forward_motion_commanded(self, incoming_cmd_vel: Twist | None) -> bool:
        if incoming_cmd_vel is None:
            return False
        return float(incoming_cmd_vel.linear.x) > self.config.forward_motion_deadband_mps

    def _command_for_state(self, state: GuardrailState, snapshot: _InputSnapshot) -> Twist:
        incoming_cmd_vel = snapshot.incoming_cmd_vel
        if incoming_cmd_vel is None or not snapshot.cmd_fresh:
            return Twist.zero()

        cmd_vel = Twist(linear=incoming_cmd_vel.linear, angular=incoming_cmd_vel.angular)

        if state == GuardrailState.PASS:
            return cmd_vel

        if state == GuardrailState.CLAMP:
            cmd_vel.linear.x = min(float(cmd_vel.linear.x), self.config.clamp_forward_speed_mps)
            return cmd_vel

        if state == GuardrailState.STOP_LATCHED:
            cmd_vel.linear.x = 0.0
            return cmd_vel

        return Twist.zero()

    def _build_decision(
        self,
        state: GuardrailState,
        reason: str,
        snapshot: _InputSnapshot,
    ) -> GuardrailDecision:
        return GuardrailDecision(
            state=state,
            cmd_vel=self._command_for_state(state, snapshot),
            reason=reason,
        )

    def _build_init_decision(self, reason: str, snapshot: _InputSnapshot) -> GuardrailDecision:
        return self._build_decision(GuardrailState.INIT, reason, snapshot)

    def _build_degraded_decision(self, reason: str, snapshot: _InputSnapshot) -> GuardrailDecision:
        return self._build_decision(GuardrailState.SENSOR_DEGRADED, reason, snapshot)

    def _store_decision_locked(self, decision: GuardrailDecision) -> None:
        previous_state = self._runtime_state.state

        self._runtime_state.last_decision = decision
        self._runtime_state.state = decision.state

        if previous_state != decision.state:
            logger.info(
                "RGB guardrail state changed",
                previous_state=previous_state.value,
                state=decision.state.value,
                reason=decision.reason,
            )

rgb_collision_guardrail = RGBCollisionGuardrail.blueprint
