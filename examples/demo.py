"""Run the guardrail end to end on synthetic frames and print its decisions.

No robot, no camera, no dimOS runtime: the module runs on the local stub
transport, fed by a synthetic sequence whose apparent motion rises past the
caution and stop thresholds and then falls back to zero. The state walks
PASS -> CLAMP -> STOP_LATCHED, then releases back through CLAMP to PASS.

Run it with:  python examples/demo.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from dimos.control.safety.rgb_collision_guardrail import RGBCollisionGuardrail
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat

FRAME_WIDTH = 160
FRAME_HEIGHT = 120
STEP_PERIOD_S = 0.15
FORWARD_SPEED_MPS = 0.4

# Pixels of apparent motion between consecutive frames. Mean optical-flow
# magnitude tracks this almost directly, so the schedule walks the thresholds:
# 0 px reads as clear, 1 px sits above the caution threshold (0.8), and 3 px is
# past the stop threshold (1.5). The trailing zeros exercise hysteresis release,
# which deliberately takes several clear frames.
SHIFT_SCHEDULE = [0, 0, 1, 1, 1, 3, 3, 3, 0, 0, 0, 0, 0]


def _texture() -> np.ndarray:
    """Smooth, high-contrast noise: gradients optical flow can actually track,
    with enough variance to clear the policy's low-texture gate."""
    rng = np.random.default_rng(0)
    coarse = rng.integers(0, 256, size=(FRAME_HEIGHT // 8, FRAME_WIDTH // 8), dtype=np.uint8)
    return cv2.resize(coarse, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    base = _texture()

    guardrail = RGBCollisionGuardrail(
        policy={"kind": "optical_flow"},
        min_publish_hz=10.0,
        # Generous freshness windows: this demo is about the flow -> hysteresis
        # ladder, not the staleness fail-closed paths (those have their own tests).
        command_timeout_s=1.0,
        image_timeout_s=1.0,
        frame_pair_max_gap_s=1.0,
    )

    published: list[Twist] = []
    guardrail.safe_cmd_vel.subscribe(published.append)
    guardrail.start()

    command = Twist(linear=[FORWARD_SPEED_MPS, 0.0, 0.0], angular=[0.0, 0.0, 0.3])
    offset = 0

    print(f"\ncommanded forward speed: {FORWARD_SPEED_MPS} m/s\n")
    header = f"{'step':>4}  {'shift':>5}  {'state':<15} {'reason':<24} {'out m/s':>7}"
    print(header)
    print("-" * len(header))

    try:
        for step, shift in enumerate(SHIFT_SCHEDULE):
            offset += shift
            frame = Image.from_numpy(np.roll(base, offset, axis=1), format=ImageFormat.GRAY)

            guardrail.incoming_cmd_vel.transport.publish(command)
            guardrail.color_image.transport.publish(frame)
            time.sleep(STEP_PERIOD_S)

            # Reading the decision directly is a demo shortcut. IMPROVEMENTS I15
            # proposes a real status stream, which would make this a subscription.
            with guardrail._condition:
                decision = guardrail._runtime_state.last_decision

            out_x = published[-1].linear.x if published else float("nan")
            state = decision.state.value if decision else "-"
            reason = decision.reason if decision else "(no decision yet)"
            print(f"{step:>4}  {shift:>5}  {state:<15} {reason:<24} {out_x:>7.2f}")
    finally:
        guardrail.stop()
        guardrail._close_module()

    print(f"\nfinal published command: {published[-1].linear.x:.2f} m/s (zeroed on stop)")


if __name__ == "__main__":
    main()
