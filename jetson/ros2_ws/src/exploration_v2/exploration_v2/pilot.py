"""Manual-pilot arrival monitoring. Replaces the AirSim-era move_drone.navigation."""

import math
import time

from enum import Enum

import numpy as np

from geometry_msgs.msg import Point, PoseStamped, Quaternion
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from .utils import lookup_static_transform


class NavigationResult(Enum):
    """Outcome of getting the drone to a commanded viewpoint."""
    SUCCESS       = "SUCCESS"
    TIMEOUT       = "TIMEOUT"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"  # Rejected before being commanded
    NO_POSE       = "NO_POSE"
    CANCELED      = "CANCELED"


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle (rad) to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ManualPilotMonitor:
    """
    Waits for a human pilot to fly the drone to a commanded viewpoint.

    Never actuates anything: it publishes the requested viewpoint and blocks
    until the measured pose stays inside tolerance for hold_time_sec, so flying
    through a waypoint does not count as an arrival.

    abort_fn is polled during the wait; return True to give up with CANCELED.
    """

    def __init__(self, node, tf_buffer, world_frame: str, body_frame: str,
                 position_tolerance: float = 1.00,
                 altitude_tolerance: float = 0.50,
                 yaw_tolerance_deg: float = 20.0,
                 hold_time_sec: float = 1.0,
                 poll_period_sec: float = 0.1,
                 log_period_sec: float = 2.0,
                 no_pose_timeout_sec: float = 10.0,
                 h_ground: float = 0.0,
                 abort_fn=None):
        self._node        = node
        self._tf_buffer   = tf_buffer
        self.world_frame  = world_frame
        self.body_frame   = body_frame
        self.pos_tol      = float(position_tolerance)
        self.alt_tol      = float(altitude_tolerance)
        self.yaw_tol      = math.radians(float(yaw_tolerance_deg))
        self.hold_sec     = float(hold_time_sec)
        self.poll_sec     = float(poll_period_sec)
        self.log_sec      = float(log_period_sec)
        self.no_pose_sec  = float(no_pose_timeout_sec)
        self.h_ground     = float(h_ground)
        self._abort_fn    = abort_fn

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.viewpoint_pub = node.create_publisher(PoseStamped, "/commanded_viewpoint", latched)
        self.guidance_pub  = node.create_publisher(MarkerArray, "/pilot_guidance", latched)

    # --- State ---

    def current_state(self) -> tuple[float, float, float, float] | None:
        """(x, y, z, yaw) of the drone body in the world frame, or None."""
        T = lookup_static_transform(self._tf_buffer, self.world_frame, self.body_frame)
        if T is None:
            return None
        yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        return float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), yaw

    # --- Wait ---

    def wait_until_at(self, x, y, z, yaw=None, timeout_sec=60.0, label="viewpoint") -> NavigationResult:
        """
        Block until the pilot holds (x, y, z[, yaw]) within tolerance.

        timeout_sec None or <= 0 waits indefinitely: use it when the pilot will
        get there eventually and giving up costs more than waiting. Losing the
        pose for no_pose_timeout_sec still fails, in either mode, so a dropped
        RTK fix or a broken TF tree does not hang the wait forever.
        """
        log = self._node.get_logger()
        yaw_str = "any" if yaw is None else f"{math.degrees(yaw):.0f}°"
        log.info(f"🕹️  FLY TO {label}: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw_str}")
        self._publish_viewpoint(x, y, z, yaw)

        indefinite   = timeout_sec is None or float(timeout_sec) <= 0.0
        deadline     = None if indefinite else time.monotonic() + float(timeout_sec)
        in_tol_since = None
        last_log     = 0.0
        pose_lost_since = None
        if indefinite:
            log.info(f"[pilot] Holding for {label} until reached — no deadline.")

        while True:
            if not self._node.context.ok() or (self._abort_fn is not None and self._abort_fn()):
                self.clear_guidance()
                return NavigationResult.CANCELED

            now   = time.monotonic()
            state = self.current_state()

            if state is None:
                in_tol_since = None
                if pose_lost_since is None:
                    pose_lost_since = now
                elif now - pose_lost_since >= self.no_pose_sec:
                    self.clear_guidance()
                    log.error(f"No pose for {self.world_frame} -> {self.body_frame} "
                              f"in {self.no_pose_sec:.0f} s. Giving up on {label}.")
                    return NavigationResult.NO_POSE
                if now - last_log >= self.log_sec:
                    last_log = now
                    log.warn(f"[pilot] No pose for {self.world_frame} -> {self.body_frame}.")
            else:
                pose_lost_since = None
                cx, cy, cz, cyaw = state
                # Signed on Z and yaw so the readout says which way to correct.
                d_xy        = float(np.hypot(x - cx, y - cy))
                dz_signed   = z - cz
                dyaw_signed = 0.0 if yaw is None else wrap_to_pi(yaw - cyaw)
                d_z, d_yaw  = abs(dz_signed), abs(dyaw_signed)

                within = (d_xy <= self.pos_tol and d_z <= self.alt_tol and d_yaw <= self.yaw_tol)
                self._publish_guidance(x, y, z, yaw, d_xy, dz_signed, dyaw_signed, within)

                if within:
                    if in_tol_since is None:
                        in_tol_since = now
                    if now - in_tol_since >= self.hold_sec:
                        log.info(f"✅ {label} reached (dxy={d_xy:.2f} m, dz={d_z:.2f} m).")
                        self.clear_guidance()
                        return NavigationResult.SUCCESS
                else:
                    in_tol_since = None

                if now - last_log >= self.log_sec:
                    last_log = now
                    left = "no deadline" if indefinite else f"{max(deadline - now, 0.0):.0f} s left"
                    log.info(
                        f"[pilot] dxy={d_xy:.2f}/{self.pos_tol:.2f} m, "
                        f"dz={d_z:.2f}/{self.alt_tol:.2f} m, "
                        f"dyaw={math.degrees(d_yaw):.0f}/{math.degrees(self.yaw_tol):.0f}°, "
                        f"{left}"
                    )

            if deadline is not None and now >= deadline:
                self.clear_guidance()
                log.warn(f"⏱️ Timed out waiting for the pilot to reach {label}.")
                return NavigationResult.TIMEOUT

            time.sleep(self.poll_sec)

    # --- RViz guidance ---

    def _publish_viewpoint(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position = Point(x=float(x), y=float(y), z=float(z))
        half = 0.5 * (0.0 if yaw is None else float(yaw))
        msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))
        self.viewpoint_pub.publish(msg)

    def _marker(self, ns, mtype, scale, rgba) -> Marker:
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.ns = ns
        m.id = 0
        m.type = mtype
        m.action = Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def _publish_guidance(self, x, y, z, yaw, d_xy, dz_signed, dyaw_signed, within):
        """Target volume + live error readout, so the pilot can fly to it in RViz."""
        # Green once inside tolerance, amber while still approaching.
        rgba = (0.2, 0.9, 0.3, 0.35) if within else (1.0, 0.65, 0.0, 0.35)
        ma = MarkerArray()

        # Acceptance volume: XY tolerance radius by Z tolerance height.
        vol = self._marker("pilot_tolerance", Marker.CYLINDER,
                           (2.0 * self.pos_tol, 2.0 * self.pos_tol, 2.0 * self.alt_tol), rgba)
        vol.pose.position = Point(x=float(x), y=float(y), z=float(z))
        vol.pose.orientation.w = 1.0
        ma.markers.append(vol)

        # Drop line to the ground, so altitude is readable in a 3D view.
        drop = self._marker("pilot_drop", Marker.LINE_LIST, (0.02, 0.0, 0.0), (*rgba[:3], 0.9))
        drop.points = [Point(x=float(x), y=float(y), z=float(z)),
                       Point(x=float(x), y=float(y), z=float(self.h_ground))]
        ma.markers.append(drop)

        if yaw is not None:
            arrow = self._marker("pilot_yaw", Marker.ARROW, (0.06, 0.12, 0.0), (*rgba[:3], 0.9))
            arrow.points = [
                Point(x=float(x), y=float(y), z=float(z)),
                Point(x=float(x + 0.5 * math.cos(yaw)), y=float(y + 0.5 * math.sin(yaw)), z=float(z)),
            ]
            ma.markers.append(arrow)

        txt = self._marker("pilot_error", Marker.TEXT_VIEW_FACING, (0.0, 0.0, 0.18), (1.0, 1.0, 1.0, 0.95))
        txt.pose.position = Point(x=float(x), y=float(y), z=float(z) + 0.35)
        txt.pose.orientation.w = 1.0
        txt.text = (f"{'HOLD' if within else 'FLY'}  "
                    f"dxy {d_xy:.2f}m  dz {dz_signed:+.2f}m  dyaw {math.degrees(dyaw_signed):+.0f}°")
        ma.markers.append(txt)

        self.guidance_pub.publish(ma)

    def clear_guidance(self):
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        self.guidance_pub.publish(ma)
