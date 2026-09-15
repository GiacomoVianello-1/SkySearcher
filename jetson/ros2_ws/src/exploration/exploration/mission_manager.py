from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from interfaces.action import StartMission


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__("mission_manager")

        self.declare_parameter("instruction", "lion plush near the drone") 
        self.declare_parameter("start_mission_action", "/start_mission")
        self.declare_parameter("report_dir", "mission_reports")
        self.declare_parameter("report_path", "")
        self.declare_parameter("wait_for_server_sec", 30.0)
        self.declare_parameter("shutdown_on_completion", True)

        self.instruction = str(self.get_parameter("instruction").value).strip()
        self.action_name = str(self.get_parameter("start_mission_action").value)
        self.report_dir = str(self.get_parameter("report_dir").value)
        self.report_path = str(self.get_parameter("report_path").value).strip()
        self.wait_for_server_sec = float(self.get_parameter("wait_for_server_sec").value)
        self.shutdown_on_completion = bool(self.get_parameter("shutdown_on_completion").value)

        self._mission_runs: list[dict] = []
        self._current_run: dict | None = None
        self._current_run_done = threading.Event()
        self._startup_thread: threading.Thread | None = None
        self._report_lock = threading.Lock()
        self._resolved_report_path: Path | None = None

        self._action_client = ActionClient(self, StartMission, self.action_name)

        if not self.instruction:
            self.get_logger().error("Parameter 'instruction' is empty. Nothing to do.")
            if self.shutdown_on_completion:
                rclpy.shutdown()
            return

        # Start once after the node is fully created.
        self.create_timer(0.2, self._start_once)
        self._started = False

    def _start_once(self):
        if self._started:
            return
        self._started = True

        self._startup_thread = threading.Thread(target=self._run_startup_sequence, daemon=True)
        self._startup_thread.start()

    def _run_startup_sequence(self):
        self.get_logger().info(f"Waiting for action server {self.action_name}...")
        if not self._action_client.wait_for_server(timeout_sec=self.wait_for_server_sec):
            self.get_logger().error(
                f"Action server {self.action_name} unavailable after {self.wait_for_server_sec:.1f}s"
            )
            self._persist_report(
                status="server_unavailable",
                summary={"completed_runs": 0, "failed_runs": 0},
                runs=[],
            )
            if self.shutdown_on_completion:
                rclpy.shutdown()
            return

        self._mission_runs = []
        self._run_single_mission()

        completed_runs = sum(1 for run in self._mission_runs if run["mission"]["status"] == "completed")
        failed_runs = len(self._mission_runs) - completed_runs
        report_path = self._persist_report(
            status="completed" if failed_runs == 0 else "completed_with_errors",
            summary={
                "completed_runs": completed_runs,
                "failed_runs": failed_runs,
            },
            runs=self._mission_runs,
        )
        self.get_logger().info(f"Mission report saved to: {report_path}")

        if self.shutdown_on_completion:
            rclpy.shutdown()

    def _run_single_mission(self) -> dict:
        self._current_run = self._create_run_record()
        self._current_run_done.clear()
        self._mission_runs.append(self._current_run)
        self._persist_report(status="running", summary=self._build_summary(), runs=self._mission_runs)

        self.get_logger().info(f"Waiting for action server {self.action_name}...")
        if not self._action_client.wait_for_server(timeout_sec=self.wait_for_server_sec):
            self.get_logger().error(
                f"Action server {self.action_name} unavailable after {self.wait_for_server_sec:.1f}s"
            )
            self._current_run["mission"]["status"] = "server_unavailable"
            self._current_run["mission"]["finished_at"] = time.time()
            self._persist_report(status="running", summary=self._build_summary(), runs=self._mission_runs)
            return self._current_run

        goal = StartMission.Goal()
        goal.instruction = self.instruction

        self._current_run["mission"]["started_at"] = time.time()
        self._persist_report(status="running", summary=self._build_summary(), runs=self._mission_runs)
        self.get_logger().info(f"Sending StartMission goal with instruction: '{self.instruction}'")

        send_future = self._action_client.send_goal_async(goal, feedback_callback=self._feedback_callback)
        send_future.add_done_callback(self._goal_response_callback)
        self._current_run_done.wait()

        return self._current_run

    def _create_run_record(self) -> dict:
        return {
            "mission": {
                "instruction": self.instruction,
                "action_name": self.action_name,
                "goal_accepted": False,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
            },
            "result": None,
            "feedback": [],
            "ground_truth": {          # To be manually annotated after the run
                "x": None,
                "y": None
            }
        }

    def _feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        entry = {
            "timestamp":             time.time(),
            "iteration":             int(fb.iteration),
            "position": {
                "x": float(fb.position.x),
                "y": float(fb.position.y),
                "z": float(fb.position.z),
            },
            "orientation": {
                "x": float(fb.orientation.x),
                "y": float(fb.orientation.y),
                "z": float(fb.orientation.z),
                "w": float(fb.orientation.w),
            },
            "distance_travelled":    float(fb.distance_travelled),
            "coverage_area_m2":      float(fb.coverage_area),
            "max_posterior":         float(fb.max_posterior),
            "ig_geometric":          float(fb.ig_geometric),
            "ig_semantic":           float(fb.ig_semantic),
            "vlm_inference_time_sec": float(fb.vlm_inference_time_sec),
            "target_estimate": {
                "valid": bool(fb.target_estimate_valid),
                "x": float(fb.target_estimate_x) if fb.target_estimate_valid else None,
                "y": float(fb.target_estimate_y) if fb.target_estimate_valid else None,
            },
        }
        if self._current_run is not None:
            self._current_run["feedback"].append(entry)
            self._persist_report(status="running", summary=self._build_summary(), runs=self._mission_runs)

        self.get_logger().info(
            f"[feedback] iter={entry['iteration']} "
            f"pos=({entry['position']['x']:.2f}, {entry['position']['y']:.2f}) "
            f"dist={entry['distance_travelled']:.2f}m "
            f"cov={entry['coverage_area_m2']:.2f}m² "
            f"max_P={entry['max_posterior']:.3f} "
            f"IG=({entry['ig_geometric']:.2f}+{entry['ig_semantic']:.2f}) "
            f"vlm={entry['vlm_inference_time_sec']:.1f}s"
        )

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("StartMission goal rejected")
            if self._current_run is not None:
                self._current_run["mission"]["goal_accepted"] = False
                self._current_run["mission"]["status"] = "goal_rejected"
                self._current_run["mission"]["finished_at"] = time.time()
                self._persist_report(status="running", summary=self._build_summary(), runs=self._mission_runs)
            self._current_run_done.set()
            return

        self.get_logger().info("StartMission goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        wrapped = future.result()
        result = wrapped.result

        result_payload = {
            "status": int(wrapped.status),
            "success": bool(result.success),
            "target_found": bool(result.target_found),
            "steps": int(result.steps),
            "distance_travelled": float(result.distance_travelled),
            "message": str(result.message),
        }

        self.get_logger().info(
            f"[result] success={result_payload['success']} "
            f"target_found={result_payload['target_found']} "
            f"steps={result_payload['steps']} "
            f"distance={result_payload['distance_travelled']:.2f} m "
            f"message='{result_payload['message']}'"
        )

        if self._current_run is not None:
            self._current_run["mission"]["goal_accepted"] = True
            self._current_run["mission"]["status"] = "completed"
            self._current_run["mission"]["finished_at"] = time.time()
            self._current_run["result"] = result_payload
            self._persist_report(status="running", summary=self._build_summary(), runs=self._mission_runs)
        self._current_run_done.set()

    def _resolve_report_path(self) -> Path:
        if self._resolved_report_path is not None:
            return self._resolved_report_path

        if self.report_path:
            self._resolved_report_path = Path(self.report_path).expanduser()
            return self._resolved_report_path

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self._resolved_report_path = Path(self.report_dir).expanduser() / f"real_mission_report_{ts}.json"
        return self._resolved_report_path

    def _build_summary(self) -> dict:
        completed_runs = sum(1 for run in self._mission_runs if run["mission"]["status"] == "completed")
        failed_runs = sum(
            1 for run in self._mission_runs if run["mission"]["status"] not in {"completed", "pending"}
        )
        return {
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
        }

    def _build_report(self, status: str, summary: dict, runs: list[dict]) -> dict:
        started_at = self._mission_runs[0]["mission"]["started_at"] if self._mission_runs else None
        finished_at = None
        for run in reversed(self._mission_runs):
            finished_at = run["mission"].get("finished_at")
            if finished_at is not None:
                break

        return {
            "status": status,
            "mission": {
                "instruction": self.instruction,
                "action_name": self.action_name,
                "started_at": started_at,
                "finished_at": finished_at,
            },
            "summary": summary,
            "runs": runs,
        }

    def _persist_report(self, status: str, summary: dict, runs: list[dict]) -> str:
        report = self._build_report(status=status, summary=summary, runs=runs)

        path = self._resolve_report_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._report_lock:
            with path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return str(path)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()