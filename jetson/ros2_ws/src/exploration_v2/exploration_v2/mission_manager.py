"""
Mission manager node for real-world visual exploration and target search.

Reads VLM clue definitions from 'vlm_clues.json' located in the package 'config'
directory (or specified via parameter), and orchestrates autonomous search missions
via the /start_mission Action Server.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from interfaces.action import StartMission


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__("mission_manager")

        # --- PARAMETERS ---
        self.declare_parameter("clues_file", "")
        self.declare_parameter("start_mission_action", "/start_mission")
        self.declare_parameter("map_name", "real_world")
        self.declare_parameter("grid_x_min", -250.0)
        self.declare_parameter("grid_x_max", 250.0)
        self.declare_parameter("grid_y_min", -250.0)
        self.declare_parameter("grid_y_max", 250.0)
        self.declare_parameter("report_dir", "mission_reports")
        self.declare_parameter("wait_for_server_sec", 30.0)
        self.declare_parameter("shutdown_on_completion", True)
        self.declare_parameter("save_logs", True)

        self.clues_file = str(self.get_parameter("clues_file").value).strip()
        self.action_name = str(self.get_parameter("start_mission_action").value).strip()
        self.map_name = str(self.get_parameter("map_name").value).strip()
        self.grid_x_min = float(self.get_parameter("grid_x_min").value)
        self.grid_x_max = float(self.get_parameter("grid_x_max").value)
        self.grid_y_min = float(self.get_parameter("grid_y_min").value)
        self.grid_y_max = float(self.get_parameter("grid_y_max").value)
        self.report_dir = str(self.get_parameter("report_dir").value).strip()
        self.wait_for_server_sec = float(self.get_parameter("wait_for_server_sec").value)
        self.shutdown_on_completion = bool(self.get_parameter("shutdown_on_completion").value)
        self.save_logs = bool(self.get_parameter("save_logs").value)

        # State tracking
        self._mission_runs: list[dict] = []
        self._current_run: dict | None = None
        self._current_run_done = threading.Event()
        self._startup_thread: threading.Thread | None = None
        self._current_trajectory: list[dict] = []
        self._episode_start_time: float = 0.0

        # Action Client
        self._action_client = ActionClient(self, StartMission, self.action_name)

        # Clues list
        self.clues: list[dict] = self._load_vlm_clues()

        # Start sequence once node is initialized
        self.create_timer(0.2, self._start_once)
        self._started = False

    def _resolve_clues_path(self) -> Path:
        """Resolves the path to vlm_clues.json."""
        if self.clues_file:
            path = Path(self.clues_file)
            if not path.is_absolute():
                path = Path.cwd() / path
            return path

        # Try package share directory (installed location)
        try:
            pkg_share = get_package_share_directory("exploration_v2")
            installed_path = Path(pkg_share) / "config" / "vlm_clues.json"
            if installed_path.exists():
                return installed_path
        except PackageNotFoundError:
            pass

        # Fallback to source directory relative to this file
        source_path = Path(__file__).resolve().parents[1] / "config" / "vlm_clues.json"
        if source_path.exists():
            return source_path

        # Fallback to local workspace config
        workspace_candidate = Path.cwd() / "src" / "exploration_v2" / "config" / "vlm_clues.json"
        if workspace_candidate.exists():
            return workspace_candidate

        return source_path

    def _load_vlm_clues(self) -> list[dict]:
        """Loads clues definitions from vlm_clues.json."""
        clues_path = self._resolve_clues_path()
        self.get_logger().info(f"Loading VLM clues from: {clues_path}")

        if not clues_path.exists():
            self.get_logger().error(f"VLM clues file not found at: {clues_path}")
            return []

        try:
            with open(clues_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                self.get_logger().error(f"Invalid JSON format in {clues_path}: expected list or object.")
                return []

            self.get_logger().info(f"Successfully loaded {len(data)} clue mission(s) from {clues_path.name}")
            return data
        except Exception as e:
            self.get_logger().error(f"Failed to read clues file {clues_path}: {e}")
            return []

    def _start_once(self):
        if self._started:
            return
        self._started = True

        self._startup_thread = threading.Thread(target=self._run_startup_sequence, daemon=True)
        self._startup_thread.start()

    def _run_startup_sequence(self):
        if not self.clues:
            self.get_logger().error("No VLM clues available. Exiting mission manager.")
            if self.shutdown_on_completion:
                rclpy.shutdown()
            return

        self.get_logger().info(f"Waiting for action server {self.action_name}...")
        if not self._action_client.wait_for_server(timeout_sec=self.wait_for_server_sec):
            self.get_logger().error(
                f"Action server {self.action_name} unavailable after {self.wait_for_server_sec:.1f}s"
            )
            if self.shutdown_on_completion:
                rclpy.shutdown()
            return

        self.get_logger().info(f"Action server {self.action_name} connected. Starting missions.")
        self._run_missions()

        if self.shutdown_on_completion:
            self.get_logger().info("All missions finished. Shutting down mission manager.")
            rclpy.shutdown()

    def _run_missions(self):
        """Executes exploration missions for each clue in vlm_clues.json."""
        for idx, clue in enumerate(self.clues, start=1):
            if not rclpy.ok():
                break

            primary_obj = str(clue.get("primary_object", "")).strip()
            if not primary_obj:
                self.get_logger().warn(f"Skipping clue #{idx}: 'primary_object' is empty.")
                continue

            surroundings = clue.get("surroundings", [])
            sigmas = [float(s) for s in clue.get("sigmas", [])]
            d_stars = [float(d) for d in clue.get("d_stars", [])]
            obj_desc = str(clue.get("object_description", clue.get("description", ""))).strip()
            full_inst = str(clue.get("full_instruction", primary_obj)).strip()
            map_name = str(clue.get("map_name", self.map_name)).strip()

            grid_x_min = float(clue.get("grid_x_min", self.grid_x_min))
            grid_x_max = float(clue.get("grid_x_max", self.grid_x_max))
            grid_y_min = float(clue.get("grid_y_min", self.grid_y_min))
            grid_y_max = float(clue.get("grid_y_max", self.grid_y_max))

            self.get_logger().info(
                f"\n=======================================================\n"
                f"[MISSION {idx}/{len(self.clues)}] Target: '{primary_obj}'\n"
                f"Surroundings: {surroundings}\n"
                f"Sigmas: {sigmas} | D*: {d_stars}\n"
                f"Full Instruction: '{full_inst}'\n"
                f"Map: '{map_name}' | Bounds: [{grid_x_min}, {grid_x_max}, {grid_y_min}, {grid_y_max}]\n"
                f"======================================================="
            )

            self._run_single_mission(
                clue=clue,
                primary_object=primary_obj,
                surroundings=surroundings,
                sigmas=sigmas,
                d_stars=d_stars,
                object_description=obj_desc,
                full_instruction=full_inst,
                map_name=map_name,
                grid_bounds=(grid_x_min, grid_x_max, grid_y_min, grid_y_max),
                mission_index=idx,
            )

    def _run_single_mission(
        self,
        clue: dict,
        primary_object: str,
        surroundings: list,
        sigmas: list[float],
        d_stars: list[float],
        object_description: str,
        full_instruction: str,
        map_name: str,
        grid_bounds: tuple[float, float, float, float],
        mission_index: int,
    ):
        self._episode_start_time = time.time()
        self._current_trajectory = []
        self._current_run = {
            "mission_index": mission_index,
            "primary_object": primary_object,
            "full_instruction": full_instruction,
            "clue": clue,
            "mission": {
                "instruction": primary_object,
                "full_instruction": full_instruction,
                "status": "pending",
                "goal_accepted": False,
                "started_at": self._episode_start_time,
                "finished_at": None,
            },
            "feedback": [],
            "result": None,
        }
        self._current_run_done.clear()
        self._mission_runs.append(self._current_run)

        goal = StartMission.Goal()
        goal.primary_object = primary_object
        goal.object_description = object_description
        goal.full_instruction = full_instruction
        goal.surroundings_json = json.dumps(surroundings)
        goal.sigmas_json = json.dumps(sigmas)
        goal.d_stars_json = json.dumps(d_stars)
        goal.map_name = map_name
        goal.grid_x_min, goal.grid_x_max, goal.grid_y_min, goal.grid_y_max = grid_bounds

        self.get_logger().info(f"Sending StartMission goal for '{primary_object}'...")
        send_future = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback
        )
        send_future.add_done_callback(self._goal_response_callback)

        # Wait until mission completion or cancellation
        self._current_run_done.wait()

    def _feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        elapsed = float(round(time.time() - self._episode_start_time, 2))

        entry = {
            "iteration": int(fb.iteration),
            "position": {"x": float(fb.position.x), "y": float(fb.position.y), "z": float(fb.position.z)},
            "distance_travelled": float(fb.distance_travelled),
            "elapsed_time": elapsed,
        }

        if self._current_run is not None:
            self._current_run["feedback"].append(entry)

        step_entry = {
            "frame": int(fb.iteration),
            "is_collision": bool(getattr(fb, "is_collision", False)),
            "action": "explore",
            "distance_travelled": float(fb.distance_travelled),
            "position": {"x": float(fb.position.x), "y": float(fb.position.y), "z": float(fb.position.z)},
            "flight_time": elapsed,
        }
        self._current_trajectory.append(step_entry)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("StartMission goal rejected by server")
            if self._current_run is not None:
                self._current_run["mission"]["goal_accepted"] = False
                self._current_run["mission"]["status"] = "goal_rejected"
                self._current_run["mission"]["finished_at"] = time.time()
            self._current_run_done.set()
            return

        self.get_logger().info("StartMission goal accepted by server")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        wrapped = future.result()
        result = wrapped.result
        total_flight_time = float(round(time.time() - self._episode_start_time, 2))
        vlm_time = float(round(getattr(result, "vlm_inference_time", 0.0), 2))

        result_payload = {
            "status": int(wrapped.status),
            "success": bool(result.success),
            "target_found": bool(result.target_found),
            "is_collision": bool(getattr(result, "is_collision", False)),
            "steps": int(result.steps),
            "distance_travelled": float(result.distance_travelled),
            "flight_time": total_flight_time,
            "vlm_inference_time": vlm_time,
            "message": str(result.message),
        }

        self.get_logger().info(
            f"[RESULT: {self._current_run.get('primary_object', 'unknown')}] "
            f"success={result_payload['success']} "
            f"target_found={result_payload['target_found']} "
            f"steps={result_payload['steps']} "
            f"distance={result_payload['distance_travelled']:.2f}m "
            f"flight_time={result_payload['flight_time']:.2f}s "
            f"msg='{result_payload['message']}'"
        )

        if self._current_run is not None:
            self._current_run["mission"]["goal_accepted"] = True
            self._current_run["mission"]["status"] = "completed" if result_payload["success"] else "finished"
            self._current_run["mission"]["finished_at"] = time.time()
            self._current_run["result"] = result_payload

            if self.save_logs:
                self._save_mission_report(result_payload)

            self._current_trajectory = []

        self._current_run_done.set()

    def _save_mission_report(self, result_payload: dict):
        """Saves real-world mission outcome and trajectory log."""
        target_name = self._current_run.get("primary_object", "target").replace(" ", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_dir = Path(self.report_dir) / f"{target_name}_{timestamp}"

        try:
            os.makedirs(report_dir / "log", exist_ok=True)

            report_data = {
                "target": self._current_run.get("primary_object", ""),
                "full_instruction": self._current_run.get("full_instruction", ""),
                "clue_data": self._current_run.get("clue", {}),
                "result": result_payload,
                "flight_time": result_payload.get("flight_time", 0.0),
                "distance_travelled": result_payload.get("distance_travelled", 0.0),
                "steps": result_payload.get("steps", 0),
            }

            with open(report_dir / "mission_summary.json", "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            with open(report_dir / "log" / "trajectory.jsonl", "w", encoding="utf-8") as f:
                for step in self._current_trajectory:
                    f.write(json.dumps(step) + "\n")

            self.get_logger().info(f"Saved mission report to: {report_dir}")
        except Exception as e:
            self.get_logger().error(f"Failed to save mission report: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()