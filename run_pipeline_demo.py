import argparse
import json
import os
import sys
import threading
import time

import redis

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Add vaultwares_agentciation to sys.path
sys.path.insert(0, os.path.abspath("vaultwares_agentciation"))

class PipelineOrchestrator:
    def __init__(self, redis_host='localhost', redis_port=6379, source_video='test_input.mp4'):
        self.r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        self.pubsub = self.r.pubsub()
        self.channel = "tasks"
        self.results = {}
        self.stop_event = threading.Event()
        self.source_video = source_video
        self.r.ping()

    def _listen_for_results(self):
        print(f"[Orchestrator] listening on '{self.channel}'...")
        self.pubsub.subscribe(self.channel)
        for message in self.pubsub.listen():
            if self.stop_event.is_set():
                break
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                except json.JSONDecodeError:
                    print("[WARN] Ignoring non-JSON message on tasks channel.")
                    continue
                action = data.get('action')
                if action == "RESULT":
                    task = data.get('task')
                    details = data.get('details', {})
                    agent = data.get('agent')
                    result = details.get('result')
                    print(f"[SUCCESS] Received result for '{task}' from {agent}: {result}")
                    self.results[task] = data

    def dispatch_task(self, target, task, details):
        payload = {
            "agent": "orchestrator",
            "action": "ASSIGN",
            "task": task,
            "target": target,
            "details": details
        }
        print(f"[Orchestrator] Dispatching '{task}' to {target}...")
        self.r.publish(self.channel, json.dumps(payload))

    def wait_for_task(self, task, timeout=300):
        start_time = time.time()
        while task not in self.results:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task '{task}' timed out after {timeout}s")
            time.sleep(1)
        result = self.results[task]
        details = result.get("details", {})
        result_text = str(details.get("result", ""))
        if result_text.startswith("ERROR:"):
            raise RuntimeError(f"Task '{task}' failed: {result_text}")
        return result

    def run(self):
        listener_thread = threading.Thread(target=self._listen_for_results, daemon=True)
        listener_thread.start()

        try:
            print("\nStarting Digital Twin Pipeline Orchestration")

            # Step 1: Extraction
            self.dispatch_task(
                target="video-specialist",
                task="sample_frames",
                details={
                    "source": "test_input.mp4",
                    "source": self.source_video,
                    "output_dir": "data/extracted_frames",
                    "fps": 2  # Higher FPS for better recon
                }
            )
            self.wait_for_task("sample_frames")

            # Step 2: Reconstruction
            self.dispatch_task(
                target="recon-professional",
                task="run_colmap",
                details={
                    "images_dir": "data/extracted_frames",
                    "output_dir": "data/reconstruction"
                }
            )
            # COLMAP can take a long time, increasing timeout
            self.wait_for_task("run_colmap", timeout=600)

            # Step 3: Scene Setup
            self.dispatch_task(
                target="omni-specialist",
                task="setup_digital_twin",
                details={
                    "stage_path": "data/reconstruction/cloud.usda",
                    "output_path": "data/digital_twin_scene.usda"
                }
            )
            self.wait_for_task("setup_digital_twin")

            print("\nDigital Twin Pipeline completed successfully!")
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")
        finally:
            self.stop_event.set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Redis-driven USD pipeline orchestrator.")
    parser.add_argument("--source", default="test_input.mp4", help="Input video path for frame extraction.")
    args = parser.parse_args()
    orchestrator = PipelineOrchestrator(source_video=args.source)
    orchestrator.run()
