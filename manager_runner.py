import os
import sys
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Add vaultwares_agentciation to sys.path
sys.path.insert(0, os.path.abspath("vaultwares_agentciation"))

from lonely_manager import LonelyManager

def alert_handler(alert):
    print(f"\n[MANAGER ALERT] {alert['severity']}: {alert['message']}")

if __name__ == "__main__":
    print("Starting Lonely Manager for USD Digital Twin project...")
    
    manager = LonelyManager(
        agent_id="usd_twin_manager",
        alert_callback=alert_handler,
        todo_path="TODO.md",
        roadmap_path="ROADMAP.md"
    )
    
    manager.start()
    
    try:
        while True:
            time.sleep(1)
            # You can add interactive logic here if needed
    except KeyboardInterrupt:
        print("Shutting down manager...")
