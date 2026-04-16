import os
import sys
import time
import threading

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Add vaultwares-agentciation to sys.path
sys.path.insert(0, os.path.abspath("..\\vaultwares-agentciation"))

from agents.video_agent import VideoAgent
from agents.text_agent import TextAgent

def start_video_agent():
    print("Starting Video Agent...")
    agent = VideoAgent(agent_id="video-specialist")
    agent.start()
    while True:
        time.sleep(1)

def start_text_agent():
    print("Starting Text Agent...")
    agent = TextAgent(agent_id="text-analyst")
    agent.start()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    t1 = threading.Thread(target=start_video_agent, daemon=True)
    t2 = threading.Thread(target=start_text_agent, daemon=True)
    
    t1.start()
    t2.start()
    
    print("Workers are online. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down workers...")
