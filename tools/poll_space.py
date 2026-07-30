"""Poll DA3 Space build status until RUNNING or ERROR."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi
from vaultwares_studio.runners import get_hf_token

api = HfApi(token=get_hf_token())
for i in range(40):
    rt = api.get_space_runtime("clopeux/vw-studio-da3")
    print(f"{time.strftime('%H:%M:%S')} stage={rt.stage}")
    if rt.stage in ("RUNNING", "ERROR", "PAUSED"):
        break
    time.sleep(30)
