"""
For OpenAI CODEX environment.
Don't forget:
- provide full internet aceess to the environment.
- add GDRIVE_API_KEY to secretes.
This script downloads a Parquet file from Google Drive and saves it to a specified directory.
Add this script to https://chatgpt.com/codex/settings/environments
`uv sync`
`uv run -m history.codex`
"""

import os
import requests
from pathlib import Path

# API key generated from Google Cloud Console (must enable Drive API)
API_KEY = os.getenv("GDRIVE_API_KEY")  # Put this in env

# You can modify this list as needed
FILES_TO_DOWNLOAD = [
    {
        "file_id": "1gQ8ZF61VxnzNqoE7W_UNquVmPq68PvSc", 
        "save_path": "nexus/history/binance-futures/BTCUSDT_1m.parquet"
    },
    # Add more entries here
]

def download_large_file(file_id: str, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={API_KEY}"

    print(f"📥 Downloading via API: {file_id} → {output_path}")

    response = requests.get(url, stream=True, verify=False)
    if response.status_code != 200:
        print(f"❌ Failed with status {response.status_code}: {response.text}")
        return

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✅ Downloaded to: {output_path}")

for f in FILES_TO_DOWNLOAD:
    download_large_file(f["file_id"], f["save_path"])
