"""Upload training data to Modal volume."""

import modal
from pathlib import Path

volume = modal.Volume.from_name("braille-training-data", create_if_missing=True)

app = modal.App("braille-upload")

@app.function(volumes={"/data": volume})
def upload_file(data: bytes):
    with open("/data/braille_train.pt", "wb") as f:
        f.write(data)
    volume.commit()
    print("Data written to volume")

@app.local_entrypoint()
def main(file_path: str = "distill/data/braille_train.pt"):
    local_path = Path(file_path)
    if not local_path.exists():
        print(f"Error: File not found: {local_path}")
        return
    
    print(f"Uploading {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)...")
    
    with open(local_path, "rb") as f:
        data = f.read()
    
    upload_file.remote(data)
    print("Done! Data uploaded to /data/braille_train.pt")
