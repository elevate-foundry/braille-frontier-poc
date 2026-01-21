"""Download trained model from Modal volume."""

import modal

volume = modal.Volume.from_name("braille-checkpoints")

app = modal.App("braille-download")

@app.local_entrypoint()
def main():
    import os
    
    # Create local directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Download files from volume
    for item in volume.listdir("/"):
        if item.path.endswith(".pt"):
            print(f"Downloading {item.path}...")
            with open(f"checkpoints/{item.path}", "wb") as f:
                for chunk in volume.read_file(item.path):
                    f.write(chunk)
            print(f"  Saved to checkpoints/{item.path}")
    
    print("\nDone! Model files saved to ./checkpoints/")
