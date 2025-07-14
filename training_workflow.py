import subprocess
import os
import shutil
from pathlib import Path

# Paths
uvr_python = "/workspace/Retrieval-based-Voice-Conversion-WebUI/cc_uvr/bin/python"
rvc_python = "/workspace/Retrieval-based-Voice-Conversion-WebUI/cc_rvc/bin/python"

input_dir = "/workspace/training_data/jennie/"
vocal_dir = "/workspace/training_data/jennie/vocal/"
clean_vocal_dir = os.path.join(vocal_dir, "clean_vocal")

# Step 1: Run UVR separation
print("🔹 Running vocal separation in cc_uvr...")
subprocess.run([uvr_python, "run_separation.py", input_dir, vocal_dir], check=True)

# Step 2: Copy *_No Echo.wav to clean_vocal
print("🔹 Copying clean vocals...")
Path(clean_vocal_dir).mkdir(parents=True, exist_ok=True)

for file in Path(vocal_dir).glob("*_(No Echo).wav"):
    shutil.copy(file, Path(clean_vocal_dir) / file.name)

# Step 3: Run RVC training
print("🔹 Starting training in cc_rvc...")
subprocess.run([
    rvc_python, "rvc_train.py",
    "--singer_name", "jennie",
    "--gpus", "0",
    "--rmvpe_gpus", "0",
    "--training_data_dir", clean_vocal_dir,
    "--speaker_id", "1"
], check=True)

print("✅ Workflow completed successfully.")