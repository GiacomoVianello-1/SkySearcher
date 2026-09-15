import torch
import transformers
import subprocess

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")

result = subprocess.run(['bash', '-c', 'source /opt/ros/humble/setup.bash && printenv ROS_DISTRO'], capture_output=True, text=True)
print(f"ROS2: {result.stdout.strip()}")

