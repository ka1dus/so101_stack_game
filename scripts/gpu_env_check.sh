#!/usr/bin/env bash
# Phase 0 check for the Windows/WSL2 GPU machine. Run INSIDE WSL Ubuntu:
#     bash gpu_env_check.sh 2>&1 | tee gpu_env_check.txt
# Needs env_check.py in the same folder. Read-only for the `lerobot` conda env:
# MuJoCo goes into a separate throwaway venv (~/.venvs/so101_stack_sim).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "===== OS / CPU / RAM"
grep PRETTY_NAME /etc/os-release
uname -r
echo "cpu cores (WSL view): $(nproc)"
lscpu | grep -E "Model name|^CPU\(s\)"
free -h | head -2
cat /mnt/c/Users/*/.wslconfig 2>/dev/null | sed 's/^/.wslconfig: /'

echo "===== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | head -4 | tail -1

echo "===== lerobot conda env (read-only)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
python - <<'EOF'
import lerobot, torch, os, subprocess
print("lerobot", getattr(lerobot, "__version__", "?"), lerobot.__file__)
src = os.path.dirname(lerobot.__file__)
try:
    print("git", subprocess.check_output(["git", "-C", src, "log", "-1", "--format=%h %cd %s"], text=True).strip())
except Exception as e:
    print("git ?", e)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, f"{p.total_memory/2**30:.1f} GiB")
    x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize()
    print("matmul ok", float((x @ x).mean()))
for m in ("av", "torchcodec", "mujoco"):
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "not importable:", type(e).__name__)
EOF
which ffmpeg && ffmpeg -version | head -1
conda deactivate

echo "===== MuJoCo offscreen render in WSL (separate venv)"
python3 -m venv ~/.venvs/so101_stack_sim 2>/dev/null || echo "python3-venv missing: sudo apt install python3-venv"
~/.venvs/so101_stack_sim/bin/pip -q install mujoco numpy pillow
for gl in egl osmesa glfw; do
  echo "--- MUJOCO_GL=$gl"
  MUJOCO_GL=$gl timeout 120 ~/.venvs/so101_stack_sim/bin/python "$HERE/env_check.py" 2>&1 | tail -8
done
