#!/bin/bash
# LingBot-World KVM4 VPS Setup Script (CPU-Only)
set -e

echo "================================================================="
echo "   Setting up LingBot-World on your Hostinger KVM4 (CPU-Only)  "
echo "================================================================="

# 1. Update packages
echo "[1/4] Updating package manager..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv python3-dev git git-lfs

# 2. Set up virtual environment
echo "[2/4] Setting up python virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# 3. Install CPU-only PyTorch and project dependencies
echo "[3/4] Installing CPU-optimized PyTorch and library dependencies..."
# Installing the CPU-only build of torch saves ~2GB of download size and runs better on VPS
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install "huggingface_hub[cli]"

# 4. Download 4-bit Quantized Model
echo "[4/4] Downloading 4-bit Quantized model weights..."
# Using the 4-bit quantized model is highly recommended for KVM VPS to save system memory
huggingface-cli download cahlen/lingbot-world-base-cam-nf4 --local-dir ./lingbot-world-base-cam

echo ""
echo "================================================================="
echo " SETUP COMPLETED SUCCESSFULLY!"
echo "================================================================="
echo "To activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo "To generate a custom video using your Action String trajectory:"
echo "   python generate.py \\"
echo "     --task i2v-A14B \\"
echo "     --size 480*832 \\"
echo "     --ckpt_dir lingbot-world-base-cam \\"
echo "     --image examples/05/image.jpg \\"
echo "     --action_path examples/05 \\"
echo "     --action_string \"w-10,j-10,s-15,l-10\" \\"
echo "     --allow_act2cam \\"
echo "     --sample_steps 20 \\"
echo "     --offload_model True \\"
echo "     --t5_cpu \\"
echo "     --prompt \"A soaring journey through a fantasy jungle...\""
echo "================================================================="
