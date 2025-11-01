#!/bin/bash
#SBATCH --job-name=noise_test
#SBATCH --partition=gpu_part
#SBATCH --nodelist=GPU06
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=noise_test_%j.out
#SBATCH --error=noise_test_%j.err

# Noise test submission script
# File path: /public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/sbatch_noise_test.sh
# Usage: sbatch sbatch_noise_test.sh

echo "=========================================="
echo "Job start time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Create log directory
mkdir -p logs

# Change to the script directory
cd /public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch

# Activate Conda environment
source /public/home/fengjiali/anaconda3/etc/profile.d/conda.sh
conda activate MKwithML2

# Core configuration
MODEL_PATH="/public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/outputs/batched/pretrain_20251030_093113/batch_10_best_model.pth"
TEST_DATA_ROOT="/public_new/work_space/fengjiali/MKwithML/Noise_test"
OUTPUT_DIR="./noise_test_results"

# Model parameters
IMG_H=1024
IMG_W=1280

# Test parameters
BATCH_SIZE=32
NUM_WORKERS=4

echo ""
echo "Starting noise robustness test..."
echo "Model path: ${MODEL_PATH}"
echo "Test data root: ${TEST_DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Run noise testing
python test_noise.py \
    --model_path ${MODEL_PATH} \
    --test_data_root ${TEST_DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --img_h ${IMG_H} \
    --img_w ${IMG_W} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS}

echo ""
echo "=========================================="
echo "Noise test finished!"
echo "Job end time: $(date)"
echo "Results saved in: ${OUTPUT_DIR}"
echo "=========================================="
