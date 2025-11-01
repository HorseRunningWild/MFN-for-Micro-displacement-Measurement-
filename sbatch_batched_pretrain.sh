#!/bin/bash
#SBATCH --job-name=dual_branch_training
#SBATCH --partition=gpu_part
#SBATCH --nodelist=GPU06
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=1024G
#SBATCH --time=72:00:00
#SBATCH --output=MFNV2_PT_%j.out
#SBATCH --error=MFN_V2_PT_%j.err

# MFNversion_2 batched training - pretraining stage
# File path: /public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/sbatch_batched_pretrain.sh
# Usage: sbatch sbatch_batched_pretrain.sh

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
PRETRAIN_DATA_ROOT="/public_new/work_space/fengjiali/MKwithML/Dataset_simulate_Second_Time"
PRETRAIN_OUTPUT_DIR="./outputs/batched/pretrain_$(date +%Y%m%d_%H%M%S)"

# Batched training parameters
NUM_BATCHES=10              # Split the data into 10 batches
EPOCHS_PER_BATCH=20         # Train 20 epochs per batch
MAX_FOLDERS=100000            # Cap at 1000 folders (~10000 images)

echo ""
echo "Starting batched pretraining..."
echo "Data root: ${PRETRAIN_DATA_ROOT}"
echo "Output dir: ${PRETRAIN_OUTPUT_DIR}"
echo "Number of batches: ${NUM_BATCHES}"
echo "Epochs per batch: ${EPOCHS_PER_BATCH}"
echo "Maximum folders: ${MAX_FOLDERS}"
echo ""

# Run batched training
python train_batched.py \
    --mode pretrain \
    --data_root ${PRETRAIN_DATA_ROOT} \
    --output_dir ${PRETRAIN_OUTPUT_DIR} \
    --num_batches ${NUM_BATCHES} \
    --epochs_per_batch ${EPOCHS_PER_BATCH} \
    --max_folders ${MAX_FOLDERS} \
    --use_model_accumulation \
    --train_ratio 0.8 \
    --seq_len_min 5 \
    --seq_len_max 10 \
    --seq_step 1 \
    --add_noise \
    --noise_std 0.01 \
    --img_h 1024 \
    --img_w 1280 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --num_workers 4 \
    --seed 42

echo ""
echo "=========================================="
echo "Batched pretraining completed!"
echo "Job end time: $(date)"
echo "Final model: ${PRETRAIN_OUTPUT_DIR}/batch_${NUM_BATCHES}_final_model.pth"
echo "Training summary: ${PRETRAIN_OUTPUT_DIR}/training_summary.json"
echo "=========================================="
