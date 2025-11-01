#!/bin/bash
#SBATCH --job-name=variable_data_size
#SBATCH --partition=gpu_part
#SBATCH --nodelist=GPU06
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=1024G
#SBATCH --time=72:00:00
#SBATCH --output=MFNV2_varsize_%j.out
#SBATCH --error=MFNV2_varsize_%j.err

# MFNversion_2 variable dataset-size training - study finetuning data requirements
# Description: Train with 1/5, 2/5, 3/5, 4/5, 5/5 of the data and stop early when validation accuracy stabilizes above a threshold

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
FINETUNE_DATA_ROOT="/public_new/work_space/fengjiali/MKwithML/Dataset_finetune"
TEST_DATA_ROOT="/public_new/work_space/fengjiali/MKwithML/Dataset_finetune_test"
OUTPUT_BASE_DIR="./outputs/variable_data_size_$(date +%Y%m%d_%H%M%S)"

# ⚠️ Important: update to your pretrained checkpoint path
PRETRAIN_CHECKPOINT="/public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/outputs/batched/pretrain_20251030_093113/batch_10_best_model.pth"

# Training parameters
MAX_EPOCHS=30              # Train up to 30 epochs per dataset fraction
ACC_THRESHOLD=0.91           # Accuracy threshold
PATIENCE=5                  # Stop after threshold met for 5 consecutive epochs

# Test-set parameters
# Format: "1,2,3" means use parts 1, 2, and 3 of the test set (total 5 parts)
# Example: "1,2,3" uses 60% of the test set
# Leave empty or use "1,2,3,4,5" to evaluate on the full test set
TEST_DATA_FRACTIONS="1,2,3"  # Default: use the full test set

echo ""
echo "Starting variable dataset-size training..."
echo "Training data root: ${FINETUNE_DATA_ROOT}"
echo "Test data root: ${TEST_DATA_ROOT}"
echo "Test partitions: ${TEST_DATA_FRACTIONS}"
echo "Output base dir: ${OUTPUT_BASE_DIR}"
echo "Pretrained checkpoint: ${PRETRAIN_CHECKPOINT}"
echo "Accuracy threshold: ${ACC_THRESHOLD}"
echo "Early-stopping patience: ${PATIENCE} epochs"
echo ""

# Verify pretrained checkpoint exists
if [ ! -f "${PRETRAIN_CHECKPOINT}" ]; then
    echo "Error: pretrained checkpoint not found: ${PRETRAIN_CHECKPOINT}"
    echo "Complete pretraining first or update PRETRAIN_CHECKPOINT"
    exit 1
fi

# Verify test data path exists
if [ ! -d "${TEST_DATA_ROOT}" ]; then
    echo "Warning: test data path not found: ${TEST_DATA_ROOT}"
    echo "Skipping test-set evaluation"
    TEST_DATA_ROOT=""
fi

# Create output base directory
mkdir -p ${OUTPUT_BASE_DIR}

# Iterate over five data fractions
for i in 1 2 3 4 5; do
    echo ""
    echo "=========================================="
    echo "Training run: using ${i}/5 of the data"
    echo "=========================================="
    
    # Compute the current fraction of folders
    # Assume N total folders and take i/5 * N each time
    DATA_FRACTION=$(echo "scale=4; $i / 5.0" | bc)
    
    # Output directory for this run
    CURRENT_OUTPUT_DIR="${OUTPUT_BASE_DIR}/data_fraction_${i}_of_5"
    
    # Build test arguments
    if [ -n "${TEST_DATA_ROOT}" ]; then
        TEST_ARGS="--test_data_root ${TEST_DATA_ROOT} --test_data_fractions ${TEST_DATA_FRACTIONS}"
    else
        TEST_ARGS=""
    fi
    
    # Launch training with early stopping enabled
    python train_variable_data_size.py \
        --mode finetune \
        --data_root ${FINETUNE_DATA_ROOT} \
        --output_dir ${CURRENT_OUTPUT_DIR} \
        --pretrain_checkpoint ${PRETRAIN_CHECKPOINT} \
        --data_fraction ${DATA_FRACTION} \
        --freeze_cnn_layers 3 \
        --max_epochs ${MAX_EPOCHS} \
        --acc_threshold ${ACC_THRESHOLD} \
        --patience ${PATIENCE} \
        --train_ratio 0.8 \
        --seq_len_min 5 \
        --seq_len_max 10 \
        --seq_step 1 \
        --img_h 1024 \
        --img_w 1280 \
        --batch_size 8 \
        --lr 5e-5 \
        --weight_decay 1e-4 \
        --optimizer adamw \
        --scheduler plateau \
        --grad_clip 1.0 \
        --use_orth_loss \
        --num_workers 4 \
        --seed 42 \
        ${TEST_ARGS}
    
    echo ""
    echo "Completed training: used ${i}/5 of the data"
    echo "Artifacts stored at: ${CURRENT_OUTPUT_DIR}"
    echo ""
done

echo ""
echo "=========================================="
echo "All variable dataset-size training runs complete!"
echo "Job end time: $(date)"
echo "Results summary: ${OUTPUT_BASE_DIR}/summary.json"
echo "=========================================="
