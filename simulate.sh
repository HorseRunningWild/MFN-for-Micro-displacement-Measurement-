#!/bin/bash
#SBATCH --job-name=simulate_fringe_full
#SBATCH --partition=cpu_part
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=56G
#SBATCH --time=24:00:00         
#SBATCH --output=/public_new/work_space/fengjiali/MKwithML/fitting/simulate/logs/simulate_fringe_full_%j.log
#SBATCH --error=/public_new/work_space/fengjiali/MKwithML/fitting/simulate/logs/simulate_fringe_full_%j.err
#SBATCH --begin=now

echo "🚀 Job started at $(date)"
echo "SBATCH assigned node: $(hostname)"

# 初始化 conda 环境（更健壮）
source ~/.bashrc
conda activate MKwithML2

# 切换到代码目录
cd /public_new/work_space/fengjiali/MKwithML/fitting/simulate

# 参数范围
start=0.00
end=40.00
step=5.00

# 循环 noise_factor
value=$start
while (( $(echo "$value <= $end" | bc -l) )); do
    # 格式化 noise_factor 目录名（保留两位小数）
    formatted=$(printf "%.2f" "$value")

    # 输出目录
    outdir="/public_new/work_space/fengjiali/MKwithML/fitting/simulate/noise_factor_New/${formatted}"
    mkdir -p "$outdir"

    echo "🔹 Running noise_factor=${formatted}"

    # 执行 Python 程序
    python simulate_program3.py \
        --stats "/public_new/work_space/fengjiali/MKwithML/fitting/simulate/fit_results_stats.txt" \
        --output "$outdir" \
        --num_groups 5 \
        --width 1280 \
        --height 1024 \
        --noise_factor "$formatted"

    # 下一步
    value=$(echo "$value + $step" | bc)
done

echo "All noise_factor simulations finished at $(date)"