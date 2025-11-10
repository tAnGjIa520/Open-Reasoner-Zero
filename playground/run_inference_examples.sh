#!/bin/bash

# ============================================================================
# Open-Reasoner-Zero 推理和评估脚本示例
# ============================================================================
# 这个脚本展示了如何使用 inference_eval_only.py 进行推理和评估
# ============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# 示例 1: 基础推理评估（使用默认参数）
# ============================================================================
example_basic() {
    print_info "示例 1: 基础推理评估"
    print_warning "请替换 /path/to/your/model 为你的实际模型路径"

    python -m playground.inference_eval_only \
        exp.pretrain=/path/to/your/model
}

# ============================================================================
# 示例 2: 指定本地模型路径
# ============================================================================
example_local_model() {
    print_info "示例 2: 使用本地模型"

    # 假设你的模型在以下路径
    MODEL_PATH="/mnt/shared-storage-user/tangjia/your_model"
    EVAL_DATA='["data/eval_data/math500.json"]'
    SAVE_PATH="./eval_results/your_model"

    python -m playground.inference_eval_only \
        exp.pretrain="${MODEL_PATH}" \
        exp.eval_prompt_data="${EVAL_DATA}" \
        exp.save_path="${SAVE_PATH}"
}

# ============================================================================
# 示例 3: 使用 Hugging Face 模型
# ============================================================================
example_hf_model() {
    print_info "示例 3: 使用 Hugging Face 模型"

    python -m playground.inference_eval_only \
        exp.pretrain=Qwen/Qwen2.5-7B \
        exp.eval_prompt_data='["data/eval_data/math500.json","data/eval_data/aime2024.json"]' \
        exp.save_path="./eval_results/qwen_7b"
}

# ============================================================================
# 示例 4: 自定义推理参数（高质量评估）
# ============================================================================
example_high_quality() {
    print_info "示例 4: 高质量推理（较慢但更准确）"

    python -m playground.inference_eval_only \
        exp.pretrain=/path/to/your/model \
        exp.temperature=1.0 \
        exp.top_p=1.0 \
        exp.generate_max_len=16000 \
        exp.vllm_num_engines=4 \
        exp.save_path="./eval_results/high_quality"
}

# ============================================================================
# 示例 5: 快速推理（降低延迟）
# ============================================================================
example_fast() {
    print_info "示例 5: 快速推理（低延迟）"

    python -m playground.inference_eval_only \
        exp.pretrain=/path/to/your/model \
        exp.temperature=0.5 \
        exp.top_p=0.9 \
        exp.generate_max_len=4096 \
        exp.vllm_num_engines=8 \
        exp.gpu_memory_utilization=0.9 \
        exp.save_path="./eval_results/fast"
}

# ============================================================================
# 示例 6: 评估多个数据集
# ============================================================================
example_multiple_datasets() {
    print_info "示例 6: 评估多个数据集"

    python -m playground.inference_eval_only \
        exp.pretrain=/path/to/your/model \
        exp.eval_prompt_data='[
            "data/eval_data/math500.json",
            "data/eval_data/aime2024.json",
            "data/eval_data/gpqa_diamond.json"
        ]' \
        exp.save_path="./eval_results/multi_dataset"
}

# ============================================================================
# 示例 7: 完整示例（推荐）
# ============================================================================
example_full() {
    print_info "示例 7: 完整示例（推荐配置）"

    MODEL_PATH="/mnt/shared-storage-user/tangjia/your_model"

    # 检查模型是否存在
    if [ ! -d "${MODEL_PATH}" ]; then
        print_error "模型路径不存在: ${MODEL_PATH}"
        print_warning "请修改 MODEL_PATH 变量为你的实际模型路径"
        return 1
    fi

    print_info "使用模型: ${MODEL_PATH}"

    python -m playground.inference_eval_only \
        exp.pretrain="${MODEL_PATH}" \
        exp.eval_prompt_data='["data/eval_data/math500.json"]' \
        exp.save_path="./eval_results/$(basename ${MODEL_PATH})" \
        exp.vllm_num_engines=4 \
        exp.temperature=1.0 \
        exp.top_p=1.0 \
        exp.generate_max_len=8000 \
        exp.gpu_memory_utilization=0.75

    if [ $? -eq 0 ]; then
        print_success "评估完成！结果已保存到 ./eval_results/$(basename ${MODEL_PATH})"
    else
        print_error "评估过程中出现错误"
        return 1
    fi
}

# ============================================================================
# 主菜单
# ============================================================================
main() {
    print_info "Open-Reasoner-Zero 推理和评估脚本示例"
    echo ""
    echo "可用的示例:"
    echo "  1) basic           - 基础推理评估"
    echo "  2) local_model     - 使用本地模型"
    echo "  3) hf_model        - 使用 Hugging Face 模型"
    echo "  4) high_quality    - 高质量推理"
    echo "  5) fast            - 快速推理"
    echo "  6) multiple_datasets - 评估多个数据集"
    echo "  7) full            - 完整示例（推荐）"
    echo ""

    if [ $# -eq 0 ]; then
        print_warning "请指定一个示例"
        echo "使用方法: bash run_examples.sh <example>"
        echo "示例: bash run_examples.sh full"
        exit 1
    fi

    case "$1" in
        basic)
            example_basic
            ;;
        local_model)
            example_local_model
            ;;
        hf_model)
            example_hf_model
            ;;
        high_quality)
            example_high_quality
            ;;
        fast)
            example_fast
            ;;
        multiple_datasets)
            example_multiple_datasets
            ;;
        full)
            example_full
            ;;
        *)
            print_error "未知的示例: $1"
            exit 1
            ;;
    esac
}

# 运行主菜单
main "$@"
