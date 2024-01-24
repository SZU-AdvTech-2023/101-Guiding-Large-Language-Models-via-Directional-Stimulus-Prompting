gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
# summarization with hint

python scripts/training/train_text_generation.py \
    --base_path_to_store_results ./rl4lms_exps \
    --project_name summarization_with_hint \
    --experiment_name flan-t5-large_nlpo_on_supervised-cnndm_1000_rg_avg \
    --config_path scripts/training/task_configs/summarization_with_hint/flan-t5_nlpo_on_supervised-cnndm_1000.yml

