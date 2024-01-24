gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
# summarization with hint

python scripts/training/train_text_generation.py \
    --base_path_to_store_results ./rl4lms_exps_morl_llama \
    --project_name summarization_with_hint_bs_meteor \
    --experiment_name flan-t5-base_nlpo_on_supervised-cnndm_1000 \
    --config_path scripts/training/task_configs/summarization_with_hint/flan-t5_nlpo_on_supervised-cnndm_1000_llama.yml

