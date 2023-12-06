CUDA_VISIBLE_DEVICES=7 python eval_i2t.py \
 cfg_path='configs/generation_config.yaml' \
 tokenizer_cfg_path='configs/tokenizer/seed_llama_tokenizer_hf.yaml' \
 transform_cfg_path='configs/transform/clip_transform.yaml' \
 model_cfg_path='configs/llm/seed_llama_14b.yaml' \
 result_path=none result_file_path=i2t_result_14b/i2t_result.json \
 seed=0