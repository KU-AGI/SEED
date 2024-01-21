CUDA_VISIBLE_DEVICES=0 python eval_t2i_only.py \
 cfg_path='configs/generation_config.yaml' \
 tokenizer_cfg_path='configs/tokenizer/seed_llama_tokenizer_hf.yaml' \
 transform_cfg_path='configs/transform/clip_transform.yaml' \
 model_cfg_path='configs/llm/seed_llama_8b.yaml' \
 result_path=none result_file_path=./t2i_result_5_captions \
 seed=0