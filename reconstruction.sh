CUDA_VISIBLE_DEVICES=0 python reconstruction.py \
 cfg_path='configs/generation_config.yaml' \
 tokenizer_cfg_path='configs/tokenizer/seed_llama_tokenizer_hf.yaml' \
 transform_cfg_path='configs/transform/clip_transform.yaml' \
 seed=0