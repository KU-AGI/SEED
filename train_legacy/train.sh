CUDA_VISIBLE_DEVICES=4,5 python train_v4_overfitting.py \
 cfg_path='configs/overfitting_test.yaml' \
 tokenizer_cfg_path='configs/tokenizer/seed_llama_tokenizer_hf.yaml' \
 transform_cfg_path='configs/transform/diffusion_transform.yaml' \
 model_cfg_path='configs/llm/seed_llama_8b.yaml' \
 result_path=none result_file_path=./train_log_direct_embedding \
 seed=0