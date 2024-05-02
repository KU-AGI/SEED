CUDA_VISIBLE_DEVICES=4 \
python3 gradio_demo/seed_llama_flask.py \
    --image_transform configs/transform/clip_transform.yaml \
    --tokenizer configs/tokenizer/seed_llama_tokenizer_hf.yaml \
    --model configs/llm/seed_llama_14b.yaml \
    --port 6067 \
    --llm_device cuda:0 \
    --tokenizer_device cuda:0 \
    --offload_encoder \
    --offload_decoder 
