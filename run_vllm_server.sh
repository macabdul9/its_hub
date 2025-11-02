CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --dtype float16 \
    --port 8100 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 128 \
    --tensor-parallel-size 1