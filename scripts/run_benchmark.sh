#!/bin/bash

python scripts/benchmark.py \
  --benchmark math500 \
  --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --alg particle-filtering \
  --endpoint http://localhost:8100/v1 \
  --api_key NO_API_KEY \
  --rm_name Qwen/Qwen2.5-Math-PRM-7B \
  --rm_device cuda:1 \
  --rm_agg_method model \
  --budgets 16,32,64 \
  --output_dir results \
  --max_tokens 2048 \
  --temperature 0.7 \
  --max_concurrency 8 \
  --shuffle_seed 42 \
  --does_eval \
  --tokens_per_step 64 \
  --is_async
