MODEL_PATH=/aifs4su/baichuan/share
MODEL_NAME=baichuan-moe-hf-latest


export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# # Perform AWQ search and save search results (we already did it for you):
# mkdir -p awq_cache
# python -m awq.entry --model_path $MODEL_PATH/$MODEL_NAME \
#     --w_bit 4 --q_group_size 128 \
#     --run_awq --dump_awq awq_cache/llama-2-7b-chat-w4-g128.pt

# Generate real quantized weights (INT4):
# mkdir -p quant_cache
# python -m awq.entry --model_path $MODEL_PATH/$MODEL_NAME \
#     --w_bit 4 --q_group_size 128 \
#     --load_awq awq_cache/llama-2-7b-chat-w4-g128.pt \
    # --q_backend real --dump_quant quant_cache/llama-2-7b-chat-w4-g128-awq.pt


# Run the TinyChat demo:
CUDA_VISIBLE_DEVICES=2 python demo.py --model_type baichuan_moe \
    --model_path $MODEL_PATH/$MODEL_NAME \
    --precision W4A16 \
    --q_group_size 128 --load_quant ../quant_cache/baichuan-moe-hf-latest-w4-g128-awq-v2.pt 

# # Split checkpoint into shards for mem-efficient loading:
# python split_ckpt.py --input_path quant_cache/llama-2-7b-chat-w4-g128-awq.pt \
#     --output_path quant_cache/llama-2-7b-chat-w4-g128-awq

# # Run the TinyChat demo in mem_efficient_load mode:
# python demo.py --model_type llama \
#     --model_path $MODEL_PATH/$MODEL_NAME \
#     --q_group_size 128 --load_quant quant_cache/llama-2-7b-chat-w4-g128-awq \
#     --precision W4A16 --mem_efficient_load
