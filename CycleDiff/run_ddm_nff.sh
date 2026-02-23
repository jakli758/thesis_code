#!/bin/bash
accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision fp16 \
    train_uncond_ldm.py \
    --cfg ./configs/nff2aff/nff_ddm_128x128.yaml 
