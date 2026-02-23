#!/bin/bash
accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision fp16 \
    train_vae.py \
    --cfg ./configs/nff2aff/nff_ae_128x128.yaml 
    
accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision fp16 \
    train_vae.py \
    --cfg ./configs/nff2aff/aff_ae_128x128.yaml