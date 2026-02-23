#!/bin/bash
accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision fp16 \
    train_uncond_ldm_cycle.py \
    --cfg ./configs/nff2aff/translation_C_disc_timestep_ode_2.yaml
