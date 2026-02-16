#!/bin/bash
python train.py --dataroot /local/data2/jakli758/dataset/gan/128 --name more_epochs --model cycle_gan \
    --batch_size=8 \
    --n_epochs=200 \
    --n_epochs_decay=200 \
