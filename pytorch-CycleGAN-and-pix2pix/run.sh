#!/bin/bash
python train.py --dataroot /local/data2/jakli758/dataset/gan/128 --name 0216_no_preprocessing --model cycle_gan \
    --batch_size=8 \
    --preprocess none
    #--n_epochs=200 \
    #--n_epochs_decay=200 \
    
