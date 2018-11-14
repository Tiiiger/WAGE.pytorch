#! /bin/bash

python train.py \
    --dataset CIFAR10 \
    --data_path ../data \
    --dir ./checkpoint/wage-original/doubleC \
    --model VGG7LP \
    --epochs=300 \
    --log-name wage-original/doubleC/ \
    --wl-weight 2 \
    --wl-grad 8 \
    --wl-activate 8 \
    --wl-error 8 \
    --wl-rand 16 \
    --seed 100 \
    --batch_size 128;
