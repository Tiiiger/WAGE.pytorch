#! /bin/bash

python train.py \
       --dataset CIFAR10 \
       --data_path ../data \
       --dir ./checkpoint/wage-modify/ceil_shift \
       --model VGG7LP \
       --epochs=300 \
       --log-name wage-modify/ceil_shift/ \
       --wl-weight 2 \
       --wl-grad 8 \
       --wl-activate 8 \
       --wl-error 8 \
       --wl-rand 16 \
       --seed $1 \
       --batch_size 128;
