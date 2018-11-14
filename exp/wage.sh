#! /bin/bash

for seed in "100" "200" "300"; do
    python train.py \
           --dataset CIFAR10 \
           --data_path ../data \
           --dir ./checkpoint/wage-replicate/sgd \
           --model VGG7LP \
           --epochs=300 \
           --log-name wage-replicate/wage/ \
           --wl-weight 2 \
           --wl-grad 8 \
           --wl-activate 8 \
           --wl-error 8 \
           --wl-rand 16 \
           --seed ${seed} \
           --batch_size 128;
done
