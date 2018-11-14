#! /bin/bash

for seed in "100" "200" "300"; do
    python train.py \
           --dataset CIFAR10 \
           --data_path ../data \
           --dir ./checkpoint/wage-qtorch/stochastic-GE \
           --model VGG7LP \
           --epochs=300 \
           --log-name wage-qtorch/wage/stochastic-GE \
           --wl-weight 2 \
           --weight-rounding nearest \
           --wl-grad 8 \
           --grad-rounding stochastic \
           --wl-activate 8 \
           --activate-rounding nearest \
           --wl-error 8 \
           --error-rounding stochastic \
           --wl-rand 16 \
           --seed ${seed} \
           --batch_size 128 \
           --qtorch;
done
