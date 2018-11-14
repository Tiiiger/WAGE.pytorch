#! /bin/bash

python train.py \
       --dataset CIFAR10 \
       --data_path ../data \
       --dir ./checkpoint/wage-investigate/stochastic-G \
       --model VGG7LP \
       --epochs=20 \
       --log-name wage-investigate/wage/stochastic-G \
       --wl-weight 2 \
       --weight-rounding nearest \
       --wl-grad 8 \
       --grad-rounding stochastic \
       --wl-activate 8 \
       --activate-rounding nearest \
       --wl-error 8 \
       --error-rounding nearest \
       --wl-rand 16 \
       --seed 100 \
       --batch_size 128 \
       --qtorch;
