#!/bin/bash

GPU_ID=3
MODEL=resnet101

for mixed_precision in no fp16
do
    for num_workers in 1 2 4 8
    do
        for batch_size in 32 64 128 256
        do
            for gradient_accumulation_steps in 1 2 3 8
            do
                accelerate launch \
                    --gpu_ids $GPU_ID \
                    --mixed_precision $mixed_precision \
                    main.py \
                    --num_workers $num_workers \
                    --batch_size $batch_size \
                    --gradient_accumulation_steps $gradient_accumulation_steps \
                    --mixed_precision $mixed_precision \
                    --model $MODEL
            done
        done
    done
done