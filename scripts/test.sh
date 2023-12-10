#!/bin/bash

accelerate launch \
    --mixed_precision no \
    main.py \
    --batch_size 32 \
    --model resnet18 \
    --num_workers 8 \
    --gradient_accumulation_steps 1 \
    --mixed_precision no