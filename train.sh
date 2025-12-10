#!/bin/bash

MODEL_PATH=$(CUDA_VISIBLE_DEVICES=0 python train_stage1.py --log stage1 --pretrained_model Saved/pretrained.pth | tee /dev/tty | tail -n 1 | sed 's/last.pth/best.pth/' | sed 's/save model to //')
CUDA_VISIBLE_DEVICES=0 python train_stage2.py --log stage2 --pretrained_model $MODEL_PATH
