#!/usr/bin/env bash
#Convolution Layerの可視化スクリプト
#GradCAM専用のフォワード定義が必要

#作りかけてそのまま

input="input_foo.png"
model="model_file_path.npz"
layer_name="conv5_3"
labelID=-1  # -1: Replace label from top1 index
class_num=1000
save_dir="Result"
save_name="output"
prefix="png"
gpu=0


mkdir -p -m +w $save_dir

python gradcam.py \
--input "$input" \
--model "$model" \
--layer $layer_name \
--label $labelID \
--class_num $class_num \
--save "$save_dir/$save_name" \
--prefix $prefix \
--gpu $gpu 


