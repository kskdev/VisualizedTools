#!/usr/bin/env bash
#Convolution Layerの可視化スクリプト
#GradCAM専用のフォワード定義が必要
input="./dog_cat.png"
model="VGG16のモデルファイル.npz"
layer_name="conv5_3"
labelID=-1
class_num=1000
save_dir="Result"
save_name="res"
prefix="png"
gpu=0

mkdir -p -m +w $save_dir

python gradcam.py \
--input "$input" \
--model "$model" \
--gpu $gpu \
--label $labelID \
--class_num $class_num \
--layer $layer_name \
--save "$save_dir/$save_name" \
--prefix $prefix
