$env:ML_DATA="./data_ssl"
# CUDA_VISIBLE_DEVICES=0 
$env:CUDA_VISIBLE_DEVICES=0 
# python test.py
#

# python mixmatch.py --helpfull
# python mixmatch.py --filters=32 --dataset=cifar10.3@500-5000 --w_match=75 --beta=0.75 --arch cnn13  --whiten
# python mixmatch.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.3@40-50 --w_match=75 --beta=0.75 --whiten --arch resnet18  --batch 64
python test.py
