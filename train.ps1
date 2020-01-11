$env:ML_DATA="./data_ssl"
# CUDA_VISIBLE_DEVICES=0 
$env:CUDA_VISIBLE_DEVICES=0 
# python test.py
#

# python mixmatch.py --helpfull
# python mixmatch.py --filters=32 --dataset=cifar10.3@500-5000 --w_match=75 --beta=0.75 --arch cnn13  --whiten
# python mixmatch.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.1@40-50 --w_match=75 --beta=0.0 --whiten --arch resnet18  --batch 64 --epochs 400 --decay_start_epoch 50 
# python deeplp.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.2@40-50 --w_match=75 --beta=0.4 --whiten --arch resnet18  --batch 64 --epochs 400 --decay_start_epoch 50 

# python mixup_sl.py --filters=128 --dataset=cifar10.3@500-5000 --beta=0.75 --arch cnn13  --whiten

python mixup_sl.py --scales 3 --repeat 2 --filters=128 --dataset=miniimagenet-50 --beta=0.2 --whiten --arch resnet18  --batch 64 --epochs 400 --decay_start_epoch 50 
python mixup_sl.py --scales 3 --repeat 2 --filters=128 --dataset=miniimagenet-50 --beta=0.4 --whiten --arch resnet18  --batch 64 --epochs 400 --decay_start_epoch 50 

# python mixmatch.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.1@40-50 --w_match=75 --beta=0.2 --whiten --arch resnet18  --batch 64 --epochs 400 --decay_start_epoch 50 
# python mixmatch.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.1@40-50 --w_match=75 --beta=2.0 --whiten --arch resnet18  --batch 64 --epochs 400 --decay_start_epoch 50 
# python mixmatch.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.1@40-50 --w_match=75 --beta=4.0 --whiten --arch resnet18  --batch 64 --epochs 400 --decay_start_epoch 50 
# python test.py

# python mixup_sl.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet-50 --beta=0.0 --whiten --arch resnet18 --epochs 400 --decay_start_epoch 50 --batch 64 
# python mixup_sl.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.3@40-50 --beta=1.0 --whiten --arch resnet18 --epochs 400 --decay_start_epoch 50 --batch 64 
# python mixup_sl.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.3@40-50 --beta=2.0 --whiten --arch resnet18 --epochs 400 --decay_start_epoch 50 --batch 64 
# python mixup_sl.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.3@40-50 --beta=0.0 --whiten --arch resnet18 --epochs 400 --decay_start_epoch 50 --batch 64  --lr 0.0002
# python .\test2.py

