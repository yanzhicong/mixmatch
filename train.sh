export ML_DATA="./data_ssl"
# CUDA_VISIBLE_DEVICES=0 python mixmatch.py --filters=32 --dataset=cifar10.3@500-5000 --w_match=75 --beta=0.75 

# CUDA_VISIBLE_DEVICES=0 python mixmatch.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.3@40-50 --w_match=75 --beta=0.75 --whiten --arch resnet18  --batch 64 --lr 0.00002

# CUDA_VISIBLE_DEVICES=0 python mixup_sl.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.3@100-50 --beta=1.0 --whiten --arch resnet18  --batch 64 --lr 0.00002

# CUDA_VISIBLE_DEVICES=0 python mixup_sl.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.3@40-50 --beta=0.0 --whiten --arch resnet18 --decay_start_epoch 50 --batch 64  --lr 0.0002
CUDA_VISIBLE_DEVICES=0 python deeplp.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet.1@40-50 --w_match=75 --beta=0.0 --whiten --arch resnet18  --batch 64 --epochs 400 --decay_start_epoch 50 

# CUDA_VISIBLE_DEVICES=0 python mixup_sl.py --scales 4 --repeat 2 --filters=64 --dataset=miniimagenet-50 --beta=0.0 --whiten --arch resnet18  --batch 64 --lr 0.002
