$env:ML_DATA="./data_ssl"
$ML_DATA="./data_ssl"
python ./scripts/create_txt_datasets.py
# python ./scripts/create_datasets.py
# $ML_DATA="./data_ssl"

foreach ( $seed in 1, 2, 3, 4, 5){
    foreach ( $size in 40, 100){
        python ./scripts/create_txt_split.py --seed=$seed --size=$size $ML_DATA/SSL/miniimagenet $ML_DATA/miniimagenet-train.txt
    }
}
