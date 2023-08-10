export MODEL_DIR=/home/black/Kaggle
echo "${MODEL_DIR}"
CUDA_VISIBLE_DEVICES=2 python test.py \
                       --file_train train.csv \
                       --file_test test.csv \
                       --pretrained_path cardiffnlp/twitter-xlm-roberta-base-sentiment \
                       --model_dir $MODEL_DIR \
                       --gpu_id 2 \
                       --num_train_epochs 10 \
                       --batch_size 128 \
                       --max_seq_len 32 \
                       --learning_rate 5e-5
