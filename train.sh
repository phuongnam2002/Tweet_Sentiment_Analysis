export MODEL_DIR=/home/black/Kaggle
echo "${MODEL_DIR}"
CUDA_VISIBLE_DEVICES=2 python train.py \
                       --file_train train.csv \
                       --file_test test.csv \
                       --pretrained_path ProsusAI/finbert \
                       --model_dir $MODEL_DIR \
                       --do_train True\
                       --gpu_id 2 \
                       --num_train_epochs 10 \
                       --batch_size 256 \
                       --max_seq_len 32 \
                       --learning_rate 1e-5
