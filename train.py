import argparse
import torch
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from loaders.dataset import SentimentDataset
from trainer.trainer import Trainer
from model.model import CustomModel


def main(args):
    print("Args={}".format(str(args)))
    set_seed(args.seed)

    # Pre Setup
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_path)

    model.to(args.device)

    print(model)
    print(model.type)
    print("Vocab size: {}".format(len(tokenizer)))

    # Load data
    train_dataset = SentimentDataset(args, tokenizer=tokenizer, mode="train")
    test_dataset = SentimentDataset(args, tokenizer=tokenizer, mode="test")

    trainer = Trainer(args=args, model=model, train_dataset=train_dataset, test_dataset=test_dataset)

    if args.do_train:
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_train", required=True, type=str)
    parser.add_argument("--file_test", required=True, type=str)
    parser.add_argument("--pretrained_path", required=True, type=str)
    parser.add_argument("--do_train", required=True, type=str)
    parser.add_argument("--gpu_id", required=True, type=str)
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--num_train_epochs", required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--max_seq_len", required=True, type=int)
    parser.add_argument("--learning_rate", required=True, type=float)

    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--col_data", default="text", type=str)
    parser.add_argument("--col_label", default="sentiment", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_steps", default=2, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--save_steps", default=5, type=int)
    parser.add_argument("--early_stopping", default=25, type=int)
    parser.add_argument("--tuning_metric", default="loss", type=str)

    args = parser.parse_args()
    main(args)
