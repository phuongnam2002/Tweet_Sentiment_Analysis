import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from utils.utils import load_data, clear_text


def convert_text_to_feature(text, tokenizer, max_seq_len):
    unk_token = tokenizer.unk_token
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token_id

    special_tokens = 2

    text = text.split()
    tokens = []
    for word in text:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]
        tokens.extend(word_tokens)

    # Truncate data
    if len(tokens) > max_seq_len - special_tokens:
        tokens = tokens[: (max_seq_len - special_tokens)]

    tokens = [cls_token] + tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    padding_length = max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = [0] * len(input_ids)

    assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(
        len(input_ids), max_seq_len
    )

    return input_ids, attention_mask, token_type_ids


class SentimentDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super().__init__()
        self.args = args

        if mode == "train":
            self.data = load_data(args.file_train, col=args.col_data)
            self.labels = load_data(args.file_train, col=args.col_label)
        else:
            self.data = load_data(args.file_test, col=args.col_data)
            self.labels = load_data(args.file_test, col=args.col_label)
        self.label_to_idx = LabelEncoder().fit(["negative", "neutral", "positive"])

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data[index]
        label = self.labels[index]
        sentence = clear_text(sentence)
        label = self.label_to_idx.transform([label])

        input_ids, attention_mask, token_type_ids = convert_text_to_feature(text=sentence, tokenizer=self.tokenizer,
                                                                            max_seq_len=self.args.max_seq_len)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(token_type_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )
