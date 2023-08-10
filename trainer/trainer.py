import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
from lion_pytorch import Lion
from transformers.optimization import get_scheduler
from sklearn.utils.class_weight import compute_class_weight

from utils.utils import load_data
from utils.earlystopping import EarlyStopping


class Trainer:
    def __init__(self, args, model, train_dataset, test_dataset):
        self.args = args

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.labels = load_data(args.file_train, col=args.col_label)
        self.softmax = nn.Softmax()
        self.model = model

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size
        )

        if self.args.max_steps > 0:
            total = self.args.max_steps
            self.args.num_train_epochs = (
                    self.args.max_steps // (len(train_loader) // self.args.gradient_accumulation_steps)
                    + 1
            )

        else:
            total = (
                    len(train_loader)
                    // self.args.gradient_accumulation_steps
                    * self.args.num_train_epochs
            )

        optimizer = self.get_optimizer()

        scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total
        )
        class_wts = compute_class_weight('balanced', np.array(self.labels[:self.args.batch_size]),
                                         np.array(self.labels[:self.args.batch_size]))
        weights = torch.tensor(class_wts, dtype=torch.float).to(self.args.device)

        criterion = nn.CrossEntropyLoss(weight=weights)

        print('__________TRAINING__________')
        print('  Num examples = ', len(self.train_dataset))
        print('  Num Epochs = ', self.args.num_train_epochs)
        print('  Total trainer batch size = ', self.args.batch_size)
        print('  Gradient Accumulation steps = ', self.args.gradient_accumulation_steps)
        print('  Total optimization steps = ', total)
        print('  Save steps = ', self.args.save_steps)
        print('  Running on device = ', self.args.device)

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for _ in train_iterator:
            self.model.train()
            epoch_iterator = tqdm(train_loader, desc="Iteration", position=0, leave=True)
            print(f"Epoch {_}")

            for id, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.args.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
                labels = torch.squeeze(batch[3]).float()

                logits = self.model(**inputs).logits
                logits = torch.argmax(logits, dim=1).float()

                loss = criterion(self.softmax(logits), self.softmax(labels))
                loss.requires_grad_(True)

                print(f"Train Loss: {loss.item()}")

                # backpropagation
                loss.backward()

                if (id + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    # update weight
                    optimizer.step()

                    self.model.zero_grad()
                    optimizer.zero_grad()

                    self.eval()
                    early_stopping(loss.item(), self.model, self.args)
                    if early_stopping.early_stop:
                        print("Early Stopping")
                        break

    def eval(self):
        accuracy = 0
        for id, batch in tqdm(enumerate(self.test_dataset)):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs = {
                "input_ids": batch[0].unsqueeze(0),
                "attention_mask": batch[1].unsqueeze(0),
                "token_type_ids": batch[2].unsqueeze(0),
            }
            with torch.no_grad():
                logits = self.model(**inputs).logits[0]

            logits = torch.argmax(logits).item()

            if logits == batch[3].item():
                accuracy += 1

        print('Accuracy = ', accuracy / len(self.test_dataset))

    def get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)

        return optimizer
