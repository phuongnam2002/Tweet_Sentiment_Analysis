import torch.nn as nn
from transformers import (
    RobertaPreTrainedModel,
    XLMPreTrainedModel,
    PretrainedConfig,
    AutoModel
)


class CustomModel(RobertaPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.model = AutoModel.from_config(config=config)

        self.dropout = nn.Dropout(p=0.1)

        self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(768, 512)  # 768 is hidden size BERT
        self.fc2 = nn.Linear(512, 3)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, **kwargs):
        pooler_output = self.model(**kwargs).pooler_output

        pooler_output = self.fc1(pooler_output)
        pooler_output = self.activation(pooler_output)
        pooler_output = self.dropout(pooler_output)
        pooler_output = self.fc2(pooler_output)
        pooler_output = self.softmax(pooler_output)

        return pooler_output
