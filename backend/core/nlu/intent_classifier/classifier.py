from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
from torch.utils.data.dataset import random_split
from backend.core.nlu.skill_data_translator import SkillsTranslator
from backend.core.nlu.encoder.bert_encoder import SentenceEncoder
from backend.configuration.config import NLU_INTENTS_CONFIG, device
import torch
import torch.nn as nn
import torch.optim as optim

from .dense_model import DenseClassifierModel

encoder = SentenceEncoder()

class IntentDataset(Dataset):
    def __init__(self, pandas_df=None):
        if pandas_df is None:
            trans = SkillsTranslator(augment_intents=False)
            df = trans.build_dataset()
        elif (isinstance(pandas_df, str)):
            df = pd.read_csv(pandas_df)
        else:
            df = pandas_df
        self.data = (df['intent_label'].to_list(), encoder.encode(df['words'].to_list(), just_embeddings=True)['embeddings'])
        self.class2id = {j : i for (i, j) in enumerate(df['intent_label'].unique())}

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return (self.class2id[self.data[0][idx]], self.data[1][idx])


class IntentClassifier:
    def __init__(self, module_name="intent_classifier"):
        self.module_name = module_name
        self.fitted = False
        self.model = None

    def _load_base_model(self):
        model = DenseClassifierModel(embedding_dim=encoder.emb_dim, intent_num_labels=self.num_classes,
                                     dropout_prob=NLU_INTENTS_CONFIG['train_dropout_prob'], from_logits=False).to(device)
        return model

    def iterate_train_epoch(self, torch_dataset, criterion, optimizer, scheduler):
        # Train the model
        train_loss = 0
        train_acc = 0

        dataloader = DataLoader(torch_dataset, batch_size=NLU_INTENTS_CONFIG["train_batch_size"], shuffle=True)

        for i, (cls, seq) in enumerate(dataloader):
            optimizer.zero_grad()
            seq, cls = seq.to(device), cls.to(device)

            output = self.model(seq)
            loss = criterion(output, cls)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_acc += (output.argmax(1) == cls).sum().item()

        # Adjust the learning rate
        scheduler.step()
        return train_loss / len(torch_dataset), train_acc / len(torch_dataset)

    def iterate_test_epoch(self, torch_dataset, criterion):
        loss = 0
        acc = 0
        dataloader = DataLoader(torch_dataset, batch_size=NLU_INTENTS_CONFIG["train_batch_size"], shuffle=True)

        for i, (cls, seq) in enumerate(dataloader):
            seq, cls = seq.to(device), cls.to(device)
            with torch.no_grad():
                output = self.model(seq)
                loss = criterion(output, cls)
                loss += loss.item()
                acc += (output.argmax(1) == cls).sum().item()

        return loss / len(torch_dataset), acc / len(torch_dataset)

    def adapt_parameters_update(self, class2id):
        self.class2id = class2id
        self.id2class = {self.class2id[j]: j for j in self.class2id.keys()}
        self.num_classes = len(self.id2class)


    def train(self):
        train_dataset = IntentDataset()
        self.adapt_parameters_update(train_dataset.class2id)
        self.model = self._load_base_model()

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(self.model.parameters(), lr=NLU_INTENTS_CONFIG["train_learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

        train_len = int(len(train_dataset) * (1 - NLU_INTENTS_CONFIG["train_validate_prob"]))
        sub_train, sub_valid = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

        for epoch in range(NLU_INTENTS_CONFIG["train_num_epochs"]):
            start_time = time.time()
            train_loss, train_acc = self.iterate_train_epoch(sub_train, criterion, optimizer, scheduler)
            valid_loss, valid_acc = self.iterate_test_epoch(sub_valid, criterion)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
        self.fitted = True

    def validate(self, examples):
        if self.fitted:
            seq = encoder.encode(examples, True)['embeddings'].to(device)
            outputs = []
            with torch.no_grad():
                for i in seq:
                    output = self.model(i)
                    id = torch.argmax(output.cpu().detach()).item()
                    outputs.append(self.id2class[id])
            return outputs
        else:
            raise NotImplementedError


if __name__ == "__main__":
    obj = IntentClassifier()
    obj.train()
    print(obj.validate(["hello, please enable the lights in the attic"]))