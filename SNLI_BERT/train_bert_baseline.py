import os
import torch
import random
import argparse

import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.utils.data as data

from transformers import BertConfig, BertTokenizer
from transformers.modeling_bert import BertModel, BertPreTrainedModel


class BertSnliClassificationModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSnliClassificationModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, int(config.hidden_size / 2)),
            nn.Linear(int(config.hidden_size / 2), self.num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = outputs[1]  # batch , hiddsize
        logits = self.classifier(pooler_output)  # batch,num_labels
        return logits


class BertSnliDataset(data.Dataset):
    def __init__(self, path="./data", train=True, bert_tokenizer=None):
        self.train = train
        self.root = path
        self.lowercase = True
        self.sentence_ids = {}

        self.train_data = []
        self.test_data = []
        self.train_path = os.path.join(path, 'train.txt')  # ./data/classifier/train.txt
        self.test_path = os.path.join(path, 'test.txt')
        self.labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        if os.path.exists(self.root + "/sent_ids.pkl"):
            self.sentence_ids = pkl.load(open(self.root + "/sent_ids.pkl", 'rb'))
        else:
            print("Sentence IDs not found!!")

        if self.train and os.path.exists(self.train_path):
            self.train_data = self.tokenize(self.train_path, bert_tokenizer)
        if (not self.train) and os.path.exists(self.test_path):
            self.test_data = self.tokenize(self.test_path, bert_tokenizer)

    def __getitem__(self, index):
        if self.train:
            return self.train_data[index]
        else:
            return self.test_data[index]

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def tokenize(self, path, bert_tokenizer):
        """Tokenizes a text file."""
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1

                tokens = line.strip().split('\t')
                label = self.labels[tokens[0]]  # 0 1 2
                premise = self.sentence_ids[tokens[1]]
                hypothesis = self.sentence_ids[tokens[2]]
                # if self.lowercase:
                #     hypothesis = hypothesis.strip().lower()
                #     premise = premise.strip().lower()

                encoded_inputs = bert_tokenizer(premise, hypothesis)  # 字典
                # print(encoded_inputs)

                input_ids, token_type_ids, attention_mask = encoded_inputs['input_ids'], encoded_inputs[
                    'token_type_ids'], encoded_inputs['attention_mask']

                lines.append([input_ids, token_type_ids, attention_mask, label])
                if len(lines) > 2000:
                    break
        print('样本个数：', len(lines))
        return lines


def collate_SNLI(batch):
    all_input_ids, all_token_type_ids, all_attention_mask, all_labels = [], [], [], []

    batch_length = []
    for b in batch:
        input_ids, token_type_ids, attention_mask, label = b
        batch_length.append(len(input_ids))

    max_batch_len = max(batch_length)
    for b in batch:
        input_ids, token_type_ids, attention_mask, label = b
        if len(input_ids) < max_batch_len:  # 如果长度不足，进行补充
            pad_len = max_batch_len - len(input_ids)
            input_ids = input_ids + [0] * pad_len
            token_type_ids = token_type_ids + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        all_input_ids.append(input_ids)
        all_token_type_ids.append(token_type_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(label)
    return torch.LongTensor(all_input_ids), torch.LongTensor(all_token_type_ids), torch.LongTensor(
        all_attention_mask), torch.LongTensor(all_labels)


def train(trainloader):
    loss_total = 0
    for batch_index, batch in enumerate(trainloader):
        input_ids, token_type_ids, attention_mask, labels = batch
        input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
            device), attention_mask.to(device), labels.to(device)

        logits = model(input_ids=input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask)

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()
        loss_total += loss.data.item()
        print('Single loss:', loss)
    print('Train avg loss:', loss_total / float(len(trainloader)))


def evaluate(testloader):
    running_loss = 0.0
    correct = 0
    total = 0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch_index, batch in enumerate(testloader):
            input_ids, token_type_ids, attention_mask, labels = batch
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device)

            logits = model(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)

            loss = criterion(logits, labels)
            running_loss += loss.item()

            predictions = np.argmax(logits.data.cpu().numpy(), 1)
            correct += len(np.where(labels.data.cpu().numpy() == predictions)[0])
            total += input_ids.size(0)
        acc = correct / float(total)
        print("Accuracy:{0}".format(acc))
        return acc


def param():
    parser = argparse.ArgumentParser(description='PyTorch baseline for Text')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='location of the data corpus')
    parser.add_argument('--model_type', type=str, default="bert",
                        help='location of the data corpus')
    parser.add_argument('--epochs', type=int, default=30,
                        help='maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-04,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1111,
                        help='seed')
    parser.add_argument('--save_path', type=str, default='./models',
                        help='used for saving the models')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='隐藏层的维度')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='label个数')
    parser.add_argument('--config_name', type=str, default='./bert-base-uncased/config.json',
                        help='bert config路径')
    parser.add_argument('--model_name_or_path', type=str, default='./bert-base-uncased',
                        help='bert 路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = param()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bert_tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    corpus_train = BertSnliDataset(train=True, path=args.data_path, bert_tokenizer=bert_tokenizer)
    corpus_test = BertSnliDataset(train=False, path=args.data_path, bert_tokenizer=bert_tokenizer)

    trainloader = torch.utils.data.DataLoader(corpus_train, batch_size=args.batch_size, collate_fn=collate_SNLI,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(corpus_test, batch_size=args.batch_size, collate_fn=collate_SNLI,
                                             shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    config = BertConfig.from_pretrained(args.config_name, num_labels=args.num_labels)
    model = BertSnliClassificationModel.from_pretrained(args.model_name_or_path, config=config).to(device)

    bert_param = list(model.bert.parameters())
    linear_param = list(model.classifier.parameters())

    optimizer = torch.optim.Adam([
        {'params': bert_param, 'lr': 2e-5},
        {'params': linear_param, 'lr': args.lr}],
        lr=args.lr, )

    best_accuracy = 0

    for epoch in range(args.epochs):
        # 1.训练
        model.train()
        train(trainloader)

        # 2.测试
        model.eval()
        curr_acc = evaluate(testloader)

        # 3.保存精度最优的模型
        if curr_acc > best_accuracy:
            print("Saving model...")
            with open(args.save_path + "/" + args.model_type + '.pt', 'wb') as f:
                torch.save(model.state_dict(), f)
            best_accuracy = curr_acc
        print("Best accuracy :{0}".format(best_accuracy))
