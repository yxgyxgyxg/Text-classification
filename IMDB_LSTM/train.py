import json
import torch
import numpy as np
import random
import torch.nn as nn

from preprocessing import IMDBDataset, read_imdb_split, collate_imdb
from models import RNN


train_data_path = 'aclImdb/train'
test_data_path = 'aclImdb/test'
vocab_path = 'output/vocab.json'
word2idx = json.load(open(vocab_path, "rb"))
output_model_path = 'output/RNN.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vocab_size = 30004
batch_size = 64
hidden_size = 300
dropout = 0.5
epochs = 30
mode = 1  # Decide whether to train or test


def train(model):
    start_epoch = 0
    best_accuracy = 0
    for epoch in range(start_epoch, epochs):
        loss_total = 0
        for batch_index, batch in enumerate(trainloader):
            texts, labels, text_lengths = batch
            texts = texts.to(device)
            labels = labels.to(device)
            probs = model(texts)
            optimizer.zero_grad()
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.data.item()

        print(loss_total / float(len(trainloader)))
        curr_acc = evaluate_model(model, testloader, criterion)
        if curr_acc > best_accuracy:
            print("saving model...")
            with open(output_model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_accuracy = curr_acc
    print("Best accuracy :{0}".format(best_accuracy))


def evaluate_model(model, testloader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(testloader):
            texts, labels, text_lengths = batch
            texts = texts.to(device)
            labels = labels.to(device)
            probs = model(texts)
            loss = criterion(probs, labels)
            running_loss += loss.item()
            predictions = np.argmax(probs.data.cpu().numpy(), 1)
            correct += len(np.where(labels.data.cpu().numpy() == predictions)[0])
            total += texts.size(0)
        acc = correct / float(total)
        print("Accuracy:{0}".format(acc))
        return acc


if __name__ == '__main__':
    # 1.data load
    print('data load start')
    train_texts, train_labels = read_imdb_split(train_data_path)
    train_data = IMDBDataset(train_texts, train_labels, word2idx)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_imdb,
                                              shuffle=True)
    test_texts, test_labels = read_imdb_split(test_data_path)
    test_data = IMDBDataset(test_texts, test_labels, word2idx, attack_label=1)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_imdb,
                                             shuffle=False)
    print('data load end')

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    # 2.train or test
    criterion = nn.CrossEntropyLoss()
    if mode == 1:
        model = RNN(vocab_size=vocab_size, embedding_dim=300, hidden_dim=300, output_dim=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
        train(model)
    else:
        baseline_model = RNN(vocab_size=vocab_size, embedding_dim=300, hidden_dim=300, output_dim=2).to(device)
        baseline_model.load_state_dict(torch.load(output_model_path))
        baseline_model.eval()  # 开启eval状态，不再随机dropout
        evaluate_model(baseline_model, testloader, criterion)
