import json
import torch
import pickle
import random
import numpy as np
import torch.nn as nn

from preprocessing import SnliDataset, transform_data, collate_snli
from models import build_embedding_matrix, ESIM

train_data_path = 'snli_1.0/snli_1.0_train.jsonl'
test_data_path = 'snli_1.0/snli_1.0_test.jsonl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vocab_path = 'output/vocab.json'
word2idx = json.load(open(vocab_path, "rb"))
output_model_path = 'output/ESIM.pt'
vocab_size = 30004
batch_size = 32
hidden_size = 300
dropout = 0.5
epochs = 5
mode = 1 # Decide whether to train or test


def train(model):
    start_epoch = 0
    max_gradient_norm = 10
    best_accuracy = 0
    for epoch in range(start_epoch, epochs):
        loss_total = 0
        for batch_index, batch in enumerate(trainloader):
            premises, hypotheses, labels, premises_lengths, hypotheses_lengths = batch
            premises = premises.to(device)
            hypotheses = hypotheses.to(device)
            hypotheses_lengths = hypotheses_lengths.to(device)
            premises_lengths = premises_lengths.to(device)
            labels = labels.to(device)
            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths)

            optimizer.zero_grad()
            loss = criterion(probs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()
            loss_total += loss.data.item()
            print(batch_index)
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
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch_index, batch in enumerate(testloader):
            premises, hypotheses, labels, premises_lengths, hypotheses_lengths = batch

            premises = premises.to(device)
            hypotheses = hypotheses.to(device)
            hypotheses_lengths = hypotheses_lengths.to(device)
            premises_lengths = premises_lengths.to(device)
            labels = labels.to(device)

            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths)
            loss = criterion(probs, labels)

            running_loss += loss.item()
            predictions = np.argmax(probs.data.cpu().numpy(), 1)
            correct += len(np.where(labels.data.cpu().numpy() == predictions)[0])
            total += premises.size(0)
        acc = correct / float(total)
        print("Accuracy:{0}".format(acc))
        return acc


if __name__ == '__main__':
    # 1.data load
    premises, hypotheses, labels = transform_data(train_data_path)
    train_data = SnliDataset(premises, hypotheses, labels, word2idx)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_snli,
                                              shuffle=True)

    premises, hypotheses, labels = transform_data(test_data_path)
    test_data = SnliDataset(premises, hypotheses, labels, word2idx, attack_label=0)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=256, collate_fn=collate_snli,
                                             shuffle=False)

    # embed_matrix = build_embedding_matrix('./embeddings/glove.840B.300d.txt',word2idx)
    # with open("./embeddings/embeddings.pkl", 'wb') as fo:     # 将数据写入pkl文件
    #     pickle.dump(embed_matrix, fo)

    embeddings_file = './embeddings/embeddings.pkl'
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    # 2. train or test
    criterion = nn.CrossEntropyLoss()
    if mode == 1:
        model = ESIM(embeddings.shape[0],
                     embeddings.shape[1],
                     hidden_size,
                     embeddings=embeddings,
                     dropout=dropout,
                     num_classes=3, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
        train(model)
    else:
        baseline_model = ESIM(vocab_size, 300, 300, num_classes=3, device=device).to(device)
        baseline_model.load_state_dict(torch.load(output_model_path))
        baseline_model.eval()  # 开启eval状态，不再随机dropout
        evaluate_model(baseline_model, testloader, criterion)
