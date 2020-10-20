import json
import nltk
import torch.utils.data as data
import torch

from pathlib import Path


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            print(text_file)
            texts.append(text_file.read_text(encoding='utf-8'))
            labels.append(0 if label_dir is "neg" else 1)
    return texts, labels


def make_dictionary(texts, max_vocabulary_size):
    import collections
    count = [('<oov>', -1)]
    min_occurrence = 0
    text_words = []
    for i in range(len(texts)):
        text_word = nltk.word_tokenize(texts[i])
        text_words += text_word
    count.extend(collections.Counter(text_words).most_common(max_vocabulary_size))
    for i in range(len(count) - 1, -1, -1):
        if count[i][1] < min_occurrence:
            count.pop(i)
        else:
            # The collection is ordered, so stop when 'min_occurrence' is reached.
            break
    word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    for i, (word, _) in enumerate(count):
        word2idx[word] = i + 3
    id2word = dict(zip(word2idx.values(), word2idx.keys()))
    with open('./output/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(word2idx, f)
    return word2idx, id2word


class IMDBDataset(data.Dataset):
    def __init__(self, texts, labels, word2idx, attack_label=None):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.labelsdict = {'negative': 0, 'positive': 1}
        self.data = self.tokenizer(attack_label)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def tokenizer(self, attack_label=None):
        data = []
        for i in range(len(self.texts)):
            text_word = nltk.word_tokenize(self.texts[i])
            text_indices = [self.word2idx['<sos>']] + [self.word2idx.get(w, 3) for w in text_word] + [
                self.word2idx['<eos>']]

            label = self.labels[i]
            if attack_label is None:
                data.append([text_indices, label, len(text_indices)])
            else:
                if label == attack_label:
                    data.append([text_indices, label, len(text_indices)])
        return data


# 批量化数据
def collate_imdb(batch):
    texts = []
    label = []
    texts_len = []
    for b in batch:
        p, l, len_pre = b
        texts.append(p)
        label.append(l)
        texts_len.append(len_pre)
    max_text_length = max([len(pre) for pre in texts])
    text_indices = [pre + [0] * (max_text_length - len(pre)) for pre in texts if len(pre) <= max_text_length]
    return torch.LongTensor(text_indices), torch.LongTensor(label), torch.LongTensor(texts_len)


train_texts, train_labels = read_imdb_split('aclImdb/train')
make_dictionary(train_texts,30000)

