import json
import codecs
import torch
import torch.utils.data as data


def transform_data(in_path):
    print("Loading", in_path)

    premises = []
    hypotheses = []
    labels = []

    with codecs.open(in_path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)

            # load premise
            raw_premise = loaded_example['sentence1_binary_parse'].split(" ")
            premise_words = []
            # loop through words of premise binary parse
            for word in raw_premise:
                # don't add parse brackets
                if word != "(" and word != ")":
                    premise_words.append(word)
            premise = " ".join(premise_words)

            # load hypothesis
            raw_hypothesis = \
                loaded_example['sentence2_binary_parse'].split(" ")
            hypothesis_words = []
            for word in raw_hypothesis:
                if word != "(" and word != ")":
                    hypothesis_words.append(word)
            hypothesis = " ".join(hypothesis_words)

            raw_label = \
                loaded_example['gold_label']
            if raw_label != '-':
                premises.append(premise.lower())
                hypotheses.append(hypothesis.lower())
                labels.append(raw_label)

    return premises, hypotheses, labels


def make_dictionary(premises, hypotheses):
    import collections
    count = [('<oov>', -1)]
    max_vocabulary_size = 30000
    min_occurrence = 0
    premises_hypotheses_words = []
    for i in range(len(premises)):
        premise_text_words = premises[i].split(' ')
        hypotheses_text_words = hypotheses[i].split(' ')
        premises_hypotheses_words += premise_text_words
        premises_hypotheses_words += hypotheses_text_words
    count.extend(collections.Counter(premises_hypotheses_words).most_common(max_vocabulary_size))
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

    data = list()
    unk_count = 0
    for word in premises_hypotheses_words:
        # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary.
        index = word2idx.get(word, 3)
        if index == 3:
            unk_count += 1
        data.append(index)
    count[0] = ('<oov>', unk_count)

    with open('./output/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(word2idx, f)

    return word2idx, id2word


class SnliDataset(data.Dataset):

    def __init__(self, premises, hypotheses, labels, word2idx, attack_label=None):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.word2idx = word2idx
        self.labelsdict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.data = self.tokenizer(attack_label)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def tokenizer(self, attack_label=None):
        data = []
        for i in range(len(self.premises)):
            # single_sentence=[]

            premise_words = self.premises[i].split(' ')
            premise_indices = [self.word2idx['<sos>']] + [self.word2idx.get(w, 3) for w in premise_words] + [
                self.word2idx['<eos>']]
            hypothese_words = self.hypotheses[i].split(' ')
            hypothese_indices = [self.word2idx['<sos>']] + [self.word2idx.get(w, 3) for w in hypothese_words] + \
                                [self.word2idx['<eos>']]
            label = self.labelsdict.get(self.labels[i])

            if attack_label is None:
                data.append([premise_indices, hypothese_indices, label, len(premise_indices), len(hypothese_indices)])
            else:
                if label == attack_label:
                    data.append(
                        [premise_indices, hypothese_indices, label, len(premise_indices), len(hypothese_indices)])
        return data


premises, hypotheses, labels = transform_data('snli_1.0/snli_1.0_train.jsonl')
make_dictionary(premises, hypotheses)


def collate_snli(batch):
    premise_indices = []
    hypothese_indices = []
    label = []
    premise_indices_len = []
    hypothese_indices_len = []
    for b in batch:
        p, h, l, len_pre, len_hyp = b
        premise_indices.append(p)
        hypothese_indices.append(h)
        label.append(l)
        premise_indices_len.append(len_pre)
        hypothese_indices_len.append(len_hyp)
    max_pre_length = max([len(pre) for pre in premise_indices])
    max_hyp_length = max([len(hyp) for hyp in hypothese_indices])
    premise_indices = [pre + [0] * (max_pre_length - len(pre)) for pre in premise_indices if len(pre) <= max_pre_length]
    hypothese_indices = [hyp + [0] * (max_hyp_length - len(hyp)) for hyp in hypothese_indices if
                         len(hyp) <= max_hyp_length]
    return torch.LongTensor(premise_indices), torch.LongTensor(hypothese_indices), torch.LongTensor(
        label), torch.LongTensor(premise_indices_len), torch.LongTensor(hypothese_indices_len)
