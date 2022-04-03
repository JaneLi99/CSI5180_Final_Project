# Jiaming Li
# University of Ottawa
# CSI 5180 Topics in AI - Virtual Assistant
# Final Project
# HOTPOTQA baseline

import torch
import json
from torchtext.legacy import data
import torch.nn as nn
import torch.optim as optim
import time
from torch.nn import Embedding

import random
import math

import matplotlib.pyplot as plt

def get_examples(file):
    ak = json.load(open(file))
    examples = []
    for j, i in enumerate(ak):
        context = "".join([k for j in i['context'] for k in j[1]])
        question = i['question']
        answer = i['answer']
        examples.append([context, question, answer])
    return examples

def get_examples(file):
    ak = json.load(open(file))
    examples = []
    for j, i in enumerate(ak):
        # Limiting examples coz ram not sufficient. find another way..some sort of yield
        if len(examples) > 50000:
          break
        context = "".join([k for j in i['context'] for k in j[1]])
        question = i['question']
        answer = i['answer']
        examples.append([context + question_pad + question, answer])
    return examples

def get_data(train_file, test_file):
    train_examples = get_examples(train_file)
    test_examples = get_examples(test_file)

    context_with_question = data.Field(sequential = True, tokenize = 'spacy', init_token = '<sos>', eos_token = '<eos>')
    answer = data.Field(sequential = True, tokenize = 'spacy', init_token = '<sos>', eos_token = '<eos>')
    fields = [('context', context_with_question), ('answer', answer)]
    train_Examples = [data.Example.fromlist([i[0], i[1]], fields) for i in train_examples]
    train_dataset = data.Dataset(train_Examples, fields)

    test_Examples = [data.Example.fromlist([i[0], i[1]], fields) for i in test_examples]
    test_dataset = data.Dataset(test_Examples, fields)

    context_with_question.build_vocab(train_dataset, min_freq = 2, max_size = 30000,vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)
    answer.vocab = context_with_question.vocab
    return context_with_question, answer, train_dataset, test_dataset

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)  # no dropout as only one layer!
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)  # no cell state!
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)
        prediction = self.fc_out(output)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        context = self.encoder(src)
        hidden = context
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        if not isinstance(m, Embedding):
            nn.init.normal_(param.data, mean=0, std=0.01)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.context
        trg = batch.answer
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.context
            trg = batch.answer
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def model_training(epochs):
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, test_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        train_loss_PPL = math.exp(train_loss)
        valid_loss_PPL = math.exp(valid_loss)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_loss_PPL:7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_loss_PPL:7.3f}')

        return train_loss, valid_loss, train_loss_PPL, valid_loss_PPL

if __name__ == "__main__":
    question_pad = ' @qpad '

    BATCH_SIZE = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_file = "data/hotpot_train_v1.1.json"
    test_file = "data/hotpot_dev_fullwiki_v1.json"
    context_with_question, answer, train_dataset, test_dataset = get_data(train_file, test_file)
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_dataset, test_dataset), batch_size = BATCH_SIZE,
        sort_key = lambda x: len(x.context), sort_within_batch = True, device = device)

    INPUT_DIM = len(context_with_question.vocab)
    OUTPUT_DIM = len(context_with_question.vocab)
    ENC_EMB_DIM = 100
    DEC_EMB_DIM = 100
    HID_DIM = 200
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = answer.vocab.stoi[answer.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    epochs = 10
    CLIP = 1

    train_loss_list = []
    valid_loss_list = []
    train_loss_PPL_list = []
    valid_loss_PPL_list = []
    epoch_list = [i for i in range(1, epochs + 1)]

    for epoch in range(epochs):
        train_loss, valid_loss, train_loss_PPL, valid_loss_PPL = model_training(epochs)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_loss_PPL_list.append(train_loss_PPL)
        valid_loss_PPL_list.append(valid_loss_PPL)

    # Training & Testing Accuracy Plot
    plt.plot(epoch_list, train_loss_list, color = 'tomato', label = 'Train Loss')
    plt.plot(epoch_list, valid_loss_list, color = 'limegreen', label = 'Valid Loss')
    plt.legend(loc = 'lower left')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Valid Loss of HOTPOTQA Dataset')
    plt.show()

    # Training & Testing Loss Plot
    plt.plot(epoch_list, train_loss_PPL_list, color = 'tomato', label = 'Train Loss PPL')
    plt.plot(epoch_list, valid_loss_PPL_list, color = 'limegreen', label = 'Valid Loss PPL')
    plt.legend(loc = 'lower left')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Valid Loss PPL of HOTPOTQA Dataset')
    plt.show()









