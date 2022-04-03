# Jiaming Li
# University of Ottawa
# CSI 5180 Topics in AI - Virtual Assistant
# Final Project
# HOTPOTQA with Attention

import torch
import json
from torchtext.legacy import data
import torch.nn as nn
import torch.optim as optim
import time
from torch.nn import Embedding

import random
import math
import torch.nn.functional as F

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
        if len(examples)>50000:
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

    context_with_question.build_vocab(train_dataset, min_freq = 2, max_size = 30000)
    answer.vocab = context_with_question.vocab

    return context_with_question, answer, train_dataset, test_dataset


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        if not isinstance(m, Embedding):
            nn.init.normal_(param.data, mean = 0, std = 0.01)

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

    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_file = "data/hotpot_train_v1.1.json"
    test_file = "data/hotpot_dev_fullwiki_v1.json"
    context_with_question, answer, train_dataset, test_dataset = get_data(train_file, test_file)
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_dataset, test_dataset), batch_size = BATCH_SIZE,
        sort_key = lambda x: len(x.context), sort_within_batch = True, device = device)

    INPUT_DIM = len(context_with_question.vocab)
    OUTPUT_DIM = len(context_with_question.vocab)
    ENC_EMB_DIM = 50
    DEC_EMB_DIM = 50
    ENC_HID_DIM = 50
    DEC_HID_DIM = 50
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)

    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = answer.vocab.stoi[answer.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    epochs = 10
    CLIP = 1

    train_loss_list_with_A = []
    valid_loss_list_with_A = []
    train_loss_PPL_list_with_A = []
    valid_loss_PPL_list_with_A = []
    epoch_list = [i for i in range(1, epochs + 1)]

    for epoch in range(epochs):
        train_loss, valid_loss, train_loss_PPL, valid_loss_PPL = model_training(epochs)

        train_loss_list_with_A.append(train_loss)
        valid_loss_list_with_A.append(valid_loss)
        train_loss_PPL_list_with_A.append(train_loss_PPL)
        valid_loss_PPL_list_with_A.append(valid_loss_PPL)

    # Training & Testing Accuracy Plot
    plt.plot(epoch_list, train_loss_list_with_A, color = 'tomato', label = 'Train Loss')
    plt.plot(epoch_list, valid_loss_list_with_A, color = 'limegreen', label = 'Valid Loss')
    plt.legend(loc = 'lower left')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Valid Loss of HOTPOTQA Dataset with Attention')
    plt.show()

    # Training & Testing Loss Plot
    plt.plot(epoch_list, train_loss_PPL_list_with_A, color = 'tomato', label = 'Train Loss PPL')
    plt.plot(epoch_list, valid_loss_PPL_list_with_A, color = 'limegreen', label = 'Valid Loss PPL')
    plt.legend(loc = 'lower left')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Valid Loss PPL of HOTPOTQA Dataset with Attention')
    plt.show()



