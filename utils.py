import torch
import numpy as np
import re
import unicodedata
import random
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction
import torch.nn as nn


def train_handler(args, encoder, decoder, lang1_tokens, lang2_tokens, lang1, lang2):
    e_optimizer = torch.optim.SGD(encoder.parameters(), lr=args.lr)
    d_optimizer = torch.optim.SGD(decoder.parameters(), lr=args.lr)
    loss_function = nn.NLLLoss()
    loss = 0
    for i in tqdm(range(1, args.epochs)):
        x_tensor, y_tensor = fetch_random_tensor(lang1_tokens, lang2_tokens, lang1, lang2)
        loss += train(args,x_tensor, y_tensor, encoder, decoder, e_optimizer, d_optimizer, loss_function)
        if i% 1000 == 0:
            print(loss/1000)
            loss = 0


def train(args, x_tensor, y_tensor, encoder, decoder, e_optimizer, d_optimizer, loss_function):
    encoder_hidden = encoder.Hidden()
    loss = 0
    # initialize optimizer to zero gradients
    e_optimizer.zero_grad()
    d_optimizer.zero_grad()

    input_length = x_tensor.size(0)
    target_length = y_tensor.size(0)
    encoder_outs = torch.zeros(args.max_length, encoder.hidden_size)

    for j in range(input_length):
        e_out, encoder_hidden = encoder(x_tensor[j], encoder_hidden)
        encoder_outs[j] = e_out[0, 0]
    decoder_in = torch.tensor([[0]])
    decoder_hidden = encoder_hidden

    if random.random() > .5:
        teacher_forcing = True
    else:
        teacher_forcing = False

    if teacher_forcing:
        for j in range(target_length):
            decoder_out, decoder_hidden, attention = decoder(decoder_in, decoder_hidden, encoder_outs)
            loss += loss_function(decoder_out, y_tensor[j])
            decoder_in = y_tensor[j]
    else:
        for j in range(target_length):
            decoder_out, decoder_hidden, attention = decoder(decoder_in, decoder_hidden, encoder_outs)
            topv, topi = decoder_out.topk(1)
            decoder_in = topi.squeeze().detach()
            loss += loss_function(decoder_out, y_tensor[j])
            if decoder_in.item() == 1:
                break

    # Backpropogation
    loss.backward()

    # Optimize
    e_optimizer.step()
    d_optimizer.step()

    return loss.item() / target_length


def test(args, encoder, decoder, lang1, lang2, lang1_sentences, lang2_sentences):
    sentences = [unicodeToAscii(i).split(' ') for i in lang1_sentences]
    tokenized = [[lang1.word_dict_lookup(i) for i in j] for j in sentences]
    lang_2_sentences = [unicodeToAscii(i).split(' ') for i in lang2_sentences]
    tensors = [torch.tensor(i).view(-1, 1) for i in tokenized]
    bleu_scores = []
    for tensor, sentence in zip(tensors, lang_2_sentences):
        encoder_hidden = encoder.Hidden()
        input_length = tensor.size(0)
        output_length = len(sentence)
        encoder_outs = torch.zeros(args.max_length, encoder.hidden_size)
        for input in range(input_length):
            e_out, encoder_hidden = encoder(tensor[input], encoder_hidden)
            encoder_outs[input] = e_out[0, 0]

        decoder_in = torch.tensor([[0]])
        decoder_hidden = encoder_hidden
        translated_words = ""
        for input in tqdm(range(output_length)):
            decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden, encoder_outs)
            value, index = decoder_out.data.topk(1)
            if index is 0:
                break
            translated_words += lang2.index_dict[index.item()]
            translated_words += " "
        translated_words=translated_words[:-1]
        translated_words += ' .'
        translated_words = translated_words.split(' ')
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([sentence], translated_words, smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(BLEUscore)
    print(np.average(bleu_scores))


def preprocess(file1, file2, max_length):
    """
    input text file
    output list[list[words]]
    one sentence -> list[words]

    """
    f = open(file1, 'r')
    lines = f.readlines()
    lang1_cleaned = [normalizeString(unicodeToAscii(line)) for line in lines]
    f = open(file2, 'r')
    lines = f.readlines()
    lang2_cleaned = [normalizeString(unicodeToAscii(line)) for line in lines]
    filtered_lang1 = []
    filtered_lang2 = []
    for i, j in zip(lang1_cleaned,lang2_cleaned):
        if len(i) > max_length or len(j) > max_length:
            pass
        else:
            filtered_lang1.append(i)
            filtered_lang2.append(j)
    return filtered_lang1, filtered_lang2


# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class LanguageModel:
    def __init__(self, lang):
        self.num_words = 2
        self.lang = lang
        self.word_dict = dict()
        self.index_dict = {0: "StartSequence", 1: "EndSequence"}

    def add_line(self, line):
        for word in line.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_dict:
            self.word_dict[word] = self.num_words
            self.index_dict[self.num_words] = word
            self.num_words += 1

    def tokens_from_line(self, line):
        return [self.word_dict[word] for word in line.split(' ')]
    def word_dict_lookup(self, word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return 0

def fetch_random_tensor(lang1_lines, lang2_lines, lang1, lang2):
    size = len(lang1_lines)
    sentence_index = random.randrange(1, size - 1)
    line1 = lang1_lines[sentence_index]
    line1.append(0)
    line2 = lang2_lines[sentence_index]
    line2.append(0)
    tensor1 = torch.tensor(line1).view(-1, 1)
    tensor2 = torch.tensor(line2).view(-1, 1)
    return tensor1, tensor2


def translate(args, sentence, encoder, decoder, lang1, lang2):
    sentence = str(sentence)
    translated_words = []
    sentence = unicodeToAscii(sentence).split(' ')
    tokenized = [lang1.word_dict[i] for i in sentence]
    tensor = torch.tensor(tokenized).view(-1, 1)
    encoder_hidden = encoder.Hidden()
    input_length = tensor.size(0)
    encoder_outs = torch.zeros(args.max_length, encoder.hidden_size)
    for input in range(input_length):
        e_out, e_hidden = encoder(tensor[input], encoder_hidden)
        encoder_outs[input] = e_out[0, 0]

    decoder_in = torch.tensor([[0]])
    decoder_hidden = encoder_hidden
    for input in range(args.max_length):
        decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden, encoder_outs)
        value, index = decoder_out.data.topk(1)
        if index is 0:
            break
        translated_words.append(lang2.index_dict[index.item()])
        print(lang2.index_dict[index.item()])
        print(index.item())
    print(translated_words)