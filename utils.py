import torch
import numpy as np
import torchvision.transforms as transforms
import re
import unicodedata
import random
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
normalization = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def train(args, lang1, lang2, e_optimizer, d_optimizer, encoder, decoder, loss_function, lang1_tokens, lang2_tokens):
    total_loss = 0
    for i in tqdm(range(args.epochs)):
        loss = 0
        x_tensor, y_tensor = fetch_random_tensor(lang1_tokens, lang2_tokens, lang1, lang2)
        encoder_hidden = encoder.Hidden()
        # initialize optimizer to zero gradients
        e_optimizer.zero_grad()
        d_optimizer.zero_grad()
        # pass data through model

        input_length = x_tensor.size(0)
        target_length = y_tensor.size(0)
        encoder_outs = torch.zeros(args.max_length, encoder.hidden_size)
        for input in range(input_length):
            e_out, e_hidden = encoder(x_tensor[input], encoder_hidden)
            encoder_outs[input] = e_out[0, 0]

        decoder_in = torch.tensor([[0]])
        decoder_hidden = encoder_hidden
        for input in range(target_length):
            decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden, encoder_outs)
            loss += loss_function(decoder_out, y_tensor[input])
            decoder_in = y_tensor[input]
        # Backpropogation
        loss.backward()

        # Optimize
        e_optimizer.step()
        d_optimizer.step()

        # Add loss for this set of inputs
        total_loss += loss.item()
        if i%1000==0:
            print(total_loss/(i+1))
        #print(sentence_bleu(references=1, hypothesis=0, smoothing_function=1))


def test(args, encoder, decoder, lang1, lang2, lang1_sentences, lang2_sentences):
    translated_words = []
    sentences = [unicodeToAscii(i).split(' ') for i in lang1_sentences]
    tokenized = [[lang1.word_dict_lookup(i) for i in j] for j in sentences]
    tensors = [torch.tensor(i).view(-1, 1) for i in tokenized]
    bleu_scores = []
    for tensor, sentence in zip(tensors, sentences):
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
        bleu_scores.append(sentence_bleu(references=sentence, hypothesis=translated_words, smoothing_function=1))
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
    #try:
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