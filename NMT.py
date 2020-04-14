import torch
import torch.nn as nn
import argparse
import os
import utils
from models import Encoder, Decoder


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', dest='hidden_size', default=200)
parser.add_argument('--output_size', dest='output_size', default=100)
parser.add_argument('--num_classes', dest='num_classes', default=10)
parser.add_argument('--epochs', dest='epochs', default=25000)
parser.add_argument('--lr', dest='lr', default=10e-3)
parser.add_argument('--max_length', dest='max_length', default=60)
parser.add_argument('--checkpoint', dest='checkpoint', default=True)
parser.add_argument(nargs=argparse.REMAINDER, dest='mode')
args = parser.parse_args()

English_sentences, Vietnamese_sentences = utils.preprocess('data/train.en.txt', 'data/train.vi.txt', args.max_length)
English = utils.LanguageModel("English")
[English.add_line(i) for i in English_sentences]
Vietnamese = utils.LanguageModel("Vietnamese")
[Vietnamese.add_line(i) for i in Vietnamese_sentences]
English_tokens = [English.tokens_from_line(i) for i in English_sentences]
Vietnamese_tokens = [Vietnamese.tokens_from_line(i) for i in Vietnamese_sentences]
encoder = Encoder(args, input_size=English.num_words)
decoder = Decoder(args, output_size=Vietnamese.num_words)
if 'train' in args.mode:
    if args.checkpoint is True:
        encoder.load_state_dict(torch.load(os.path.join('model', 'encoder.pkl')))
        decoder.load_state_dict(torch.load(os.path.join('model', 'decoder.pkl')))
    loss_function = nn.NLLLoss()
    e_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    utils.train(args, English, Vietnamese, e_optimizer, d_optimizer, encoder, decoder, loss_function, English_tokens, Vietnamese_tokens)
    torch.save(encoder.state_dict(), os.path.join('model', 'encoder.pkl'))
    torch.save(decoder.state_dict(), os.path.join('model', 'decoder.pkl'))

elif 'translate' in args.mode:
    encoder.load_state_dict(torch.load(os.path.join('model', 'encoder.pkl')))
    decoder.load_state_dict(torch.load(os.path.join('model', 'decoder.pkl')))
    encoder.eval()
    decoder.eval()
    while True:
        sentence = input()
        utils.translate(args, sentence, encoder, decoder, English, Vietnamese)
elif 'test' in args.mode:
    encoder.load_state_dict(torch.load(os.path.join('model', 'encoder.pkl')))
    decoder.load_state_dict(torch.load(os.path.join('model', 'decoder.pkl')))
    encoder.eval()
    decoder.eval()
    English_sentences, Vietnamese_sentences = utils.preprocess('data/tst2013.en.txt', 'data/tst2013.vi.txt', args.max_length)
    utils.test(args, encoder, decoder, English, Vietnamese, English_sentences, Vietnamese_sentences)
