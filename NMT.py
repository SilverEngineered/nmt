import torch
import argparse
import os
import utils
from models import Encoder, Decoder


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', dest='hidden_size', default=200)
parser.add_argument('--epochs', dest='epochs', type=int, default=100000)
parser.add_argument('--lr', dest='lr', default=.01)
parser.add_argument('--max_length', dest='max_length', default=30)
parser.add_argument('--checkpoint', dest='checkpoint', default=False)
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

args.output_size = Vietnamese.num_words
if 'train' in args.mode:
    if args.checkpoint is True:
        encoder.load_state_dict(torch.load(os.path.join('model', 'encoder.pkl')))
        decoder.load_state_dict(torch.load(os.path.join('model', 'decoder.pkl')))
        encoder.eval()
        decoder.eval()
    utils.train_handler(args, encoder, decoder, English_tokens, Vietnamese_tokens, English, Vietnamese)
    torch.save(encoder.state_dict(), os.path.join('model', 'encoder.pkl'))
    torch.save(decoder.state_dict(), os.path.join('model', 'decoder.pkl'))

elif 'translate' in args.mode:
    encoder.load_state_dict(torch.load(os.path.join('model', 'encoder.pkl')))
    decoder.load_state_dict(torch.load(os.path.join('model', 'decoder.pkl')))
    encoder.eval()
    decoder.eval()
    while True:
        print("Please Enter Input...")
        sentence = input()
        utils.translate(args, sentence, encoder, decoder, English, Vietnamese)
elif 'test' in args.mode:
    encoder.load_state_dict(torch.load(os.path.join('model', 'encoder.pkl')))
    decoder.load_state_dict(torch.load(os.path.join('model', 'decoder.pkl')))
    encoder.eval()
    decoder.eval()
    English_sentences, Vietnamese_sentences = utils.preprocess('data/tst2013.en.txt', 'data/tst2013.vi.txt', args.max_length)
    utils.test(args, encoder, decoder, English, Vietnamese, English_sentences, Vietnamese_sentences)
