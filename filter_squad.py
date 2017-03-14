import argparse
import json
import os

import nltk
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("--context_size_th", type=int, default=150)
    parser.add_argument("--ques_size_th", type=int, default=15)
    return parser.parse_args()


def filter_squad(in_, context_size_th, ques_size_th):
    num_q_0, num_q_1, num_q_2 = 0, 0, 0
    for article in tqdm(in_['data']):
        num_q_0 += sum(len(para['qas']) for para in article['paragraphs'])
        article['paragraphs'] = [paragraph for paragraph in article['paragraphs']
                                 if len(nltk.word_tokenize(paragraph['context'])) <= context_size_th]
        num_q_1 += sum(len(para['qas']) for para in article['paragraphs'])
        for paragraph in article['paragraphs']:
            paragraph['qas'] = [qa for qa in paragraph['qas']
                                if len(nltk.word_tokenize(qa['question'])) <= ques_size_th]
        num_q_2 += sum(len(para['qas']) for para in article['paragraphs'])

    print("Initial number of questions: {}".format(num_q_0))
    print("Removed {} questions during filtering based on context size.".format(num_q_0 - num_q_1))
    print("Removed {} questions during filtering based on question size.".format(num_q_1 - num_q_2))
    print("Final number of questions: {}".format(num_q_2))


def cut_squad(in_, context_size_th, ques_size_th):
    pass


def get_sent_ans_pair(context, answer_start, answer_text):
    pass


def main():
    args = get_args()
    with open(args.in_path, 'r') as fp:
        in_ = json.load(fp)

    filter_squad(in_, args.context_size_th, args.ques_size_th)
    with open(args.out_path, 'w') as fp:
        json.dump(in_, fp)


if __name__ == "__main__":
    main()
