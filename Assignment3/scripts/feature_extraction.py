import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
from collections import defaultdict


def clean_text(text):
    # convert to lower case
    res = text.lower()

    # remove puctuations and numbers
    res = re.sub('['+string.punctuation+']', '', res)
    res = re.sub(r'[~^0-9]', '', res).split()
    return res


def encode_text(text, enc_dict):
    cln_txt = clean_text(text)
    enc = np.zeros(len(enc_dict)).astype(int)

    for word in cln_txt:
        if word in enc_dict:
            enc[enc_dict[word]] += 1

    return enc



class featureExtractor:

    def __init__(self, data_file, stop_word_file=None, extract_from=['spam', 'ham'], top_words=300):
        self.data_file = data_file
        self.stop_word_file = stop_word_file
        
        self.extract_from = extract_from
        self.top_words = top_words
        
        self.stop_words = []

    def load_data(self):
        # load dataframe
        self.df = pd.read_csv(self.data_file, delimiter=',')
        if self.stop_word_file is not None:
            self.stop_words = np.loadtxt(self.stop_word_file,
                                        delimiter=",",
                                        dtype=object)

    def build_dict(self, label=['spam', 'ham'], top_words=200):
        # to create frequency array of the words
        # in all mails with 'label'

        freq = defaultdict(int)
        for i, msg in enumerate(self.df["Message"]):
            if self.df["Spam/Ham"][i] not in label:
                continue

            try:
                cln_txt = clean_text(msg)
                for word in cln_txt:
                    freq[word] += 1
            except AttributeError:
                #print(f"Message {i} is not string, skipping")
                #print(msg)
                pass

        # sort items in dictionary in descending order
        freq_dict = dict(freq)
        self.freq_dict = dict(
            sorted(freq_dict.items(), key=lambda item: -item[1]))

        # choose top k words excluding stop words
        cnt = 0
        self.encoding_dict = {}
        for i, w, in enumerate(self.freq_dict):
            if w not in self.stop_words:
                self.encoding_dict[w] = cnt
                cnt += 1
            if cnt >= top_words:
                break

    def encode_text(self, text):
        cln_txt = clean_text(text)
        enc = np.zeros(len(self.encoding_dict)).astype(int)

        for word in cln_txt:
            if word in self.encoding_dict:
                enc[self.encoding_dict[word]] += 1

        return enc

    def run(self):
        self.load_data()
        self.build_dict(label=self.extract_from,
                        top_words=self.top_words)


if __name__ == "__main__":
    data_file = "../data/enron_spam_data.csv"
    stop_word_file = "../data/stop_words.csv"

    fe = featureExtractor(data_file=data_file,
                          stop_word_file=stop_word_file,
                          extract_from=['spam'],
                          top_words=500)
    fe.run()

    print(fe.encoding_dict)