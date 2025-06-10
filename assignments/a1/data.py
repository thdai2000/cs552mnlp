from torch.utils.data import Dataset, DataLoader
import datasets
import math
import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm
import nltk
from utils import isEnglish, lowerCase, replaceRare, isUnkSeq
import re

################################################
##       Part1 --- Data Preprocessing         ##
################################################

def filter_by_length(dataset, min_len=100, max_len=128):
    """Filter out sequences with len(s)<min_len and len(s)>max_len.
    Hint: You can use the processing functions provided by Huggingface. 
    (https://huggingface.co/docs/datasets/en/process) """
    dataset = dataset.filter(lambda x: len(x['text'].split()) >= min_len and len(x['text'].split()) <= max_len)
    return dataset

def data_clean(dataset, min_len, max_len):
    """We perform three steps to clean the dataset."""
    # 1- Filter out sequences with len(s)<min_len and len(s)>max_len.
    ## Hint: implement and use the `filter_by_length` function.
    dataset = filter_by_length(dataset, min_len, max_len)
   
    # 2- Remove the samples with = * = \n patterns. (* denotes any possible sequences, e.g. `= = <section> = = \n `)
    dataset = dataset.filter(lambda x: not re.search(r'=\s.*=\s\n', x['text']))

    # 3- Remove Non-English sequences.
    ## Hint: You can use isEnglish(sample) to find non-English sequences.
    dataset = dataset.filter(lambda x: isEnglish(x['text']))

    # 4- Lowercase all sequences.
    ## Hint: You can use lowerCase(sample) to lowercase the given sequence.
    dataset = dataset.map(lambda x: lowerCase(x))

    return dataset

def count_tokens(dataset):
    """Counts the frequency of each token in the dataset.
    You should return a dict with token as keys, frequency as values.
    Hint: you can use Counter() class to help."""

    token_counter = Counter()
    for sample in dataset:
        for token in sample['text'].split():
            token_counter.update([token])

    token_freq_dict = dict(token_counter)

    return token_freq_dict

def build_vocabulary(dataset, min_freq=5, unk_token='<unk>'):
    """Builds a vocabulary dict for the given dataset."""
    # 1- Get unique tokens and their frequencies.
    ## Hint: Use `count_tokens()`.
    token_freq_dict = count_tokens(dataset)

    # 2- Find a set of rare tokens with frequency lower than `min_freq`.
    #    Replace them with `unk_token`.
    rare_tokens_set = set()
    for token, freq in token_freq_dict.items():
        if freq <= min_freq:
            rare_tokens_set.add(token)
    dataset = dataset.map(replaceRare, fn_kwargs={"rare_tokens": rare_tokens_set,
                                                    "unk_token": unk_token})

    # 3- Filter out sequences with more than 15% rare tokens.
    ## Hint: Use `isUnkSeq()` function.
    dataset = dataset.filter(lambda x: not isUnkSeq(x, unk_token, unk_thred=0.15))

    # 4- Recompute the token frequency to get final vocabulary dict.
    token_freq_dict = count_tokens(dataset)

    return dataset, token_freq_dict

class RNNDataset(Dataset):
    def __init__(self,
                dataset: datasets.arrow_dataset.Dataset,
                max_seq_length: int,):
        self.train_data = self.prepare_rnn_lm_dataset(dataset)
        self.max_seq_length = max_seq_length + 2 # as <start> and <stop> will be added
        self.dataset_vocab = self.get_dataset_vocabulary(dataset)
        # TODO: defining a dictionary maps tokens to a unique index in dataset_vocab.
        self.token2idx = {}
        self.idx2token = {}
        for i, token in enumerate(self.dataset_vocab):
            self.token2idx[token] = i
            self.idx2token[i] = token
        
        self.pad_idx = self.token2idx["<pad>"]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # TODO: Get a list of tokens of the given sequence. Represent each token with its index in `self.token2idx`.
        ## Hint: what is the index of `<unk>`?
        token_list = self.train_data[idx].split()
        # having a fallback to <unk> token if an unseen word is encoded.
        token_ids = []

        for token in token_list:
            if token in self.token2idx.keys():
                token_ids.append(self.token2idx[token])
            else:
                token_ids.append(self.token2idx['<unk>'])

        # TODO: Add padding token to the sequence to reach the max_seq_length.
        assert len(token_ids) <= self.max_seq_length
        if len(token_ids) < self.max_seq_length:
            token_ids = token_ids + [self.token2idx['<pad>']] * (self.max_seq_length - len(token_ids))

        return torch.tensor(token_ids)

    def get_dataset_vocabulary(self, dataset: datasets.arrow_dataset.Dataset):
        vocab = set()
        print("Getting the vocabulary for the train dataset")
        for sample in tqdm(dataset):
            vocab.update(set(sample["text"].split()))
        vocab.update(set(["<start>", "<stop>", "<pad>" ]))
        vocab = sorted(vocab)
        return vocab

    @staticmethod
    def prepare_rnn_lm_dataset(target_dataset: datasets.arrow_dataset.Dataset):
        """
        A "<start>" token has to be added before every sentence and a <stop> afterwards.
        
        :param args: target_dataset: the target dataset to extract samples
        return: a list of strings each containing 'window_size' tokens.
        """
        prepared_dataset = []
        for sample in target_dataset:
            prepared_dataset.append(f"<start> {sample['text']} <stop>")
        return prepared_dataset


def get_dataloader(rnn_dataset, test_ratio=0.1):
    # TODO: split train/test dataset.
    # you can add several lines of codes here
    # train_size = len(rnn_dataset) * (1 - test_ratio)
    # test_size = len(rnn_dataset) * test_ratio
    rnn_train_dataset, rnn_test_dataset = torch.utils.data.random_split(rnn_dataset, lengths=[1-test_ratio, test_ratio])

    # TODO: get pytorch DataLoader
    ## Hint: training dataset need to be shuffled, but test dataset does not.
    train_dataloader = DataLoader(rnn_train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(rnn_test_dataset, batch_size=8, shuffle=False)

    return train_dataloader, test_dataloader


################################################
##       Part3 --- Data Preparation         ##
################################################
class CustomTokenizer:
    def __init__(self, vocab, pad_token="<pad>", unk_token="<unk>", bos_token="<start>", eos_token="<stop>", split=None):
        self.vocab = vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        # TODO: define the following attributes
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.pad_token_id = self.word_to_index[self.pad_token]
        self.unk_token_id = self.word_to_index[self.unk_token]
        self.bos_token_id = self.word_to_index[self.bos_token]
        self.eos_token_id = self.word_to_index[self.eos_token]
        self.tokenizer = split

    def encode(self, text, max_length=None):
        """This method takes a natural text and encodes it into a sequence of token ids using the vocabulary.
        Consider adding the BOS and EOS tokens to the sequence at appropriate positions.
        If max_length is provided, it should pad or truncate the sequence to the given length. Note that you
        should also take into account the BOS and EOS tokens when calculating the max length if you use them.
        
        Args:
            text (str): Text to encode.
            max_length (int, optional): Maximum encoding length. Defaults to None.

        Returns:
            List[int]: List of token ids.
        """
        # TODO: encode the given text into a sequence of token ids using the vocabulary.
        # use nltk to tokenize sentence (used in this assignment)
        # the result of using nltk is much better than using space
        if self.tokenizer is not None:
            token_ids = ([self.bos_token_id] +
                         [self.word_to_index[token.lower()] if token.lower() in self.word_to_index.keys() else self.unk_token_id
                          for token in self.tokenizer(text)] +
                         [self.eos_token_id]) # lowering the case to decrease # of <unk>; having a fallback to <unk> token if an unseen word is encoded.
        # use space to tokenize sentence
        else:
            token_ids = ([self.bos_token_id] +
                         [self.word_to_index[token.lower()] if token.lower() in self.word_to_index.keys() else self.unk_token_id
                          for token in text.split()] +
                         [self.eos_token_id])
        if max_length is not None:
            # only truncate the non-special-token part
            if len(token_ids) > max_length:
                token_ids = token_ids[0] + token_ids[1:max_length-1] + token_ids[-1]
            # pad to max_length
            elif len(token_ids) < max_length:
                token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))

        return token_ids

    def decode(self, sequence, skip_special_tokens=True):
        """This method takes a sequence of token ids and decodes it into a language tokens.
        If skip_special_tokens is True, it should skip decoding special tokens such as <pad>, <start>, <stop>, <unk> etc.

        Args:
            sequence (List[int]): Sequence to be decoded.
            skip_special_tokens (bool, optional): Whether to skip special tokens when decoding. Defaults to True.

        Returns:
            List[str]: List of decoded tokens.
        """
        # TODO: decode the given sequence into a list of tokens using the vocabulary.
        special_token_ids = [self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id]
        if skip_special_tokens:
            tokens = [self.index_to_word[token_id] for token_id in sequence if token_id not in special_token_ids]
        else:
            tokens = [self.index_to_word[token_id] for token_id in sequence]
        return tokens

class SCompDataset(Dataset):
   def __init__(self,
               dataset: datasets.arrow_dataset.Dataset,
               tokenizer: object,
               max_seq_length: int
               ):
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

   def __len__(self):
        return len(self.dataset)

   def __getitem__(self, idx):
        # TODO: tokenize the input and output sequences and create the input mask
        # make sure all ids are padded to the max_seq_length

        # pad for both input and output, in order to do batch processing.
        # Padding tokens will be ignored when computing attention and loss
        input_ids = self.tokenizer.encode(self.dataset[idx][0], max_length=self.max_seq_length)
        output_ids = self.tokenizer.encode(self.dataset[idx][1], max_length=self.max_seq_length)
        input_mask = [(idx != self.tokenizer.pad_token_id) for idx in input_ids]  # zero for <pad>, one for the rest

        return {"input_ids": torch.tensor(input_ids), 
                "output_ids": torch.tensor(output_ids),
                "input_mask": torch.tensor(input_mask, dtype=float)}

class ScompT5Dataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw_dataset = raw_dataset

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        # TODO: Tokenize the input and output sequences and create the input mask.
        tokenized_input = self.tokenizer(self.raw_dataset[index][0],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_length)
        input_ids = torch.tensor(tokenized_input['input_ids'])
        input_mask = torch.tensor(tokenized_input['attention_mask'])
        tokenized_label = self.tokenizer(self.raw_dataset[index][1],
                                         padding='max_length',
                                         truncation=True,
                                         max_length=self.max_length)
        label_ids = torch.tensor(tokenized_label['input_ids'])
        label_mask = torch.tensor(tokenized_label['attention_mask'])

        return {
            'input_ids': input_ids.to(dtype=torch.long), 
            'input_mask': input_mask.to(dtype=torch.long), 
            'label_ids': label_ids.to(dtype=torch.long),
            'label_mask': label_mask.to(dtype=torch.long)
        }