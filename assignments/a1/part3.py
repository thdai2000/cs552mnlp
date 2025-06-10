import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # limiting to one GPU

from datasets import load_dataset
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import evaluate
import gensim
import transformers
import nltk

import re
import random
import numpy as np
import torch

## Enter your SCIPER here: ##
SCIPER = "369945"

try:
    assert re.match("\d{6}", SCIPER)[0] == SCIPER, "Invalid SCIPER given. please enter your correct 6-digit SCIPER number above!"
except:
    print("Invalid SCIPER given. please enter your correct 6-digit SCIPER number above!")

student_seed = int(SCIPER)


"""Set seed for reproducibility."""
random.seed(student_seed)
np.random.seed(student_seed)
torch.manual_seed(student_seed)
torch.cuda.manual_seed_all(student_seed)

############################################################################
# We will use NLTK to tokenize the text
from datasets import load_dataset
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

scomp = load_dataset("embedding-data/sentence-compression")

scomp_small = scomp.filter(lambda x: len(x['set'][0].split()) > 10 and len(x['set'][0].split()) < 100)

scomp_train, scomp_val, scomp_test = scomp_small["train"][:10000]['set'], scomp_small["train"][10000:11000]['set'], scomp_small["train"][11000:12000]['set']

from utils import load_binary

wikitext_vocab = load_binary("wikitext_vocab.pkl")

from data import CustomTokenizer

custom_tokenizer = CustomTokenizer(vocab=wikitext_vocab, split=word_tokenize)

from data import SCompDataset

MAX_SEQ_LENGTH = 128
scomp_train_ds = SCompDataset(scomp_train, custom_tokenizer, MAX_SEQ_LENGTH)
scomp_val_ds = SCompDataset(scomp_val, custom_tokenizer, MAX_SEQ_LENGTH)
scomp_test_ds = SCompDataset(scomp_test, custom_tokenizer, MAX_SEQ_LENGTH)

from torch.utils.data import DataLoader

# feel free to change batch size according to your GPU memory
scomp_train_dataloader = DataLoader(scomp_train_ds, batch_size=32, shuffle=True)
scomp_val_dataloader = DataLoader(scomp_val_ds, batch_size=32, shuffle=True)
scomp_test_dataloader = DataLoader(scomp_test_ds, batch_size=32, shuffle=True)

from modeling import VanillaLSTM

vocab_size = len(wikitext_vocab)
embedding_dim = 100
hidden_dim = 100
num_layers = 2
dropout_rate = 0.15

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

pretrained_encoder = VanillaLSTM(vocab_size, embedding_dim, hidden_dim,
                                  num_layers, dropout_rate=dropout_rate).to(device)

# TODO: Load the pretrained model from the file
pretrained_encoder.load_state_dict(torch.load('./models/lstm_scratch.pt'))

from modeling import EncoderDecoder

lr = 1e-3
dropout_rate = 0.15
bos_token_id = custom_tokenizer.bos_token_id
encoder_decoder = EncoderDecoder(hidden_dim, vocab_size, vocab_size, bos_token_id=bos_token_id, dropout_rate=dropout_rate, pretrained_encoder=pretrained_encoder).to(device)
optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)
criterion = nn.NLLLoss(ignore_index=custom_tokenizer.pad_token_id)
num_params = sum(p.numel() for p in encoder_decoder.parameters() if p.requires_grad)
print(f'The model has {num_params:,} trainable parameters')

from modeling import seq2seq_train

# ETS: ~30 mins to run with a batch size of 32 for 20 epochs
seq2seq_train(model=encoder_decoder,
              train_loader=scomp_train_dataloader,
              eval_loader=scomp_val_dataloader,
              optimizer=optimizer,
              criterion=criterion,
              device=device,
              tensorboard_path="./tensorboard/encoder_decoder")
# save the model
torch.save(encoder_decoder.state_dict(), "models/encoder_decoder.pt")

from modeling import seq2seq_generate
test_generations = seq2seq_generate(encoder_decoder, scomp_test_dataloader, custom_tokenizer, device=device)

