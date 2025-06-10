import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from tqdm import tqdm
from tqdm.notebook import tqdm  # changed: for better output in jupyter notebook (otherwise it will create new lines every second)
import evaluate
from torch.utils.tensorboard import SummaryWriter
################################################
##       Part2 --- Language Modeling          ##
################################################   
class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 dropout_rate,
                 embedding_weights=None,
                 freeze_embeddings=False):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # pass embeeding weights if exist
        if embedding_weights is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights).float())
            self.embedding.weight.requires_grad = not freeze_embeddings
        else:  # train from scratch embeddings
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # TODO: Define **bi-directional** LSTM layer with `num_layers` and `dropout_rate`.
        ## Hint: what is the input/output dimensions? how to set the bi-directional model?
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        # TODO: Define the feedforward layer with `num_layers` and `dropout_rate`.
        ## Hint: what is the input/output dimensions for a bi-directional LSTM model?
        self.fc = nn.Linear(hidden_dim*2, vocab_size)

    def forward(self, input_id):
        embedding = self.dropout(self.embedding(input_id)) # embedding: (N,L,d_e)
        
        # TODO: Get output from (LSTM layer-->dropout layer-->feedforward layer)
        ## You can add several lines of code here.
        hidden_states, _ = self.lstm(embedding) # hidden_states: (N,L,d_h*2)
        dropout = self.dropout(hidden_states) # dropout: (N,L,d_h*2)
        output = self.fc(dropout) # output: (N,L,V)
        return output

def train_lstm(model, train_loader, optimizer, criterion, device="cuda:0", tensorboard_path="./tensorboard"):
    """
    Main training pipeline. Implement the following:
    - pass inputs to the model
    - compute loss
    - perform forward/backward pass.
    """
    tb_writer = SummaryWriter(tensorboard_path)
    # Training loop
    model.train()
    running_loss = 0
    epoch_loss = 0
    print("Start training on device {}".format(device))
    for i, data in enumerate(tqdm(train_loader)):
        # get the inputs
        inputs = data.to(device) # (N,L)

        # TODO: get the language modelling labels form inputs
        labels = inputs[:, 1:] # (N,L-1)
    
        # TODO: Implement forward pass. Compute predicted y by passing x to the model
        y_pred = model(inputs) # (N,L,V)
        y_pred = y_pred[:, :-1, :].permute(0, 2, 1) # (N,V,L-1)

        # TODO: Compute loss
        loss = criterion(y_pred, labels)
        
        # TODO: Implement Backward pass. 
        # Hint: remember to zero gradients after each update. 
        # You can add several lines of code here.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i>0 and i % 500 == 0. :
            print(f'[Step {i + 1:5d}] loss: {running_loss / 500:.3f}')
            tb_writer.add_scalar("lstm/train/loss", running_loss / 500, i)
            running_loss = 0.0

    tb_writer.flush()
    tb_writer.close()
    print(f'Epoch Loss: {(epoch_loss / len(train_loader)):.4f}')
    return epoch_loss / len(train_loader)

def test_lstm(model, test_loader, criterion, device="cuda:0"):
    """
    Main testing pipeline. Implement the following:
    - get model predictions
    - compute loss
    - compute perplexity.
    """

    # Testing loop
    batch_loss = 0

    model.eval()
    for data in tqdm(test_loader):
        # get the inputs
        inputs = data.to(device)
        labels = data[:, 1:].to(device)

        # TODO: Run forward pass to get model prediction.
        y_pred = model(inputs)
        y_pred = y_pred[:, :-1, :].permute(0, 2, 1)
        
        # TODO: Compute loss
        loss = criterion(y_pred, labels)
        batch_loss += loss.item()

    test_loss = batch_loss / len(test_loader)
    
    # TODO: Get test perplexity using `test_loss``
    perplexity = np.power(np.e, test_loss)  # Since PyTorch use e as log base in cross-entropy by default, I also use e here
    print(f'Test loss: {test_loss:.3f}')
    print(f'Test Perplexity: {perplexity:.3f}')
    return test_loss, perplexity

################################################
##       Part3 --- Finetuning          ##
################################################ 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, pretrained_encoder, hidden_size):
        super(Encoder, self).__init__()
        self.pretrained_encoder = pretrained_encoder
        self.hidden_size = hidden_size

    def forward(self, input_ids, input_mask):
        # TODO: Implement forward pass.
        # Hint 1: You should take into account the fact that pretrained encoder is bidirectional.
        # Hint 2: Check out the LSTM docs (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
        # Hint 3: Do we need all the components of the pretrained encoder?
        # no, the weights that project to vocab is not needed

        embedding = self.pretrained_encoder.dropout(self.pretrained_encoder.embedding(input_ids))  # embedding: (N,L,h_e)

        # pack the variant-length sequences to accelerate calculation
        # it will also skip the <pad> tokens in the backward pass (directly starts from <eos>)
        lengths, indices = torch.sort(input_mask.sum(dim=1), descending=True) # input_mask: (N,L)
        embedding_sorted = embedding[indices] # embedding_sorted: (N,L,h_e)
        embedding_packed = pack_padded_sequence(embedding_sorted, lengths.cpu(), batch_first=True)
        hidden_states_packed, (h_n, _) = self.pretrained_encoder.lstm(embedding_packed) # h_n: (4,N,d_h)
        max_seq_len = embedding.size(1)
        hidden_states, _ = pad_packed_sequence(hidden_states_packed, total_length=max_seq_len, batch_first=True) # hidden_states: (N,L,d_h*2)

        # recover the original order of samples
        _, original_indices = torch.sort(indices, descending=False)
        encoder_outputs = hidden_states[original_indices] # encoder_outputs: (N,L,d_h*2)
        h_n = h_n[:, original_indices, :]

        # concatenate the final forward hidden state of the last layer and the final reverse hidden state of the last layer
        encoder_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) # encoder_hidden: (N,d_h*2)

        return encoder_outputs, encoder_hidden

'''
for the concatenation of the h_n

【AI tool】: GPT4

【Prompt】: what does each dimension of output and h_n of a two-layer bi-directional LSTM in pytorch contain?
How do I concatenate the h_n of the last layer, and keep the order of forward and backward hidden state the same as output?

【Output】: 
In PyTorch, when using a two-layer bidirectional LSTM, the outputs and the hidden states (h_n) have specific structures that reflect the architecture of the network.

Output Structure (output)
The output tensor from an LSTM gives you the output features (h_t) from the last layer of the LSTM for each time step. For a two-layer bidirectional LSTM, the dimensions of output are as follows:

output: [seq_len, batch, num_directions * hidden_size]
Where:

seq_len is the length of the input sequence.
batch is the batch size.
num_directions is 2 for a bidirectional LSTM.
hidden_size is the number of features in the hidden state.
This tensor provides the LSTM's output for each time step, including both the forward and backward directions in the last layer.

Hidden State Structure (h_n)
The h_n tensor contains the hidden states for the last element of the sequence. For a two-layer bidirectional LSTM, its dimensions are as follows:

h_n: [num_layers * num_directions, batch, hidden_size]
Where:

num_layers is the number of layers in the LSTM.
num_directions is 2 for a bidirectional LSTM.
batch is the batch size.
hidden_size is the number of features in the hidden state.
h_n essentially captures the final hidden state for each layer and each direction.

Concatenating the Last Layer's Forward and Backward h_n
To concatenate the forward and backward hidden states of the last layer and maintain the order as in the output, you should extract the relevant slices from h_n and use torch.cat to concatenate them. Here’s how you do it:

python
Copy code
import torch

# Assuming h_n is your hidden state tensor from the LSTM
# Shape of h_n: [num_layers * num_directions, batch, hidden_size]

# Extract the forward and backward hidden states of the last layer
h_n_forward = h_n[-2, :, :]  # The second last is the forward hidden state of the last layer
h_n_backward = h_n[-1, :, :]  # The last is the backward hidden state of the last layer

# Concatenate the hidden states along the hidden_size dimension
h_n_concat = torch.cat((h_n_forward, h_n_backward), dim=-1)

# Now, h_n_concat contains the concatenated forward and backward hidden states of the last layer
# with shape [batch, 2 * hidden_size]
This approach ensures that the concatenated hidden states align with the order and structure of the output tensor, effectively capturing the comprehensive context from the final layer of your bidirectional LSTM, which is particularly useful when passing this as the initial hidden state to a decoder in a seq2seq model.

【How do I verified the correctness of the output?】
I printed the sencond last and the last hidden state in h_n, and the first part of the output at the last time step, as well as 
the second part of the output at the first time step, and found they were the same

【Why the AI tool suggests doing so?】
As the note of in the pytorch document (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) says, 
"For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.
Example of splitting the output layers when batch_first=False: output.view(seq_len, batch, num_directions, hidden_size)."
"For bidirectional LSTMs, h_n is not equivalent to the last element of output;
the former contains the final forward and reverse hidden states,
while the latter contains the final forward hidden state and the initial reverse hidden state.",
in order to maintain the order of forward and reverse hidden states, one should keep the final forward hidden state in the 
0 dimention, and the final reverse hidden state in the 1 dimention, and then concatenate the two.

As the blog (https://discuss.pytorch.org/t/get-the-last-hidden-state-in-stacked-lstm/131040/2) suggests, the last layer's
hidden states are stacked in a "reverse" order, where the final layer is at the bottom.

Therefore, the second last hidden state should be the final forward hidden state of the last layer,
and the last hidden state should be the final reverse hidden state of the last layer.
'''


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.query_weights = nn.Linear(hidden_size, hidden_size)
        self.value_weights = nn.Linear(hidden_size, hidden_size)
        self.combined_weights = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        # TODO: Implement forward pass.
        # Note: this part requires several lines of code

        # Attention weights
        # assume query has the shape (N,L_de,d_h), values the shape (N,L_en,d_h)
        # the aim is to use matrix multiplication to calculate the attentions for all queries in the decoder together
        # thus avoiding the for loop when doing teacher forcing training
        # the special case is when doing inference, the number of query is one
        query_expanded = query.unsqueeze(2) # (N,L_de,1,d_h)
        values_expanded = values.unsqueeze(1) # (N,1,L_en,d_h)
        added = torch.tanh(self.query_weights(query_expanded) + self.value_weights(values_expanded)) # (N,L_de,L_en,d_h)
        combined_weights = self.combined_weights(added).squeeze(-1) # (N,L_de,L_en)

        # The context vector is the weighted sum of the values.
        if mask is not None:
            mask = mask.unsqueeze(1)
            combined_weights = combined_weights.masked_fill(mask == 0, -float('inf')) # mask: (N,1,L_en)
        weights = F.softmax(combined_weights, dim=-1) # weights: (N,L_de,L_en)
        context = torch.matmul(weights, values) # weights (N,L_de,L_en) * values (N,L_en,d_h) = context (N,L_de,d_h)

        return context, weights

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, bos_token_id, dropout_rate=0.15, encoder_embedding=None):
        super(Decoder, self).__init__()
        # Note: feel free to change the architecture of the decoder if you like
        if encoder_embedding is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = encoder_embedding
        self.attention = AdditiveAttention(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # changed: I use hidden states to calculate the attention, in line with the slide
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_size * 2, output_size)  # changed: the output then is the concat of hidden state and context vector, thus doubling the dimension
        self.bos_token_id = bos_token_id
        # added
        self.projection = nn.Linear(hidden_size * 2, hidden_size) # to project the encoder outputs and encoder hidden to hidden_size

    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensors=None, device=0):
        # TODO: Implement forward pass.
        # Note: this part requires several lines of code
        # Hint: Use target_tensors to handle training and inference appropriately

        encoder_outputs = self.projection(encoder_outputs) # (N,L,d_h)
        encoder_hidden = self.projection(encoder_hidden) # (N,d_h)

        # training by teacher forcing
        # assume target_tensors has the shape (N,L)
        if target_tensors is not None:
            gt_embeddings = self.dropout(self.embedding(target_tensors))  # (N,L,d_e)
            output, decoder_hidden = self.gru(gt_embeddings, encoder_hidden.unsqueeze(0))  # (N,L,d_h), (1,N,d_h)
            context, weights = self.attention(output, encoder_outputs, input_mask)  # (N,L,d_h), (N,L,L)
            decoder_outputs = self.out(torch.cat((output, context), dim=-1))  # (N,L,d_h*2)->(N,L,V)
        # inference or training without teacher forcing
        else:
            batch_size = encoder_outputs.size(0)  # N=1 when generating
            sequence_length = encoder_outputs.size(1)
            decoder_outputs = torch.zeros(batch_size, sequence_length, self.out.out_features, device=device)  # (N,L,V)
            input_id = torch.tensor(self.bos_token_id, device=device).expand(batch_size)  # (N)
            decoder_hidden = encoder_hidden.unsqueeze(0)  # (1,N,d_h)
            for t in range(sequence_length):
                input_embedding = self.dropout(self.embedding(input_id)).unsqueeze(1)  # (N,1,d_e)
                output, decoder_hidden = self.gru(input_embedding, decoder_hidden)  # (N,1,d_h), (1,N,d_h)
                context, weights = self.attention(output, encoder_outputs, input_mask)  # (N,1,d_h), (N,1,L)
                logits = self.out(torch.cat((output, context), dim=-1)).squeeze(1)  # (N,1,d_h*2) -> (N,1,V) -> (N,V)
                decoder_outputs[:, t, :] = logits
                # Get the next input by selecting the word with the highest probability
                topv, topi = logits.topk(1)
                input_id = topi.squeeze().detach()  # Prepare for the next iteration

        return decoder_outputs, decoder_hidden

    
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, bos_token_id, dropout_rate=0.15, pretrained_encoder=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(pretrained_encoder, hidden_size)
        # the embeddings in the encoder and decoder are tied as they're both from the same language
        self.decoder = Decoder(hidden_size, output_vocab_size, bos_token_id, dropout_rate, pretrained_encoder.embedding)

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs, input_mask)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden


def seq2seq_eval(model, eval_loader, criterion, device=0):
    model.eval()
    epoch_loss = 0
    
    for i, data in tqdm(enumerate(eval_loader), total=len(eval_loader)):
        # TODO: Get the inputs
        input_ids, target_ids, input_mask = data['input_ids'], data['output_ids'], data['input_mask']
        input_ids, target_ids, input_mask = input_ids.to(device), target_ids.to(device), input_mask.to(device)

        # TODO: Forward pass
        decoder_outputs, decoder_hidden = model(input_ids, input_mask, targets=target_ids)

        # in my implementation, the decoder outputs have the same sequence length (max_seq_length) as the target
        # the padding token will not contribute to the loss since pad_tokeb_id is ignored in the NLLloss
        batch_max_seq_length = target_ids.size(1)
        assert decoder_outputs.size(1) == batch_max_seq_length
        labels = target_ids  # (N,L)

        # TODO: Compute loss
        decoder_outputs = decoder_outputs.permute(0, 2, 1)  # (N,V,L)
        decoder_outputs = nn.LogSoftmax(dim=1)(decoder_outputs)  # need to calculate the logsofmax before NLLloss
        loss = criterion(decoder_outputs[:, :, :-1], labels[:, 1:])  # discard the <bos> label, and the hidden states of <eos> for the output
        epoch_loss += loss.item()

    model.train()

    return epoch_loss / len(eval_loader)

def seq2seq_train(model, train_loader, eval_loader, optimizer, criterion, num_epochs=20, device=0, tensorboard_path="./tensorboard"):
    tb_writer = SummaryWriter(tensorboard_path)
    # Training loop
    model.train()
    best_eval_loss = 1e3 # used to do early stopping
    waiting = 0

    for epoch in tqdm(range(num_epochs), leave=False, position=0):
        running_loss = 0
        epoch_loss = 0
        
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, position=1):
            # TODO: Get the inputs
            input_ids, target_ids, input_mask = data['input_ids'], data['output_ids'], data['input_mask']
            input_ids, target_ids, input_mask = input_ids.to(device), target_ids.to(device), input_mask.to(device)
            
            # Forward pass
            decoder_outputs, decoder_hidden = model(input_ids, input_mask, targets=target_ids)

            # in my implementation, the decoder outputs have the same sequence length (max_seq_length) as the target
            # the padding token will not contribute to the loss since pad_tokeb_id is ignored in the NLLloss
            batch_max_seq_length = target_ids.size(1)
            assert decoder_outputs.size(1) == batch_max_seq_length
            labels = target_ids  # (N,L)

            # TODO: Compute loss
            decoder_outputs = decoder_outputs.permute(0, 2, 1)  # (N,V,L)
            decoder_outputs = nn.LogSoftmax(dim=1)(decoder_outputs)
            loss = criterion(decoder_outputs[:, :, :-1], labels[:, 1:])  # discard the <bos> label, and the hidden states of <eos> for the output
            epoch_loss += loss.item()
            
            # TODO: Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99. :    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} | Train Loss: {(epoch_loss / len(train_loader)):.4f}')
        eval_loss = seq2seq_eval(model, eval_loader, criterion, device=device)
        print(f'Epoch {epoch + 1} | Eval Loss: {(eval_loss):.4f}')
        tb_writer.add_scalar("ec-finetune/loss/train", epoch_loss / len(train_loader), epoch)
        tb_writer.add_scalar("ec-finetune/loss/eval", eval_loss, epoch)
        
        # TODO: Perform early stopping based on eval loss
        # Make sure to flush the tensorboard writer and close it before returning
        if eval_loss <= best_eval_loss:
            best_eval_loss = eval_loss
            waiting = 0
        else:
            waiting += 1
            if waiting > 5:  # for 5 epoch, if there is no decrease in eval_loss, then early stop
                tb_writer.flush()
                tb_writer.close()
                return epoch_loss / len(train_loader)

    tb_writer.flush()
    tb_writer.close()
    return epoch_loss / len(train_loader)

def seq2seq_generate(model, test_loader, tokenizer, device=0):
    generations = []

    model.eval()

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        # TODO: get the inputs
        input_ids, target_ids, input_mask = data['input_ids'], data['output_ids'], data['input_mask']
        input_ids, target_ids, input_mask = input_ids.to(device), target_ids.to(device), input_mask.to(device)

        # TODO: Forward pass
        decoder_outputs, decoder_hidden = model(input_ids, input_mask, targets=None)
        # decoder_outputs (N,L,V)
        _, topi = decoder_outputs.topk(1)
        outputs = topi.squeeze() # (N,L)

        # TODO: Decode outputs to natural language text
        # Note we expect each output to be a string, not list of tokens here
        for o_id, output in enumerate(outputs):
            generations.append({"input": " ".join(tokenizer.decode(input_ids[o_id].tolist(), skip_special_tokens=True)),
                                "reference": " ".join(tokenizer.decode(target_ids[o_id].tolist(), skip_special_tokens=True)),
                                "prediction": " ".join(tokenizer.decode(output.tolist(), skip_special_tokens=True))})
    
    return generations

def evaluate_rouge(generations):
    # TODO: Implement ROUGE evaluation
    references = [sample['reference'] for sample in generations]
    predictions = [sample['prediction'] for sample in generations]

    rouge = evaluate.load('rouge')

    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return rouge_scores

def t5_generate(dataset, model, tokenizer, device=0):
    # TODO: Implement T5 generation
    generations = []

    for sample in tqdm(dataset, total=len(dataset)):
        reference = tokenizer.decode(sample['label_ids'].to(device), skip_special_tokens=True)

        # Hint: use huggingface text generation
        outputs = model.generate(sample['input_ids'].to(device).unsqueeze(0)) # the pretrained t5 expects batched data, thus adding dimension
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generations.append({
            "input": tokenizer.decode(sample['input_ids'].to(device), skip_special_tokens=True),
            "reference": reference, 
            "prediction": prediction})
    
    return generations