from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm
import jsonlines

import torch
import torch.utils.data
from torch import nn, optim
import numpy as np 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

# 1.1
class SADataset(Dataset):
    """
    Implement SADataset in Pytorch
    """
    def __init__(self, data_repo, tokenizer, sent_max_length=128):
        
        self.label_to_id = {"positive": 0, "negative": 1}
        self.id_to_label = {0: "positive", 1: "negative"}
        
        self.tokenizer = tokenizer

        ##############################################################################################
        # TODO: Get the special token and token id for PAD from defined tokenizer (self.tokenizer).  #
        ##############################################################################################
        
        self.pad_token = self.tokenizer.pad_token
        self.pad_id = self.tokenizer.pad_token_id
        
        ############################################################################
        #                               END OF YOUR CODE                           #
        ############################################################################

        self.text_samples = []
        self.samples = []
        
        print("Building SA Dataset...")
        
        with jsonlines.open(data_repo, "r") as reader:
            for sample in tqdm(reader.iter()):
                self.text_samples.append(sample)

                ###############################################################################
                # TODO: build input token indices (input_ids) and label indices (label_id):   #
                #     - remember to truncate tokens if it exceeds the sent_max_length;        #
                #     - remember to add special tokens for RoBERTa model input format;        #
                #     - set label id to None if no label is available.
                #     Ed note: the sent_max_length for truncating tokens does not include
                #     the special tokens later added to the RoBERTa model input format,
                #     i.e., the final input length can exceed sent_max_length
                #     after adding special tokens #
                ###############################################################################

                input_ids = self.tokenizer.encode(sample["review"], add_special_tokens=False)[:sent_max_length]
                input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
                label_id = self.label_to_id[sample["label"]]
                
                ############################################################################
                #                               END OF YOUR CODE                           #
                ############################################################################
                
                self.samples.append({"ids": input_ids, "label": label_id})

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return deepcopy(self.samples[index])

    def padding(self, inputs, max_length=-1):
        """
        Pad inputs to the max_length.
        
        INPUT: 
          - inputs: input token ids
          - max_length: the maximum length you should add padding to.
          
        OUTPUT: 
          - pad_inputs: token ids padded to `max_length` """

        ##############################################################################################
        # TODO: if max_length < 0, pad to the length of the longest input sample in the batch        #
        ##############################################################################################

        if max_length == -1:
            max_length = max(map(len, inputs))
        pad_inputs = []
        for input in inputs:
            pad_inputs.append(input + [self.tokenizer.pad_token_id] * (max_length - len(input)))
        pad_inputs = torch.tensor(pad_inputs, dtype=torch.long)

        ############################################################################
        #                               END OF YOUR CODE                           #
        ############################################################################

        return pad_inputs
        
    def collate_fn(self, batch):
        """
        Convert batch inputs to tensor of batch_ids and labels.
        
        INPUT: 
          - batch: batch input, with format List[Dict1{"ids":..., "label":...}, Dict2{...}, ..., DictN{...}]
          
        OUTPUT: 
          - tensor_batch_ids: torch tensor of token ids of a batch, with format Tensor(List[ids1, ids2, ..., idsN])
          - tensor_labels: torch tensor for corresponding labels, with format Tensor(List[label1, label2, ..., labelN])
        """
        ##############################################################################################
        # TODO: implement collate_fn for batchify input into preferable format.                     #
        ##############################################################################################

        tensor_batch_ids = self.padding([sample['ids'] for sample in batch], max_length=-1)
        tensor_labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long)
        
        ############################################################################
        #                               END OF YOUR CODE                           #
        ############################################################################

        return tensor_batch_ids, tensor_labels
    
    def get_text_sample(self, index):
        return deepcopy(self.text_samples[index])
    
    def decode_class(self, class_ids):
        """
        Decode to output the predicted class name.
        
        INPUT: 
          - class_ids: index of each class.
          
        OUTPUT: 
          - labels_from_ids: a list of label names. """
        ##############################################################################################
        # TODO: implement class decoding function, return "unknown" for unknown class id prediction. #
        ##############################################################################################
        
        label_name_list = [self.id_to_label[cid] for cid in class_ids]

        ############################################################################
        #                               END OF YOUR CODE                           #
        ############################################################################
        
        return label_name_list

# 1.2
def compute_metrics(predictions, gold_labels):
    """
    Compute evaluation metrics (confusion matrix and F1 scores) for SA task. only use numpy
    
    INPUT: 
      - gold_labels: real labels;
      - predictions: model predictions.
    OUTPUT:
      - confusion matrix; [[TP, FN], [FP, TN]], where positive is 0 and negative is 1
      - f1 scores for positive and negative classes.
    """
    ##############################################################################
    # TODO: Implement metrics computation.                                       #
    ##############################################################################

    predictions = np.array(predictions)
    gold_labels = np.array(gold_labels)
    TP = ((predictions == gold_labels) & (predictions == 0)).sum()
    FP = ((predictions != gold_labels) & (predictions == 0)).sum()
    FN = ((predictions != gold_labels) & (predictions == 1)).sum()
    TN = ((predictions == gold_labels) & (predictions == 1)).sum()
    confusion_matrix = [[TP, FN], [FP, TN]]
    recall_p = TP / (TP + FN)
    precision_p = TP / (TP + FP)
    recall_n = TN / (TN + FP)
    precision_n = TN / (TN + FN)
    f1_positive = 2 * precision_p * recall_p / (precision_p + recall_p)
    f1_negative = 2 * precision_n * recall_n / (precision_n + recall_n)

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    
    return confusion_matrix, f1_positive, f1_negative


def train(train_dataset, dev_dataset, model, device, batch_size, epochs,
          learning_rate, warmup_percent, max_grad_norm, model_save_root,
          tensorboard_path="./tensorboard"):
    '''
    Train models with predefined datasets.

    INPUT:
      - train_dataset: dataset for training
      - test_dataset: dataset for evlauation
      - model: model to train
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
      - epochs: total epochs to train the model
      - learning_rate: learning rate of optimizer
      - warmup_percent: percentage of warmup steps
      - max_grad_norm: maximum gradient for clipping
      - model_save_root: path to save model checkpoints
    '''
    
    tb_writer = SummaryWriter(tensorboard_path)
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn
    )

    ###########################################################################################
    # TODO: Define optimizer and learning rate scheduler with learning rate and warmup steps. # 
    ###########################################################################################
    # Replace "..." statement with your code
    
    # calculate total training steps
    total_steps = len(train_dataloader) * epochs
    warmup_steps = total_steps * warmup_percent
    
    # set up AdamW optimizer and constant learning rate scheduler with warmup
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    
    model.zero_grad()
    model.train()
    best_dev_macro_f1 = 0
    total_train_step = 0
    log_freq = 20
    save_repo = model_save_root + 'lr{}-warmup{}'.format(learning_rate, warmup_percent)
    
    for epoch in range(epochs):
        
        train_loss_accum = 0.0
        epoch_train_step = 0
        running_loss = 0.0
        ##############################################################################################################
        # TODO: Implement the training process. You should calculate the loss then update the model with optimizer.  # 
        #       You should also keep track on the training step and update the learning rate scheduler.              # 
        ##############################################################################################################
        # Replace "..." with your code
        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            
            # clear the gradients of all optimized parameters
            optimizer.zero_grad()

            epoch_train_step += 1
            total_train_step += 1

            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            # get model's single-batch outputs and loss
            outputs = model(input_ids, labels=labels)  # loss (1,)  logits (N,2)
            loss = outputs.loss
            
            # conduct back-proporgation
            loss.backward()

            # truncate gradient to max_grad_norm
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            train_loss_accum += loss.mean().item()
            
            running_loss += loss.mean().item()
            if i > 0 and i % log_freq == 0:
                # print(f'[Step {total_train_step:5d}] loss: {running_loss / log_freq:.3f}')
                tb_writer.add_scalar("task1_roberta/loss/train", running_loss / log_freq, total_train_step)
                running_loss = 0.0

            # step forward optimizer and scheduler
            optimizer.step()
            scheduler.step()

        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        
        epoch_train_loss = train_loss_accum / epoch_train_step

        # epoch evaluation
        dev_loss, confusion, f1_pos, f1_neg = evaluate(dev_dataset, model, device, batch_size)
        macro_f1 = (f1_pos + f1_neg) / 2
        tb_writer.add_scalar("task1_roberta/loss/eval", dev_loss, total_train_step)
        
        print(f'Epoch: {epoch} | Training Loss: {epoch_train_loss:.3f} | Validation Loss: {dev_loss:.3f}')
        print(f'Epoch {epoch} SA Validation:')
        print(f'Confusion Matrix:')
        print(confusion)
        print(f'F1: ({f1_pos*100:.2f}%, {f1_neg*100:.2f}%) | Macro-F1: {macro_f1*100:.2f}%')
        
        ##############################################################################################################
        # TODO: Update the highest macro_f1. Save best model and tokenizer to <save_repo>.                           # 
        ##############################################################################################################
        # Replace "..." statement with your code
        if macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = macro_f1
            model.save_pretrained(save_repo)
            train_dataset.tokenizer.save_pretrained(save_repo)
            print("Model Saved!")
        
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
    
    tb_writer.flush()
    tb_writer.close()


def evaluate(eval_dataset, model, device, batch_size, use_labels=True, result_save_file=None):
    '''
    Evaluate the trained model.

    INPUT: 
      - eval_dataset: dataset for evaluation
      - model: trained model
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
      - use_labels: whether the gold labels should be used as one input to the model
      - result_save_file: path to save the prediction results
    '''
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=batch_size,
        collate_fn=eval_dataset.collate_fn
    )
    
    eval_loss_accum = 0
    eval_step = 0
    batch_preds = []
    batch_labels = []
    
    model.eval()
    
    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        
        eval_step += 1
        
        with torch.no_grad():
            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            #####################################################
            #      TODO: get model outputs, loss and logits.    # 
            #####################################################
            # Replace "..." statement with your code
            if use_labels:
                outputs = model(input_ids, labels=labels)  # loss (1,)  logits (N,2)
                loss = outputs.loss
                logits = outputs.logits
            else:
                outputs = model(input_ids)
                logits = outputs.logits

            #######################################################
            #                    END OF YOUR CODE                 #
            #######################################################

            batch_preds.append(logits.detach().cpu().numpy())
            if use_labels:
                batch_labels.append(labels.detach().cpu().numpy())
                eval_loss_accum += loss.mean().item()

    #####################################################
    #          TODO: get model predicted labels.        # 
    #####################################################
    # Replace "..." statement with your code
    predicted_labels = [np.argmax(batch, axis=1) for batch in batch_preds]
    pred_labels = list(np.concatenate(predicted_labels))

    #####################################################
    #                   END OF YOUR CODE                #
    #####################################################
    
    if result_save_file:
        pred_results = eval_dataset.decode_class(pred_labels)
        with jsonlines.open(result_save_file, mode="w") as writer:
            for sid, pred in enumerate(pred_results):
                sample = eval_dataset.get_text_sample(sid)
                sample["prediction"] = pred
                writer.write(sample)
    
    if use_labels:
        eval_loss = eval_loss_accum / eval_step
        gold_labels = list(np.concatenate(batch_labels))
        confusion, f1_pos, f1_neg = compute_metrics(pred_labels, gold_labels)
        return eval_loss, confusion, f1_pos, f1_neg
    else:
        return None
