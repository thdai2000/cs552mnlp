def search(
        self,
        inputs: dict,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences=1,
        length_penalty: float = 0.0
) -> dict:
    """Generates sequences of token ids with self.model,
    (which has a language modeling head) using beam search. This means that
    given a probability distribution over the possible next tokens and
    a beam width (here num_beams), needs to keep track of the most probable
    num_beams candidates. (Hint: use log probabilities!)

    - This function always does early stopping and does not handle the case
        where we don't do early stopping, or stops at max_new_tokens.
    - It only handles inputs of batch size = 1.
    - It only handles beam size > 1.
    - It includes a length_penalty variable that controls the score assigned
        to a long generation. This is implemented by exponiating the amount
        of newly generated tokens to this value. Then, divide the score which
        can be calculated as the sum of the log probabilities so far.

    Args:
        inputs (dict): the tokenized input dictionary returned by the tokenizer
        max_new_tokens (int): the maximum numbers of new tokens to generate
                              (i.e. not including the initial input tokens)
        num_beams (int): number of beams for beam search
        num_return_sequences (int, optional):
            the amount of best sequences to return. Cannot be more than beam size.
            Defaults to 1.
        length_penalty (float, optional):
            exponential penalty to the length that is used with beam-based generation.
            It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
            Defaults to 0.0.

    Returns:
        dict: dictionary with two key values:
                - "sequences": torch.LongTensor depicting the best generated sequences (token ID tensor)
                    * shape (num_return_sequences, maximum_generated_sequence_length)
                    * ordered from best scoring sequence to worst
                    * if a sequence has reached end of the sentence,
                      you can fill the rest of the tensor row with the pad token ID
                - "sequences_scores": length penalized log probability score list, ordered by best score to worst
    """
    ########################################################################

    self.model.eval()


    return_sequences_candidates = []
    return_sequences_candidates_scores = []
    return_sequences_candidates_len = []

    top_k_inputs = [inputs]
    top_k_logprobs = [0] * num_beams

    for cur_len in range(max_new_tokens):

        # for each of k hypotheses, find top k next words and calculate scores
        next_ids_all = []  # k squared elements
        next_logprobs_all = []
        next_hypotheses_all = []
        for top_i, input in enumerate(top_k_inputs):
            outputs = self.model(**input)
            next_logprobs = torch.nn.functional.log_softmax(outputs['logits'][:, -1, :], dim=-1)  # (N,V)
            next_k_logprobs, next_k_ids = torch.topk(next_logprobs, num_beams, dim=-1)  # (N,k), k=n_beams
            for i, next_id in enumerate(next_k_ids[0]):
                next_ids_all.append(next_id.item())
                next_logprobs_all.append(top_k_logprobs[top_i] + next_k_logprobs[0][i].item())
                next_hypotheses_all.append(torch.cat((input['input_ids'], next_id.view(1, -1)), dim=1))

        # of k^2 hypotheses, keep k with the highest scores, serving as inputs of the next step
        sorted_idx = torch.argsort(torch.tensor(next_logprobs_all), descending=True)
        top_k_inputs = []
        top_k_logprobs = []
        counter = 0
        for i in sorted_idx.tolist():
            # A hypothesis is considered complete and kept as a return sequence candidate only when:
            # 1. it reaches the length of max_new_tokens, or
            # 2. it produces an EOS, in this case, we search the k-highest on the rest hypotheses
            if next_ids_all[i] == self.eos_token_id or cur_len == max_new_tokens - 1:
                return_sequences_candidates.append(next_hypotheses_all[i].squeeze(0))  # the eos token is included
                return_sequences_candidates_scores.append(next_logprobs_all[i])
                return_sequences_candidates_len.append(cur_len + 1)
            else:
                top_k_inputs.append({'input_ids': next_hypotheses_all[i],
                                     'attention_mask': torch.ones((1, next_hypotheses_all[i].shape[1]),
                                                                  dtype=torch.long)})
                top_k_logprobs.append(next_logprobs_all[i])
                counter += 1
                if counter == num_beams:
                    break

    # select and sort candidate return sequences according to penalized scores
    return_sequences_candidates_scores = [score / (return_sequences_candidates_len[i] ** length_penalty) for i, score in
                                          enumerate(return_sequences_candidates_scores)]
    sorted_scores, sorted_idx = torch.sort(torch.tensor(return_sequences_candidates_scores), descending=True)
    print("Hypotheses scores:")
    print(sorted_scores)
    print("Length of hypotheses")
    print([len(return_sequences_candidates[i]) for i in sorted_idx.tolist()])

    # determine the max_seq_len
    final_return_sequences = [return_sequences_candidates[i] for i in sorted_idx.tolist()[:num_return_sequences]]
    max_seq_len = 0
    for seq in final_return_sequences:
        if len(seq) > max_seq_len:
            max_seq_len = len(seq)
    return_sequences_tensor = torch.zeros((num_return_sequences, max_seq_len), dtype=torch.long)

    # save return sequences as tensor
    for i, seq in enumerate(final_return_sequences):
        paddings = torch.tensor([self.pad_token_id] * (max_seq_len - len(seq)), dtype=torch.long)
        return_seq_tensor = torch.cat((seq, paddings))
        return_sequences_tensor[i, :] = return_seq_tensor

    sorted_scores = sorted_scores.tolist()[:num_return_sequences]

    print(return_sequences_tensor)

    return {'sequences': return_sequences_tensor, 'sequences_scores': sorted_scores}