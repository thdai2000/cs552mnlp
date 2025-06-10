from typing import Optional

import torch 
import numpy as np

from transformers.utils import logging

logger = logging.get_logger(__name__)

def filter_using_adaptive_plausibility_constraint(
    scores: torch.FloatTensor,
    alpha: float, 
    filter_value: float = -float("Inf"), 
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Filters tokens in the vocabulary using the adaptive plausibility constraint as described in the Contrastive Decoding paper.

    Args:
        scores (torch.FloatTensor): The input scores of shape (batch_size, vocab_size) representing logits for each token in the vocabulary.
        alpha (float): A hyperparameter controlling the strength of the adaptive plausibility constraint.
        filter_value (float, optional): The value to replace scores below the adaptive plausibility threshold. Defaults to -float("Inf").
        min_tokens_to_keep (int, optional): Minimum number of tokens to retain based on their ranking by score. Defaults to 1.

    Returns:
        torch.Tensor: The filtered scores after applying the adaptive plausibility constraint, of the same shape as the input scores.

    Note:
        This function assumes that `scores` are the logits of tokens in the vocabulary.
        It calculates an adaptive plausibility threshold based on the maximum probability of tokens in each batch and the specified alpha value.
        The tokens with scores below this threshold are replaced with `filter_value`.
    """
    # TODO: implement this function

    # normalize logits to probs
    scores_normalized = torch.nn.functional.softmax(scores, dim=-1)  # (N,V)

    # replace scores below the threshold
    max_scores = torch.max(scores_normalized, dim=-1, keepdim=True)[0]
    scores_normalized.masked_fill_(scores_normalized < alpha * max_scores, filter_value)

    # calculate the logprobs for the scores >= threshold
    scores_normalized[scores_normalized >= alpha * max_scores] \
        = torch.log(scores_normalized[scores_normalized >= alpha * max_scores])

    return scores_normalized 

@torch.no_grad()
def greedy_search_with_contrastive_decoding(
    model,
    amateur_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    alpha: float,
    amateur_temperature: float,
    **model_kwargs,
) -> torch.Tensor:
    """
    Performs greedy search using contrastive decoding subject to the adaptive plausibility constraint.

    Args:
        model (torch.nn.Module): The expert model used for generating text.
        amateur_model (torch.nn.Module): The amateur model (e.g., `GPT2-Small`) for contrastive decoding.
        input_ids (torch.Tensor): Tensor containing the input token IDs.
        max_new_tokens (int): Maximum number of tokens to generate.
        alpha (float): A hyperparameter controlling the strength of the adaptive plausibility constraint.
        amateur_temperature (float): Temperature parameter for generating tokens using the amateur model.
        **model_kwargs: Additional keyword arguments to be passed to the model during generation.

    Returns:
        torch.Tensor: Tensor containing the generated token IDs.

    Note:
        This function performs greedy search, iteratively generating tokens until `max_new_tokens` are reached.
        During token generation, it employs contrastive decoding by utilizing both the expert and amateur models.
        The generated tokens are subject to an adaptive plausibility constraint, enhancing the diversity and coherence of the generated text.
    """
    
    bos_token_id = model.config.bos_token_id
    eos_token_id = model.config.eos_token_id
    pad_token_id = model.config.eos_token_id

    input_ids, _, model_kwargs = model._prepare_model_inputs(input_ids, bos_token_id, model_kwargs)

    model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
        input_ids, pad_token_id, eos_token_id
    )

    num_tokens = 0

    while True:

        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)  # input_ids (N,L)

        # TODO: forward pass to get the next token distribution using the expert model
        outputs = model(**model_inputs)
        next_token_logits = outputs['logits'][:, -1, :]  # (N,V)

        model_inputs_amateur =  amateur_model.prepare_inputs_for_generation(input_ids)
        
        # TODO: forward pass to get the next token distribution using the amateur model            
        outputs_amateur = amateur_model(**model_inputs_amateur)
        next_token_logits_amateur = outputs_amateur['logits'][:, -1, :]  # (N,V)

        # TODO: normalize the token distribution by taking the log softmax using the amateur_temperature 
        next_token_logits_amateur = next_token_logits_amateur / amateur_temperature
        next_tokens_scores_amateur = torch.nn.functional.log_softmax(next_token_logits_amateur, dim=-1)

        next_tokens_scores = filter_using_adaptive_plausibility_constraint(
            next_token_logits,
            alpha=alpha,
        )  # (N,V)

        # TODO: contrast the next_tokens_scores (that you get after filtering)
        #       with the next token scores of the amateur model.
        #       Hint: reread Section 3.3 of the paper if it's not clear how!
        mask = next_tokens_scores != -float("Inf")
        next_tokens_scores[mask] = next_tokens_scores[mask] - next_tokens_scores_amateur[mask]

        # TODO: take the argmax to predict the next token then add it to the input_ids 
        next_tokens = torch.argmax(next_tokens_scores, dim=-1, keepdim=True)  # (N,1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )

        num_tokens += 1

        # stop when sentence exceed the maximum length
        if num_tokens > max_new_tokens:
            break
      
    return input_ids
    

@torch.no_grad()
def greedy_search(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    max_new_tokens: Optional[int] = None,
) -> torch.LongTensor:
    """
    Implements vanilla greedy search using the HuggingFace library generate function.

    Args:
        model (torch.nn.Module): The model used for text generation.
        input_ids (torch.LongTensor): Tensor containing the input token IDs.
        attention_mask (Optional[torch.LongTensor], optional): Tensor indicating which tokens should be attended to. Defaults to None.
        max_new_tokens (Optional[int], optional): Maximum number of tokens to generate. If None, it uses the model's maximum length configuration. Defaults to None.

    Returns:
        torch.LongTensor: Tensor containing the generated token IDs.

    Note:
        This function utilizes the HuggingFace library's generate function to perform greedy search.
        It generates tokens iteratively until `max_new_tokens` are reached or until the model's maximum length if `max_new_tokens` is None.
    """
    
    max_new_tokens = max_new_tokens if max_new_tokens is not None else model.config.max_length
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
    )

