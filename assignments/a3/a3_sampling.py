import torch
from typing import Any, Dict
from a3_utils import *


class TopKSamplerForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)
    
    @torch.no_grad()
    def sample(
        self,
        inputs: dict,
        top_k: int,
        temperature: float,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-k sampling. 
        This means that we sample the next token from the top-k scoring tokens 
        by using their probability values.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - It only handles top_k => 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.
        
        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_k (int): the number of highest probability vocabulary tokens 
                         to keep for top-k filtering/sampling
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-k sampled sequence made of token ids of size (1,generated_seq_len)
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens, top_k=top_k)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################

        # I follow the original paper (Holtzman et al., ICLR2020) to implement top-k sampling
        for cur_len in range(max_new_tokens):

            outputs = self.model(**inputs)

            # softmax the logits with temperature, and select top-k tokens from the resulted distribution
            logits = outputs['logits'][:, -1, :] / temperature  # (1,V)
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (1,V)
            top_k_probs, top_k_ids = torch.topk(probs, k=top_k, dim=-1)  # (1,k)

            # set the prob of other tokens as zero
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, top_k_ids, 1)
            probs = probs * mask

            # normalize the probs of top-k tokens
            sum_top_k_probs = torch.sum(top_k_probs[0][:top_k])
            probs[mask] = probs[mask] / sum_top_k_probs

            # sample a token from the top-k tokens
            next_id = torch.multinomial(probs, num_samples=1)[0].item()

            # prepare the next inputs and handle early stopping
            inputs = self.prepare_next_inputs(inputs, torch.tensor([next_id]))
            if next_id == self.eos_token_id:
                break

        return inputs['input_ids']


class TopPSamplerForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)
    
    @torch.no_grad()
    def sample(
        self,
        inputs: dict,
        top_p: float,
        temperature: float,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-p sampling. 
        This means that we sample the next token from the smallest set of most 
        probable tokens with probabilities that cumulatively add up to top_p *or higher*.
        If there are no tokens falling in the top_p cumulative probability mass 
        (e.g. because the top scoring tokens probability is larger than top_p) 
        then samples the top scoring token.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.

        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_p (float): the cumulative probability mass to select the smallest 
                           set of most probable tokens with probabilities that 
                           cumulatively add up to top_p or higher.
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-p sampled sequence made of token ids of size (1,generated_seq_len)
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################

        # I follow the original paper (Holtzman et al., ICLR2020) to implement top-p sampling
        for cur_len in range(max_new_tokens):

            outputs = self.model(**inputs)

            # softmax the logits with temperature
            logits = outputs['logits'][:, -1, :] / temperature  # (1,V)
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # select top-p tokens from the resulted distribution
            # we first sort tokens by their probs
            # and then select tokens with probs that cumulatively add up right below p, plus their next token
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)  # (1,V)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).squeeze()
            n_to_select = torch.sum(cumulative_probs < top_p, dim=-1).item() + 1
            top_p_probs = sorted_probs[0][:n_to_select]
            top_p_ids = sorted_ids[0][:n_to_select]

            # set the prob of other tokens as zero
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, top_p_ids.unsqueeze(0), 1)
            probs = probs * mask

            # normalize the probs of top-p tokens
            sum_top_p_probs = torch.sum(top_p_probs)
            probs[mask] = probs[mask] / sum_top_p_probs

            # sample a token from the top-p tokens
            next_id = torch.multinomial(probs, num_samples=1)[0].item()

            # prepare the next inputs and handle early stopping
            inputs = self.prepare_next_inputs(inputs, torch.tensor([next_id]))
            if next_id == self.eos_token_id:
                break

        return inputs['input_ids']


def main():
    ############################################################################
    # NOTE: You can use this space for testing but you are not required to do so!
    ############################################################################
    seed = 421
    torch.manual_seed(seed)
    torch.set_printoptions(precision=16)
    model_name = "vicgalle/gpt2-alpaca-gpt4"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


if __name__ == '__main__':
    main()
