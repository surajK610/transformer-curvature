import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

import torch
import matplotlib.pyplot as plt
from curvature_utils import compute_global_curvature, compute_local_curvature, early_decode
import argparse
import random

def generate_random_token_sequence(seq_length, tokenizer):
  '''Generates a random sequence of tokens of length seq_length, 
  sampling from the valid tokens in the tokenizer vocabulary uniformly 
  at random.
  Params:
      seq_length: int, length of sequence to generate
      tokenizer: transformers.PreTrainedTokenizer, tokenizer to use
  Returns:
      random_tokens: list of str, list of tokens sampled uniformly at random
      random_sequence: str, string of tokens sampled uniformly at random'''
  valid_token_ids = [i for i in range(tokenizer.vocab_size) if i != tokenizer.pad_token_id]
  random_token_ids = random.choices(valid_token_ids, k=seq_length)
  random_tokens = tokenizer.convert_ids_to_tokens(random_token_ids)
  random_sequence = tokenizer.convert_tokens_to_string(random_tokens)
  return random_tokens, random_sequence

def repeat_sequence(num_examples, string_sequence):
  '''Repeats a token sequence num_examples times.
  Params:
      num_examples: int, number of times to repeat token sequence
      string_sequence: str, the string sequence to repeat
  Returns:
      repeated_sequence: repeated string sequence
  '''
  return string_sequence * num_examples
  
def repeated_sequence_analysis(string_sequence, num_repeats, model):
  '''Analyzes the curvature of a repeated sequence.
  Params:
      string_sequence: str, the string sequence to repeat
      num_repeats: int, number of times to repeat token sequence  
      model: HookedTransformer, the model to analyze
  Returns:
      results: dict, dictionary of results
  '''
  repeated_string_sequence = repeat_sequence(num_repeats, string_sequence)
  repeated_tokens = model.to_tokens(repeated_string_sequence)[0]
  seq_length = int(len(repeated_tokens[1:])/num_repeats) # ignores start token 
  
  dataset_early_decoding_probs = {}
  dataset_early_decoding_rrs = {}
  
  model_logits, model_cache = model.run_with_cache(repeated_tokens, remove_batch_dim=True)
  loss = model.loss_fn(model_logits, repeated_tokens.unsqueeze(0), per_token=True)[0].cpu()

  results = {}

  for stream_index in range(seq_length):
    print((stream_index + 1)/seq_length)
    losses = loss[stream_index::seq_length]
    logits = model_logits[0, stream_index::seq_length, :].cpu()
    dataset_curvatures = compute_global_curvature(model_cache, model.cfg.n_layers, stream_index, seq_length)
    dataset_log_probs = torch.nn.functional.log_softmax(logits.cpu(), dim=-1)

    for layer in range(model.cfg.n_layers):
      decodes = early_decode(model, model_cache, layer)
      softmax_logits = torch.nn.functional.log_softmax(decodes[0, stream_index::seq_length, :].cpu(), dim=-1)
      ed_correct_probs = softmax_logits[:, repeated_tokens[stream_index + 1]]

      ## TODO: Convert to compute logit attribution instead of reciprocal rank
      ed_reciprocal_ranks = []
      for i in range(len(ed_correct_probs)):
        vocab_probs, _ = torch.sort(softmax_logits[i], descending=True)
        rank = (vocab_probs == ed_correct_probs[i]).nonzero(as_tuple=True)[0][0]
        reciprocal_rank = 1/(rank + 1)
        ed_reciprocal_ranks.append(reciprocal_rank)

      dataset_early_decoding_probs[layer] = ed_correct_probs
      dataset_early_decoding_rrs[layer] = ed_reciprocal_ranks

    index_results = {
        "logits": logits,
        "losses": losses,
        "curvatures": dataset_curvatures,
        "log_probabilities": dataset_log_probs,
        "ed_log_probabilities": dataset_early_decoding_probs,
        "ed_reciprocal_ranks": dataset_early_decoding_rrs
    }
    results[stream_index] = index_results

  return results

def plot_curvature_vs_repetitions(results, sequence_position, offset=0):
  x = list(range(len(results[sequence_position]["curvatures"])))
  y = results[sequence_position]["curvatures"]
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title("Global Curvature of the Residual Stream")
  plt.ylabel("Global Curvature")
  plt.xlabel("Repetitions")
  plt.show()

def plot_earliest_1_rr_vs_repetitions(results, sequence_position, offset=0):
  rr_dict = results[sequence_position]["ed_reciprocal_ranks"]
  x = list(range(len(rr_dict[0])))
  earliest_rr_of_1 = []
  for i in range(len(x)):
    early_rr = 12 # default value if no rr of 1
    for layer in range(len(rr_dict)):
      if rr_dict[layer][i] == 1:
        early_rr = layer
        break
    earliest_rr_of_1.append(early_rr)
  y = earliest_rr_of_1
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title("Earliest RR of 1 Layer vs Repetitions")
  plt.ylabel("Layer")
  plt.xlabel("Repetitions")
  plt.show()

def plot_rr_vs_repetitions(results, sequence_position, offset=0, layer=11):
  rr_dict = results[sequence_position]["ed_reciprocal_ranks"]
  x = list(range(len(rr_dict[layer])))
  y = rr_dict[layer]
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title(f"RR of Layer {layer} vs Repetitions")
  plt.ylabel("RR")
  plt.xlabel("Repetitions")
  plt.show()

def plot_loss_vs_repetitions(results, sequence_position, offset=0):
  x = list(range(len(results[sequence_position]["losses"])))
  y = results[sequence_position]["losses"]
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title("Loss vs Repetitions")
  plt.ylabel("Loss")
  plt.xlabel("Repetitions")
  plt.show()

def main(FLAGS):
  torch.set_grad_enabled(False)
  torch.manual_seed(0)
  
  model = HookedTransformer.from_pretrained("gpt2-small", device=FLAGS.device)
  
  
  
  
if __name__ == 'main':
  parser = argparse.ArgumentParser(
              prog='Repeat Sequence Experiment',
              description='tests curvature as repeat sequence length increases')
  parser.add_argument('--seq-length', default=40, type=int, help='number of tokens per in-context example')
  parser.add_argument('--min-examples', default=1, type=int, help='minimum number of in context examples')
  parser.add_argument('--max-examples', default=100, type=int, help='maximum number of in context examples')
  ## TODO: ensure that max-examples is within context window length
  parser.add_argument("--use-data-source", default=False, type=bool, 
        help="whether to use a data source or generate random sequence")
  parser.add_argument("--data-path", default="datasets/repeat_sequence.txt", type=str, 
        help="data path where sequence is located")
  
  parser.add_argument("--device", default="cuda:0", type=str, help="device to run experiment on")
  
  FLAGS = parser.parse_args()
  main(FLAGS)
  