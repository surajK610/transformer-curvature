import json
import warnings
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformers import AutoTokenizer

import torch
import matplotlib.pyplot as plt
from .utils.curvature_utils import compute_global_curvature, early_decode
from .utils.plotting_functions import plot_curvature_vs_repetitions, plot_rr_vs_repetitions, plot_loss_vs_repetitions
import argparse
import random
import os

def word_transformation_analysis(data, model):

  gpt2_tokens = model.to_tokens(data)[0]
  dataset_early_decoding_probs = {}
  dataset_early_decoding_rrs = {}
  gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)
  loss = model.loss_fn(gpt2_logits, gpt2_tokens.unsqueeze(0), per_token=True)[0].cpu()

  results = {}

  losses = loss[2::4]
  logits = gpt2_logits[0, 2::4, :].cpu()
  dataset_curvatures = compute_global_curvature(gpt2_cache, model.cfg.n_layers, 2, 4)
  dataset_log_probs = torch.nn.functional.log_softmax(logits.cpu(), dim=-1)

  for layer in range(model.cfg.n_layers):
    decodes = early_decode(model, gpt2_cache, layer)
    softmax_logits = torch.nn.functional.log_softmax(decodes[0, 2::4, :].cpu(), dim=-1)
    ed_correct_probs = softmax_logits[:, gpt2_tokens[3::4].to("cpu")]
    ed_correct_probs = ed_correct_probs[torch.eye(ed_correct_probs.shape[0]).bool()]

    # Compute reciprocal ranks
    ed_reciprocal_ranks = []
    for i in range(len(ed_correct_probs)):
      vocab_probs, _ = torch.sort(softmax_logits[i], descending=True)
      rank = (vocab_probs == ed_correct_probs[i]).nonzero(as_tuple=True)[0][0]
      reciprocal_rank = 1/(rank + 1)
      #if reciprocal_rank != 1:
        #print(model.to_string(gpt2_tokens[3::3]))
      ed_reciprocal_ranks.append(reciprocal_rank)

    dataset_early_decoding_probs[layer] = ed_correct_probs
    dataset_early_decoding_rrs[layer] = ed_reciprocal_ranks

  results = {
      "logits": logits,
      "losses": losses,
      "curvatures": dataset_curvatures,
      "log_probabilities": dataset_log_probs,
      "ed_log_probabilities": dataset_early_decoding_probs,
      "ed_reciprocal_ranks": dataset_early_decoding_rrs
  }

  return results

def copy_task():
  pass

def past_tense_regular_task():
  pass

def plural_regular_task():
  pass

def main(FLAGS):
  torch.set_grad_enabled(False)
  torch.manual_seed(0)
  
  model = HookedTransformer.from_pretrained("gpt2-small", device=FLAGS.device)
  
  if FLAGS.copy_task:
    copy_task()
  if FLAGS.past_tense_regular_task:
    past_tense_regular_task()
  if FLAGS.plural_regular_task:
    plural_regular_task()    
  
if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  parser = argparse.ArgumentParser(
              prog='Repeat Sequence Experiment',
              description='tests curvature as repeat sequence length increases')
  parser.add_argument('--seq-length', default=5, type=int, help='number of tokens per in-context example')
  parser.add_argument('--num-trials', default=10, type=int, help='number of trials to run with this seq length')
  parser.add_argument('--num-repetitions', default=100, type=int, help='maximum number of repeated in context examples')
  # parser.add_argument("--use-data-source", default=False, type=bool, 
  #       help="whether to use a data source or generate random sequence")
  # parser.add_argument("--data-path", default="datasets/repeat_sequence.txt", type=str, 
  #       help="data path where sequence is located")
  parser.add_argument("--device", default="cuda:0", type=str, help="device to run experiment on")
  
  FLAGS = parser.parse_args()
  main(FLAGS)
  