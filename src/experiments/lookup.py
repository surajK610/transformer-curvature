import string
import sys
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
#sys.path.append('../src')
from .utils.curvature_utils import compute_global_curvature, compute_prob_kls, early_decode, layer_wise_consec_kl_divs, layer_wise_curvature, \
                                    logits_to_ave_logit_diff, residual_stack_to_logit_diff, \
                                    layer_wise_norm_running_average
from .utils.plotting_functions import plot_curvature_loss_vs_repetitions, plot_curvature_vs_repetitions, \
                                       plot_rr_vs_repetitions, plot_loss_vs_repetitions, plot_clusters, \
                                       plot_layer_curvature_loss_vs_repetitions, plot_logit_lens
                                       
import argparse
import random
import os
import warnings
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from tqdm import tqdm

def lookup_sequence_norm_analysis(model, dataset_size, 
                                  rand_seq_length, num_repetitions,
                                  key_len, size_dict, token_sep, device = "cpu", 
                                  resid_post=False):
  
  ## must be run from home directory of the project
  print("Generating/Loading Lookup Data...")
  
  path = f"datasets/lookup_task/examples-size={dataset_size}-rand_seq_len={rand_seq_length}-num_reps={num_repetitions}-key_len={key_len}-dic_sz={size_dict}"
  if not os.path.exists(path):
    os.system(f"python src/experiments/utils/lookup_task_gen.py --size={dataset_size} --rand-seq-length={rand_seq_length} --num-repetitions={num_repetitions} --key-len={key_len} --size-dict={size_dict} --token-sep={token_sep}")
  
  examples = np.loadtxt(path, dtype=int)
  if examples.ndim == 1:
    examples = np.expand_dims(examples, axis=0)
  
  sample_vectors = []
  sample_losses = []
  sample_kl_divs = []
  results = []

  for example in tqdm(examples):
    seq_length = int(len(example[1:])/num_repetitions)
    value_idx = seq_length - 2 ## index of the colon 
    
    example = torch.tensor(example).to(device)
    model_logits, model_cache = model.run_with_cache(example, remove_batch_dim=True)
    loss = model.loss_fn(model_logits, example.unsqueeze(0), per_token=True)[0].cpu()
    
    losses = loss[value_idx::seq_length]
    logits = model_logits[0, value_idx::seq_length, :].cpu()
    log_probs = torch.nn.functional.log_softmax(logits.cpu(), dim=-1)
    
    dataset_norms = layer_wise_norm_running_average(model_cache, model.cfg.n_layers, value_idx, seq_length)
    dataset_kl_divergences = layer_wise_consec_kl_divs(model, model_cache, model.cfg.n_layers, value_idx, seq_length, 
                                                       resid_post=resid_post)
      
    index_results = {
          "logits": logits,
          "losses": losses,
          "log_probabilities": log_probs,
          "running_norms": dataset_norms,
          "kl_divergences": dataset_kl_divergences,
    }
    results.append(index_results)
    sample_vectors.append(index_results['running_norms'])
    sample_losses.append(index_results['losses'])
    sample_kl_divs.append(index_results['kl_divergences'])
  return results, sample_vectors, sample_losses, sample_kl_divs

def logit_lens_analysis(model, dataset_size, 
                        rand_seq_length, num_repetitions,
                        key_len, size_dict, token_sep,
                        device):
  
  ## must be run from home directory of the project
  print("Generating/Loading Lookup Data...")
  
  path = f"datasets/lookup_task/examples-size={dataset_size}-rand_seq_len={rand_seq_length}-num_reps={num_repetitions}-key_len={key_len}-dic_sz={size_dict}"
  if not os.path.exists(path):
    os.system(f"python src/experiments/utils/lookup_task_gen.py --size={dataset_size} --rand-seq-length={rand_seq_length} --num-repetitions={num_repetitions} --key-len={key_len} --size-dict={size_dict} --token-sep={token_sep}")
    
  examples = np.loadtxt(path, dtype=int)
  
  tokens, answer_tokens = [], []
  for example in tqdm(examples):    
    tokens.append(example)
    answer_tokens.append([example[-1], example[-2]]) ## example[-2] should always be incorrect (sep token)
    
  answer_tokens = torch.stack(answer_tokens).to(device)
  tokens = torch.stack(tokens).to(device)

  original_logits, cache = model.run_with_cache(tokens)
  return {'original_logits':original_logits, 'cache':cache, 
          'answer_tokens':answer_tokens}


def main(FLAGS):
  print("Running the script", sys.argv[0])
  torch.set_grad_enabled(False)
  torch.manual_seed(0)
  np.random.seed(0)
  
  model = HookedTransformer.from_pretrained("gpt2-small", device=FLAGS.device) ## should add device
  num_layers = model.cfg.n_layers
  
  if FLAGS.experiment == "generate_streamlit_data":  
    results, sample_vectors, sample_losses, sample_kl_divs = lookup_sequence_norm_analysis(model, FLAGS.size, 
                                                                                    FLAGS.rand_seq_length, FLAGS.num_repetitions, 
                                                                                    FLAGS.key_len, FLAGS.size_dict, FLAGS.token_sep, 
                                                                                    device=FLAGS.device, resid_post=True)
    keys_layers = list(range(num_layers))
    ## total is the sum of the layers curvature (equivalent to the global curvature)
    
    layers = {}
    for layer in keys_layers:
      layers[str(layer)] = {}
      layers[str(layer)]['running_norm'] = [[norm[layer] for norm in running_norms] for running_norms in sample_vectors]
      layers[str(layer)]['consec_kls'] = [[kl[layer] for kl in consec_kl] for consec_kl in sample_kl_divs]
    layers['loss'] = [sample_loss.cpu().tolist() for sample_loss in sample_losses]
    os.makedirs(f"outputs/lookup/streamlit/", exist_ok=True)
    json.dump(layers, open(f"outputs/lookup/streamlit/{FLAGS.name}.json", "w"))
    
  elif FLAGS.experiment == "run_logit_lens_analysis":
    os.makedirs(f"outputs/lookup/logit_lens/size={FLAGS.size}" + \
                        f"-rand_seq_len={FLAGS.rand_seq_length}-num_reps={FLAGS.num_repetitions}" + \
                        f"-key_len={FLAGS.key_len}-dic_sz={FLAGS.size_dict}", exist_ok=True)

    for i in range(FLAGS.num_repetitions):
      results = logit_lens_analysis(model, FLAGS.size, 
                                    FLAGS.rand_seq_length, FLAGS.num_repetitions, 
                                    FLAGS.key_len, FLAGS.size_dict, FLAGS.token_sep, 
                                    device=FLAGS.device)
      plot_logit_lens(results['cache'], results['prompts'], 
                      results['answer_tokens'], model, 
                      cumulative=True, show=False, 
                      save_fig=True, save_path=f"outputs/lookup/logit_lens/size={FLAGS.size}" + \
                        f"-rand_seq_len={FLAGS.rand_seq_length}-num_reps={FLAGS.num_repetitions}" + \
                        f"-key_len={FLAGS.key_len}-dic_sz={FLAGS.size_dict}/cum_diff_{i}.png")
      
if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  parser = argparse.ArgumentParser(
              prog='Lookup Sequence with Noise Experiment',
              description='tests norm convergence progression as lookup key length increases')
  parser.add_argument('--size', default=100, type=int, help='size of the dataset')
  parser.add_argument('--num-repetitions', default=100, type=int, help='maximum number of repeated in context examples')
  parser.add_argument('--rand-seq-length', default=3, type=int, help='how many random tokens before the key value pair')
  parser.add_argument('--key-len', default=1, type=int, help='how many tokens in the key')
  parser.add_argument('--size-dict', default=1, type=int, help='how many key value pairs in the dictionary')
  parser.add_argument('--token-sep', default=25, type=int, help='token id of the separator token (default 25 is colon)')
  parser.add_argument('--save-path', default="datasets/lookup_task/", type=str, help='where to save the dataset')
  parser.add_argument("--device", default="cpu", type=str, help="device to run experiment on")
  parser.add_argument("--name", default="layer_wise_analysis", type=str, help="name to call output")
  parser.add_argument("--resid", default="no", type=str, help="residuals or out")
  parser.add_argument('--experiment', default='generate_streamlit_data', type=str, 
                      help='experiment to run (generate_streamlit_data, run_logit_lens_analysis)')
  
  FLAGS = parser.parse_args()
  
  assert FLAGS.experiment in ['generate_streamlit_data', 'run_logit_lens_analysis']
  
  main(FLAGS)