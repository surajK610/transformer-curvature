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
from .utils.curvature_utils import compute_global_curvature, early_decode, layer_wise_curvature, \
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

# -----------------------------------------------------------DATA-----------------------------------------------------------#

## got from smallest logits after a BOS token
LEAST_COMMON_TOKENS = [154, 14695, 15272, 24973, 27013, 13150,   174, 40219, 30905,   215,
                      30208,   124,   204, 39820, 36173,   209,   178,   181,   177,   202,
                        183,   186, 30897,   216,   213,   211,   207, 37574,   205,   210,
                      40240,   197, 40241,   212,   193,   194, 42089,   180,   187, 13198,
                        200,   192, 30898,   179,   206,   173,   201,   184,   182,   214,
                        188, 45544, 39752,   185,   199,   217,   191,   208,   221,   190,
                        219,   196,   189,  4690,   218,   203,   125, 30212,   195, 13945,
                      18945, 23090, 31573, 14827,  9286, 30762, 17629,  5815,   153, 39714,
                      30899, 44392, 39142, 39749,  7782,   155, 24307, 16103, 25618,  3523,
                      20554, 19965, 19476,  7601, 22315, 15755, 27924,  8815,  9364,  4183]

## got from https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt
MOST_COMMON_TOKENS = [262, 286, 290, 284, 257, 287, 329, 318, 319, 326, 416, 428, 351, 1312, 
                      345, 340, 407, 393, 307, 389, 422, 379, 355, 534, 477, 423, 649, 517, 
                      281, 373, 356, 481, 1363, 460, 514, 546, 611, 2443, 616, 468, 2989, 
                      1479, 475, 674, 530, 584, 466, 645, 1321, 640, 484, 2524, 339, 510, 
                      743, 644, 543, 511, 1705, 503, 779, 597, 612, 766, 691, 523, 465, 618, 
                      2800, 994, 1597, 508, 3992, 635, 783, 1037, 651, 9114, 1570, 2691, 269, 
                      304, 717, 716, 587, 561, 703, 547, 502, 264, 2594, 617, 777, 3904, 663, 
                      588, 2139, 2124, 621, 1064]

# -----------------------------------------------------GENERATION FUNCTIONS-----------------------------------------------------#

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
  return random_token_ids, random_sequence

def generate_sequence_from_sample(sample_tokens, seq_length, tokenizer):
  '''Generates a sequence of tokens from a sample.
  Params:
      sample: dict, sample from a dataset
      tokenizer: transformers.PreTrainedTokenizer, tokenizer to use
  Returns:
      tokens: list of str, list of tokens from sample
      sequence: str, string of tokens from sample'''
  random_token_ids = random.choices(sample_tokens, k=seq_length)
  random_tokens = tokenizer.convert_ids_to_tokens(random_token_ids)
  random_sequence = tokenizer.convert_tokens_to_string(random_tokens)
  return random_token_ids, random_sequence

def generate_patterned_sequence(seq_length, alpha=True, lower=True, all_types=False):
  '''Generates a patterned sequence of tokens of length seq_length,
  either from lowercase letters, uppercase letters, or digits.
  Params:
      seq_length: int, length of sequence to generate
      tokenizer: transformers.PreTrainedTokenizer, tokenizer to use
      alpha: bool, whether to use letters or digits
      lower: bool, whether to use lowercase or uppercase letters
  Returns:
      pattern: str, string of tokens sampled uniformly at random'''
  if all_types:
    if seq_length > len(string.digits):
      return ValueError("Sequence length must be less than or equal to 10 to use digits")
    pattern_poss = random.choice([string.ascii_lowercase, string.ascii_uppercase, string.digits], 
                                 weights=[len(string.ascii_lowercase), len(string.ascii_uppercase), len(string.digits)])
  else:
    if alpha and lower:
      pattern_poss = string.ascii_lowercase
    elif alpha and not lower:
      pattern_poss = string.ascii_uppercase
    else:
      pattern_poss = string.digits
  start = random.choice(range(len(pattern_poss) - seq_length))
  pattern = pattern_poss[start:start+seq_length]
  return ' ' + ' '.join(list(pattern))

def repeat_sequence(num_examples, string_sequence):
  '''Repeats a token sequence num_examples times.
  Params:
      num_examples: int, number of times to repeat token sequence
      string_sequence: str, the string sequence to repeat
  Returns:
      repeated_sequence: repeated string sequence
  '''
  return string_sequence * num_examples

def repeat_sequence_w_space(num_examples, string_sequence):
  '''Repeats a token sequence num_examples times with a space 
  between each repetition.
  Params:
      num_examples: int, number of times to repeat token sequence
      string_sequence: str, the string sequence to repeat
  Returns:
      repeated_sequence: repeated string sequence
  '''
  s =""
  for _ in range(max(1, num_examples)):
      s+=string_sequence+' '
  return ' '+s.strip()

# --------------------------------------------------------------ANALYSIS FUNCTIONS-----------------------------------------------------#

from tqdm import tqdm

def repeated_sequence_analysis(string_sequence, num_repeats, model, with_space=False, compute_global=True, norm=False):
  '''Analyzes the curvature of a repeated sequence.
  Params:
      string_sequence: str, the string sequence to repeat
      num_repeats: int, number of times to repeat token sequence  
      model: HookedTransformer, the model to analyze
  Returns:
      results: dict, dictionary of results
  '''
  if with_space:
    repeated_string_sequence = repeat_sequence_w_space(num_repeats, string_sequence)
  else:
    repeated_string_sequence = repeat_sequence(num_repeats, string_sequence)
  
  repeated_tokens = model.to_tokens(repeated_string_sequence)[0]
  seq_length = int(len(repeated_tokens[1:])/num_repeats) # ignores start token 
  
  dataset_early_decoding_probs = {}
  dataset_early_decoding_rrs = {}
  
  model_logits, model_cache = model.run_with_cache(repeated_tokens, remove_batch_dim=True)
  loss = model.loss_fn(model_logits, repeated_tokens.unsqueeze(0), per_token=True)[0].cpu()

  results = {}

  for stream_index in range(seq_length):
    # print((stream_index + 1)/seq_length)
    losses = loss[stream_index::seq_length]
    logits = model_logits[0, stream_index::seq_length, :].cpu()
    if norm: 
      dataset_curvatures = layer_wise_norm_running_average(model_cache, model.cfg.n_layers, stream_index, seq_length)
    else:
      if compute_global: 
        dataset_curvatures = compute_global_curvature(model_cache, model.cfg.n_layers, stream_index, seq_length)
      else: 
        dataset_curvatures = layer_wise_curvature(model_cache, model.cfg.n_layers, stream_index, seq_length)
      
    dataset_log_probs = torch.nn.functional.log_softmax(logits.cpu(), dim=-1)
    # answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)

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

def repeated_sequence_sample_generation(num_samples, seq_len, 
                                        seq_position, num_repeats, 
                                        model, tokenizer, 
                                        how='random', compute_global=True,
                                        norm=False):
  
  sample_vectors = []
  sample_sequences = []
  sample_losses = []
  
  for i in tqdm(range(num_samples)):
    total_len = -1
    while total_len != (seq_len * num_repeats + 1):
      if how == 'random':
        _, seq = generate_random_token_sequence(seq_len, tokenizer)
      elif how == 'pattern':
        seq = generate_patterned_sequence(seq_len, alpha=True, lower=True)
      elif how == 'top100':
        _, seq = generate_sequence_from_sample(MOST_COMMON_TOKENS, seq_len, tokenizer)
      elif how == 'bottom100':
        _, seq = generate_sequence_from_sample(LEAST_COMMON_TOKENS, seq_len, tokenizer)
      else:
        raise ValueError("How Value not recognized - should be in ['random', 'patterned', 'top100', 'bottom100']")
      
      repeated_string_sequence = repeat_sequence(num_repeats, seq)
      repeated_tokens = model.to_tokens(repeated_string_sequence, prepend_bos=True)[0]
      total_len = len(repeated_tokens)
  
    seq_length = int(len(repeated_tokens[1:])/num_repeats) # ignores start token 
    
    model_logits, model_cache = model.run_with_cache(repeated_tokens, remove_batch_dim=True)
    loss = model.loss_fn(model_logits, repeated_tokens.unsqueeze(0), per_token=True)[0].cpu()
    
    results = {}
    for stream_index in range(seq_length):
      losses = loss[stream_index::seq_length]
      logits = model_logits[0, stream_index::seq_length, :].cpu()
      if norm: 
        dataset_curvatures = layer_wise_norm_running_average(model_cache, model.cfg.n_layers, stream_index, seq_length)
      else:
        if compute_global:
          dataset_curvatures = compute_global_curvature(model_cache, model.cfg.n_layers, stream_index, seq_length)
        else:
          dataset_curvatures = layer_wise_curvature(model_cache, model.cfg.n_layers, stream_index, seq_length)
      dataset_log_probs = torch.nn.functional.log_softmax(logits.cpu(), dim=-1)
      
      index_results = {
          "logits": logits,
          "losses": losses,
          "curvatures": dataset_curvatures,
          "log_probabilities": dataset_log_probs,
      }
      results[stream_index] = index_results
    y = results[seq_position]["curvatures"]
    sample_vectors.append(y)
    sample_sequences.append(seq)
    sample_losses.append(results[seq_position]["losses"])
  return sample_vectors, sample_sequences, sample_losses


def logit_lens_analysis(num_samples, tokenizer, num_repeats, seq_length, model, device, how='pattern'):
  prompts, answers = [], []
  for _ in range(num_samples):
    if how=='pattern':
      seq = generate_patterned_sequence(seq_length, alpha=True, lower=True)
    elif how=='random':
      tokens, seq = generate_random_token_sequence(seq_length, tokenizer)
    elif how=='top100':
      tokens, seq = generate_sequence_from_sample(MOST_COMMON_TOKENS, seq_length, tokenizer)
    elif how=='bottom100':
      tokens, seq = generate_sequence_from_sample(LEAST_COMMON_TOKENS, seq_length, tokenizer)
    results = repeat_sequence(num_repeats, seq)
    prompts.append(results)
    if how=='pattern':
      answers.append((seq[0:2], seq[2:4]))
    else:
      answers.append((tokenizer.decode(tokens[0]), tokenizer.decode(tokens[1])))
    
  print('prompts', prompts)
  print('answers', answers)
    
  answer_tokens = torch.tensor([[model.to_single_token(corr), model.to_single_token(incorr)] for corr, incorr in answers]).to(device)

  tokens = model.to_tokens(prompts, prepend_bos=True)
  tokens = tokens.to(device)

  original_logits, cache = model.run_with_cache(tokens)
  return {'prompts':prompts, 'answers':answers, 
          'original_logits':original_logits, 'cache':cache, 
          'answer_tokens':answer_tokens}

  # print("Per prompt logit difference:", logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True))
  # print("Average logit difference:", logits_to_ave_logit_diff(original_logits, answer_tokens).item())



def clustering_analysis(num_samples, seq_len, seq_position, num_repeats, model, tokenizer, num_clusters=4, n_components=10):
  sample_vectors, sample_sequences, _ = repeated_sequence_sample_generation(num_samples, seq_len, seq_position, num_repeats, model, tokenizer)
  sample_vectors_p, sample_sequences_p, _ = repeated_sequence_sample_generation(num_samples, seq_len, seq_position, num_repeats, model, tokenizer, how='patterned')
  sample_vectors_b, sample_sequences_b, _ = repeated_sequence_sample_generation(num_samples, seq_len, seq_position, num_repeats, model, tokenizer, how='bottom100')
  sample_vectors_t, sample_sequences_t, _ = repeated_sequence_sample_generation(num_samples, seq_len, seq_position, num_repeats, model, tokenizer, how='top100')
  
  print(np.array(sample_vectors).shape)
  print(np.array(sample_vectors_p).shape)
  print(np.array(sample_vectors_t).shape)
  print(np.array(sample_vectors_b).shape)
  
  sample_vectors = sample_vectors + sample_vectors_p + sample_vectors_t + sample_vectors_b
  sample_sequences = sample_sequences + sample_sequences_p + sample_sequences_t + sample_sequences_b
  
  np.save(f"outputs/repeat/cluster_analysis_1/sample_vectors.npy", sample_vectors)
  np.save(f"outputs/repeat/cluster_analysis_1/sample_sequences.npy", sample_vectors)
  
  scaler = StandardScaler()
  normalized_sample_vectors = scaler.fit_transform(sample_vectors)
  pca = PCA(n_components=n_components)
  pca_curvatures = pca.fit_transform(normalized_sample_vectors)
  
  KMeans_model = KMeans(n_clusters=num_clusters, random_state=0).fit(pca_curvatures)
  cluster_labels = KMeans_model.labels_
  cluster_groups = {}
  
  for i in range(num_clusters):
    cluster_groups[i] = list(np.array(sample_sequences)[cluster_labels == i])
  return cluster_groups, sample_vectors, sample_sequences, cluster_labels


def main(FLAGS):
  print(sys.argv[0])
  torch.set_grad_enabled(False)
  torch.manual_seed(0)
  np.random.seed(0)
  
  model = HookedTransformer.from_pretrained("gpt2-small", device=FLAGS.device)
  tokenizer = AutoTokenizer.from_pretrained("gpt2", device=FLAGS.device)
  num_layers = model.cfg.n_layers
  
  if FLAGS.experiment == "generate_streamlit_data":  
    norm = True
    sample_vectors, sample_sequences, sample_losses = repeated_sequence_sample_generation(FLAGS.num_trials, FLAGS.seq_length, 
                                                                                          FLAGS.seq_length - 1, FLAGS.num_repetitions, 
                                                                                          model, tokenizer, how='random', 
                                                                                          compute_global=False, norm=norm)

    sample_vectors_p, sample_sequences_p, sample_losses_p = repeated_sequence_sample_generation(FLAGS.num_trials, FLAGS.seq_length, 
                                                                                                FLAGS.seq_length - 1, FLAGS.num_repetitions, 
                                                                                                model, tokenizer, how='pattern', 
                                                                                                compute_global=False, norm=norm)
    
    sample_vectors_t, sample_sequences_t, sample_losses_t = repeated_sequence_sample_generation(FLAGS.num_trials, FLAGS.seq_length, 
                                                                                                FLAGS.seq_length - 1, FLAGS.num_repetitions, 
                                                                                                model, tokenizer, how='top100', 
                                                                                                compute_global=False, norm=norm)
    
    sample_vectors_b, sample_sequences_b, sample_losses_b = repeated_sequence_sample_generation(FLAGS.num_trials, FLAGS.seq_length, 
                                                                                                FLAGS.seq_length - 1, FLAGS.num_repetitions, 
                                                                                                model, tokenizer, how='bottom100', 
                                                                                                compute_global=False, norm=norm)
    
    
    keys_layers = list(range(num_layers))
    if not norm:
      keys_layers += ['total'] 
    ## total is the sum of the layers curvature (equivalent to the global curvature)
    
    layers_random = { str(layer):[[curv[layer] for curv in curvatures] for curvatures in sample_vectors]
                                        for layer in keys_layers}
    layers_random['loss'] = [sample_loss.cpu().tolist() for sample_loss in sample_losses]
    
    layers_pattern= {str(layer): [[curv[layer] for curv in curvatures] for curvatures in sample_vectors_p]
                                           for layer in keys_layers}
    layers_pattern['loss'] = [sample_loss.cpu().tolist() for sample_loss in sample_losses_p]
    
    layers_top100 = {str(layer): [[curv[layer] for curv in curvatures] for curvatures in sample_vectors_t]
                                           for layer in keys_layers}
    layers_top100['loss'] = [sample_loss.cpu().tolist() for sample_loss in sample_losses_t]
    
    layers_bottom100 = {str(layer): [[curv[layer] for curv in curvatures] for curvatures in sample_vectors_b]
                                           for layer in keys_layers}
    layers_bottom100['loss'] = [sample_loss.cpu().tolist() for sample_loss in sample_losses_b]
    
    os.makedirs(f"outputs/repeat/streamlit/", exist_ok=True)
    pd.Series(sample_sequences).to_csv(f"outputs/repeat/streamlit/sequences_random_n.csv", header=False)
    pd.Series(sample_sequences_p).to_csv(f"outputs/repeat/streamlit/sequences_pattern_n.csv", header=False)
    pd.Series(sample_sequences_t).to_csv(f"outputs/repeat/streamlit/sequences_top100_n.csv", header=False)
    pd.Series(sample_sequences_b).to_csv(f"outputs/repeat/streamlit/sequences_bottom100_n.csv", header=False)
    json.dump(layers_random, open(f"outputs/repeat/streamlit/layers_loss_random_n.json", "w"))
    json.dump(layers_pattern, open(f"outputs/repeat/streamlit/layers_loss_pattern_n.json", "w"))
    json.dump(layers_top100, open(f"outputs/repeat/streamlit/layers_loss_top100_n.json", "w"))
    json.dump(layers_bottom100, open(f"outputs/repeat/streamlit/layers_loss_bottom100_n.json", "w"))

  elif FLAGS.experiment == "run_clustering_analysis":
    cluster_groups, sample_vectors, sample_sequences, cluster_labels = clustering_analysis(FLAGS.num_trials, FLAGS.seq_length, FLAGS.seq_length - 1, 
                        FLAGS.num_repetitions, model, tokenizer, num_clusters=FLAGS.num_clusters)
    os.makedirs(f"outputs/repeat/{FLAGS.name}/", exist_ok=True)
    json.dump(cluster_groups, open(f"outputs/repeat/{FLAGS.name}/cluster_groups.json", "w"))
    np.save(f"outputs/repeat/{FLAGS.name}/sample_vectors.npy", sample_vectors)
    np.save(f"outputs/repeat/{FLAGS.name}/sample_sequences.npy", sample_vectors)
    plot_clusters(sample_vectors, sample_sequences, cluster_labels, show=False, save_fig=True, 
                  save_path=f"outputs/repeat/{FLAGS.name}/cluster_analysis.html")
    plot_clusters(sample_vectors, sample_sequences, cluster_labels, hover=True, show=False, save_fig=True, 
                  save_path=f"outputs/repeat/{FLAGS.name}/cluster_analysis_hover.html")
  
  elif FLAGS.experiment == "run_logit_lens_analysis":
    how = FLAGS.logit_lens_how
    os.makedirs(f"outputs/repeat/{FLAGS.name}/{how}/", exist_ok=True)
    for i in range(FLAGS.num_repetitions):
      results = logit_lens_analysis(FLAGS.num_trials, tokenizer, i, FLAGS.seq_length,
                                    model, device=FLAGS.device, how=how)
      plot_logit_lens(results['cache'], results['prompts'], 
                      results['answer_tokens'], model, 
                      cumulative=True, show=False, 
                      save_fig=True, save_path=f"outputs/repeat/{FLAGS.name}/{how}/logit_lens-seq_len={FLAGS.seq_length}-num_reps={i}-cum.png")
  else:
    for i in range(FLAGS.num_trials):
      
      if FLAGS.experiment == "run_randomized_repeat":
        _, seq = generate_random_token_sequence(FLAGS.seq_length, tokenizer)
        print("Randomly Generated Sequence: ", seq)
      
      elif FLAGS.experiment == "run_patterned_repeat":
        seq = generate_patterned_sequence(FLAGS.seq_length, alpha=True, lower=True)
        print("Patterned Generated Sequence: ", seq)
      
      elif FLAGS.experiment == "run_top100_repeat":
        _, seq = generate_sequence_from_sample(MOST_COMMON_TOKENS, FLAGS.seq_length, tokenizer)
        print("Top 100 Generated Sequence: ", seq)
        
      elif FLAGS.experiment == "run_bottom100_repeat":
        _, seq = generate_sequence_from_sample(LEAST_COMMON_TOKENS, FLAGS.seq_length, tokenizer)
        print("Bottom 100 Generated Sequence: ", seq) 
        
      else:
        raise ValueError("Experiment not recognized")
      
      results = repeated_sequence_analysis(seq, FLAGS.num_repetitions, model)
        
      os.makedirs(f"outputs/repeat/{FLAGS.name}/seq_len={FLAGS.seq_length}-trial={i}", exist_ok=True)
      
      with open(f"outputs/repeat/{FLAGS.name}/seq_len={FLAGS.seq_length}-trial={i}/sequence.txt", 'w') as file:
        file.write(seq)
      plot_rr_vs_repetitions(results, FLAGS.seq_length - 1, offset=FLAGS.seq_length*2, layer=8, show=False, save_fig=True, 
                            save_path=f"outputs/repeat/{FLAGS.name}/seq_len={FLAGS.seq_length}-trial={i}/rr_vs_repetitions_trial.png") 
      plot_curvature_vs_repetitions(results, FLAGS.seq_length - 1, offset=FLAGS.seq_length*2, show=False, save_fig=True, 
                                    save_path=f"outputs/repeat/{FLAGS.name}/seq_len={FLAGS.seq_length}-trial={i}/curvature_vs_repetitions.png")
      plot_loss_vs_repetitions(results, FLAGS.seq_length - 1, offset=FLAGS.seq_length*2, show=False, save_fig=True, 
                              save_path=f"outputs/repeat/{FLAGS.name}/seq_len={FLAGS.seq_length}-trial={i}/loss_vs_repetitions.png")
      plot_curvature_loss_vs_repetitions(results, FLAGS.seq_length - 1, offset=FLAGS.seq_length*2, show=False, save_fig=True, 
                              save_path=f"outputs/repeat/{FLAGS.name}/seq_len={FLAGS.seq_length}-trial={i}/curvature_loss_vs_repetitions.png")
       
       
if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  parser = argparse.ArgumentParser(
              prog='Repeat Sequence Experiment',
              description='tests curvature as repeat sequence length increases')
  parser.add_argument('--seq-length', default=5, type=int, help='number of tokens per in-context example')
  parser.add_argument('--num-trials', default=10, type=int, help='number of trials to run with this seq length')
  parser.add_argument('--num-repetitions', default=100, type=int, help='maximum number of repeated in context examples')
  parser.add_argument('--name', default='random_sequences', type=str, help='name of experiment')
  parser.add_argument('--logit-lens-how', default='pattern', type=str, help='how to generate logit lens data (random, pattern, top100, bottom100)')
  parser.add_argument('--experiment', default='run_randomized_repeat', type=str, 
                      help='experiment to run (run_randomized_repeat, run_patterned_repeat, run_top100_repeat, run_bottom100_repeat, run_clustering_analysis, run_logit_lens_analysis, generate_streamlit_data)')
  parser.add_argument('--num-clusters', default=4, type=int, help='number of clusters to use in clustering analysis')
  parser.add_argument("--device", default="cuda:0", type=str, help="device to run experiment on")
  
  FLAGS = parser.parse_args()
  main(FLAGS)