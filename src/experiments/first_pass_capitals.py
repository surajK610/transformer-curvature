import warnings
from transformer_lens import HookedTransformer

import torch
import matplotlib.pyplot as plt
from .utils.curvature_utils import compute_global_curvature_nonrepeating
from .utils.plotting_functions import plot_curvature_vs_icl_examples, plot_global_curvature_dissected
import argparse
import random
import os
from datasets import lang0, lang1, lang2, lang3, counterlang1, counterlang2, counterlang3, an0, an1, an2, an3, counteran1, counteran2, counteran3

  
def first_pass_capital_analysis(model):
  '''Analyzes the curvature of a capital prediction with different numbers of in context examples.'''
  datasets = [lang0, lang1, lang2, lang3, counterlang1, counterlang2, counterlang3, an0, an1, an2, an3, counteran1, counteran2, counteran3]
  dataset_curvatures = []
  dataset_mlp_curvatures = []
  dataset_attn_curvatures = []

  for ds in datasets:
    curvatures = []
    mlp_curvatures = []
    attn_curvatures = []
    for prompt in ds:
      gpt2_tokens = model.to_tokens(prompt)
      _ , gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)
      curvatures.append(compute_global_curvature_nonrepeating(gpt2_cache, model.cfg.n_layers, -1))
      mlp_curvatures.append(compute_global_curvature_nonrepeating(gpt2_cache, model.cfg.n_layers, -1, include_mlps=True, include_attn=False))
      attn_curvatures.append(compute_global_curvature_nonrepeating(gpt2_cache, model.cfg.n_layers, -1, include_mlps=False, include_attn=True))

    dataset_curvatures.append(curvatures)
    dataset_mlp_curvatures.append(mlp_curvatures)
    dataset_attn_curvatures.append(attn_curvatures)
  return dataset_curvatures, dataset_mlp_curvatures, dataset_attn_curvatures
  
  
    
def main(FLAGS):
  torch.set_grad_enabled(False)
  torch.manual_seed(0)
  
  model = HookedTransformer.from_pretrained("gpt2-small", device=FLAGS.device)
  
  dataset_curvatures, dataset_mlp_curvatures, dataset_attn_curvatures = first_pass_capital_analysis(model)
  plot_curvature_vs_icl_examples(dataset_curvatures, show=False, save_fig=True, save_path="outputs/first_pass_capitals/curvature_vs_icl_examples.png")
  plot_global_curvature_dissected(dataset_curvatures, dataset_mlp_curvatures, dataset_attn_curvatures,
                                    show=False, save_fig=True, save_path="outputs/first_pass_capitals/curvature_dissected.png")
  
      
  
if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  parser = argparse.ArgumentParser(
              prog='First Pass at Capitals',
              description='computes curvature as increase number of ICL examples for capital predictions')  
  # parser.add_argument("--use-data-source", default=False, type=bool, 
  #       help="whether to use a data source or generate random sequence")
  # parser.add_argument("--data-path", default="datasets/repeat_sequence.txt", type=str, 
  #       help="data path where sequence is located")
  parser.add_argument("--device", default="cuda:0", type=str, help="device to run experiment on")
  
  FLAGS = parser.parse_args()
  main(FLAGS)
  