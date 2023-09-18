import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import torch

import plotly.express as px

## Some plotting functions 
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

## Computes the global curvature of the model (distance traveled / displacement)
def compute_global_curvature(cache, total_layers, stream_idx, sequence_length, include_mlps=True, include_attn=True):
  curvatures = []
  for str_idx in range(stream_idx, len(cache["resid_pre", 0]), sequence_length):
    displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][str_idx] - cache["resid_pre", 0][str_idx])))
    distance = 0.0
    for layer in range(total_layers):
      if include_mlps:
        distance+=torch.linalg.norm(cache['mlp_out', layer][str_idx])
      if include_attn:
        distance+=torch.linalg.norm(cache['attn_out', layer][str_idx])

    curvatures.append((distance/displacement).cpu().numpy())
  return curvatures

## Early decodes the model at a given layer
def early_decode(model, cache, layer, mid=False):
  return model.unembed(cache[f'blocks.{layer}.hook_resid_{"mid" if mid else "post"}'].unsqueeze(0))


## Computes the kl distribution between the probability distributions of the ith and i+1th token predictions
def compute_prob_kls(prob_dists):
  kls = []
  for i in range(len(prob_dists) - 1):
    kls.append(torch.nn.functional.kl_div(prob_dists[i+1], prob_dists[i], log_target=True).cpu().numpy())
  return kls

## Computes the kl distribution between the probability distributions of the ith and i+gapth token predictions
def compute_long_range_kls(prob_dists, gap=40):
  kls = []
  for i in range(len(prob_dists) - gap):
    kls.append(torch.nn.functional.kl_div(prob_dists[i+gap], prob_dists[i], log_target=True).cpu().numpy())
  return kls

