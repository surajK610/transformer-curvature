import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities

from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import torch

import plotly.express as px

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
  '''Plots a tensor as an image.'''
  px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
  '''Plots a tensor as a line plot.'''
  px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
  '''Plots a tensor as a scatter plot.'''
  x = utils.to_numpy(x)
  y = utils.to_numpy(y)
  px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


def compute_global_curvature(cache, total_layers, stream_idx, sequence_length, include_mlps=True, include_attn=True):
  '''Computes the global curvature of the residual stream of a model with repeating sequences.'''
  curvatures = []
  for str_idx in range(stream_idx, len(cache["resid_pre", 0]), sequence_length):
    if include_mlps and include_attn:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][str_idx] - cache["resid_pre", 0][str_idx])))
    elif include_mlps:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][str_idx] - cache["resid_mid", 0][str_idx])))
    elif include_attn:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_mid", total_layers - 1][str_idx] - cache["resid_pre", 0][str_idx])))
    else:
      raise ValueError("Must include either MLPs or attention in the curvature calculation.")
    
    distance = 0.0
    for layer in range(total_layers):
      if include_mlps:
        distance+=torch.linalg.norm(cache['mlp_out', layer][str_idx])
      if include_attn:
        distance+=torch.linalg.norm(cache['attn_out', layer][str_idx])

    curvatures.append((distance/displacement).cpu().numpy())
  return curvatures

def compute_global_curvature_nonrepeating(cache, total_layers, stream_idx, include_mlps=True, include_attn=True):
  '''Computes the global curvature of the residual stream of a model.'''
  displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][stream_idx] - cache["resid_pre", 0][stream_idx])))
  distance = 0.0
  for layer in range(total_layers):
    if include_mlps:
      distance+=torch.linalg.norm(cache['mlp_out', layer][stream_idx])
    if include_attn:
      distance+=torch.linalg.norm(cache['attn_out', layer][stream_idx])

  return (distance/displacement).cpu().numpy()
    
def early_decode(model, cache, layer, mid=False):
  '''Decodes the residual stream of a model at a given layer, returning the decoded tokens.'''
  return model.unembed(cache[f'blocks.{layer}.hook_resid_{"mid" if mid else "post"}'].unsqueeze(0))

def compute_prob_kls(prob_dists):
  '''Computes the kl distribution between the probability distributions of the ith and i+1th token predictions'''
  kls = []
  for i in range(len(prob_dists) - 1):
    kls.append(torch.nn.functional.kl_div(prob_dists[i+1], prob_dists[i], log_target=True).cpu().numpy())
  return kls

def compute_long_range_kls(prob_dists, gap=40):
  '''Computes the kl distribution between the probability distributions of the ith and i+gapth token predictions'''
  kls = []
  for i in range(len(prob_dists) - gap):
    kls.append(torch.nn.functional.kl_div(prob_dists[i+gap], prob_dists[i], log_target=True).cpu().numpy())
  return kls


def layer_wise_curvature(cache, total_layers, stream_idx, sequence_length, include_mlps=True, include_attn=True, cumulative=False):
  curvatures = []
  for str_idx in range(stream_idx, len(cache["resid_pre", 0]), sequence_length):
    if include_mlps and include_attn:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][str_idx] - cache["resid_pre", 0][str_idx])))
    elif include_mlps:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][str_idx] - cache["resid_mid", 0][str_idx])))
    elif include_attn:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_mid", total_layers - 1][str_idx] - cache["resid_pre", 0][str_idx])))
    else:
      raise ValueError("Must include either MLPs or attention in the curvature calculation.")
    
    dict_distance = {}
    if cumulative:
        distance = 0.0
    for layer in range(total_layers):
      if not cumulative:
          distance = 0.0
    
      if include_mlps:
        distance+=torch.linalg.norm(cache['mlp_out', layer][str_idx])
      if include_attn:
        distance+=torch.linalg.norm(cache['attn_out', layer][str_idx])
      dict_distance[layer] = (distance/displacement).clone().detach().item()
    dict_distance['total'] = sum(dict_distance.values())
    curvatures.append(dict_distance)
  return curvatures



def logit_attribution():
  pass