import streamlit as st
import pandas as pd
import json
import numpy as np
from experiments.utils.plotting_functions import plot_layer_curvature_loss_vs_repetitions
from experiments.utils.curvature_utils import residual_stack_to_logit_diff
from experiments.repeat import logit_lens_analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import transformer_lens.utils as tutils

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
# Set page config
st.set_page_config(page_title="Streamlit App", page_icon="🧊", layout="wide")

N_COMPONENTS = 10
N_CLUSTERS = 4
RANDOM_STATE = 0
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2", device="cpu")

def intro(offset=0):  

  st.write("# Curvature Repeat Visualizations")

  st.sidebar.success("Select a visualization:")

  st.markdown(
    """
    The experiments in this app show layerwise curvature modeled as a function of the number of repeated sequences. 
    
    We test sequences that are 
    1. **random** (i.e. randomly generated tokens)
    2. **patterned** (i.e. patterned tokens that are individual characters)
    3. **top100** (i.e. patterned tokens that are the top 100 most frequent tokens)
    4. **bottom100** (i.e. patterned tokens that are the bottom 100 most frequent tokens)
    
    The number of repeated sequences is varied from 1 to 100. We use [transformer-lens](https://github.com/neelnanda-io/TransformerLens)
    """)

# -------------------------------------  ORIGINAL REPEATED SEQ FUNCTIONS --------------------------------------
def random_sequences(offset=(0, 100)):
  st.write("## Random Sequences")
  option = st.selectbox(
    'Would you like to visualize curvature or running norm?',
    ('curvature', 'norm'))
  if option == 'norm':
    st.write("### Mean Layer-Wise Running Norm Random Sequences")
    dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_random_n.json')
  else:
    st.write("### Mean Layer-Wise Curvature Random Sequences")
    dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_random.json')
  plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset)
  
def patterned_sequences(offset=(0, 100)):
  st.write("## Patterned Sequences")
  option = st.selectbox(
    'Would you like to visualize curvature or running norm?',
    ('curvature', 'norm'))
  if option == 'norm':
    st.write("### Mean Layer-Wise Running Norm Patterned Sequences")
    dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_pattern_n.json')
  else:
    st.write("### Mean Layer-Wise Curvature Patterned Sequences")
    dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_pattern.json')
  plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset)

def top100_sequences(offset=(0, 100)):
  st.write("## Top 100 Sequences")
  option = st.selectbox(
    'Would you like to visualize curvature or running norm?',
    ('curvature', 'norm'))
  if option == 'norm':
    st.write("### Mean Layer-Wise Running Norm Top 100 Sequences")
    dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_top100_n.json')
  else:
    st.write("### Mean Layer-Wise Curvature Top 100 Sequences")
    dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_top100.json')
  plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset)

def bottom100_sequences(offset=(0, 100)):
  st.write("## Bottom 100 Sequences")
  option = st.selectbox(
    'Would you like to visualize curvature or running norm?',
    ('curvature', 'norm'))
  if option == 'norm':
    st.write("### Mean Layer-Wise Running Norm Bottom 100 Sequences")
    dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_bottom100_n.json')
  else:
    st.write("### Mean Layer-Wise Curvature Bottom 100 Sequences")
    dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_bottom100.json')
  plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset)
    
def logit_lens(offset=(0, 100)):
  st.write("## Logit Lens Analysis")
  option = st.selectbox(
    'How would you like sequences to be generated?',
    ('pattern', 'random', 'top100', 'bottom100'))
  how=option 
   
  seq_length=4
  for i in range(5):
    results = logit_lens_analysis(40, tokenizer, i, seq_length,
                                  model, device="cpu", how=how)
    answer_residual_directions = model.tokens_to_residual_directions(results['answer_tokens'])
    ## num_samples decoding vectors
    logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    accumulated_residual, labels = results['cache'].accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
    logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, results['cache'], logit_diff_directions, results['prompts'])
    
    ax1.plot(np.arange(model.cfg.n_layers*2+1)/2, tutils.to_numpy(logit_lens_logit_diffs))
    ax1.set_xticks(np.arange(model.cfg.n_layers*2+1)/2)
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_title(f"Cumulative")
    
    per_layer_residual, labels = results['cache'].decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, results['cache'], logit_diff_directions, results['prompts'])
    ax2.plot(np.arange(model.cfg.n_layers*2+2)/2, tutils.to_numpy(per_layer_logit_diffs))
    ax2.set_xticks(np.arange(model.cfg.n_layers*2+2)/2)
    ax2.set_xticklabels(labels, rotation=90)
    ax2.set_title(f"Layerwise")
    fig.suptitle(f'Repetitions={i}')
    st.pyplot(fig)
    
def clustering(offset=(0, 100)):
  sample_vectors_r = json.load(open('outputs/repeat/streamlit/layers_loss_random.json', "r"))['total']
  sample_vectors_p = json.load(open('outputs/repeat/streamlit/layers_loss_pattern.json', "r"))['total']
  sample_vectors_t = json.load(open('outputs/repeat/streamlit/layers_loss_top100.json', "r"))['total']
  sample_vectors_b = json.load(open('outputs/repeat/streamlit/layers_loss_bottom100.json', "r"))['total']
  
  sample_sequences_r = pd.read_csv('outputs/repeat/streamlit/sequences_random.csv', names=['seq'])['seq'].tolist()
  sample_sequences_p = pd.read_csv('outputs/repeat/streamlit/sequences_pattern.csv', names=['seq'])['seq'].tolist()
  sample_sequences_t = pd.read_csv('outputs/repeat/streamlit/sequences_top100.csv', names=['seq'])['seq'].tolist()
  # sample_sequences_b = pd.read_csv('outputs/repeat/streamlit/sequences_bottom100.csv', names=['seq'])['seq'].tolist()
  sample_sequences_b = []

  # Open the data file and read it line by line
  with open('outputs/repeat/streamlit/sequences_bottom100.csv', 'r') as file:
      for line in file:
          try:
            # Attempt to parse the line and add it to the list of valid rows
            row_data = line.strip().split(',')[-1]  # Modify this line as needed for your data format
            sample_sequences_b.append(row_data)
          except ValueError:
            pass
              # Handle the error (e.g., skip the row, log the error, etc.)
              # sample_sequences_b.append('UNK')
  
  sample_vectors = np.concatenate((sample_vectors_r, sample_vectors_p, sample_vectors_t, sample_vectors_b))
  sample_sequences = sample_sequences_r + sample_sequences_p + sample_sequences_t + sample_sequences_b
  
  scaler = StandardScaler()
  normalized_sample_vectors = scaler.fit_transform(sample_vectors)
  pca = PCA(n_components=N_COMPONENTS)
  pca_curvatures = pca.fit_transform(normalized_sample_vectors)
  
  KMeans_model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE).fit(pca_curvatures)
  cluster_labels = KMeans_model.labels_
  cluster_groups = {}
  
  for i in range(N_CLUSTERS):
    cluster_groups[i] = list(np.array(sample_sequences)[cluster_labels == i])
  
  df_pca = pd.DataFrame(pca_curvatures[:, :3], columns=['PC1', 'PC2', 'PC3'])
  df_pca['Cluster'] = cluster_labels 
  
  fig = px.scatter_3d(
      df_pca, x='PC1', y='PC2', z='PC3',
      title='Clustered Curvature Graph',
      labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
      hover_name=sample_sequences, 
      color='Cluster'
      )
  st.plotly_chart(fig, use_container_width=True)

# ------------------------- DATA READING AND PROCESSING -----------------------------

@st.cache_data
def compute_mean_layers_lookup(file):
  layer_dict = json.load(open(file, "r"))
  for key, value in layer_dict.items():
    if key == "loss":
      layer_dict[key] = np.mean(value, axis=0)
    else:
      layer_dict[key]["running_norm"] = np.mean(value["running_norm"], axis=0)
      layer_dict[key]["consec_kls"] = np.mean(value["consec_kls"], axis=0)
  return layer_dict

@st.cache_data
def compute_mean_layers(file):
  layer_dict = json.load(open(file, "r"))
  for key, value in layer_dict.items():
    layer_dict[key] = np.mean(value, axis=0)
  return layer_dict

# ----------------------------------- LOOKUP SEQUENCES -------------------------------------

def lookup_sequences(offset=(0, 100)):
  st.write("## Lookup Task Sequences")
  option = st.selectbox(
    'Would you like to visualize mlp/attn outs or resid post?',
    ('mlp/attn out', 'resid post'))
  params = st.selectbox(
    'Which set of parameters would you like to use?',
    ('Rand Seq Len = 0, Key Len = 4, Dict Size = 1', 'Rand Seq Len = 3, Key Len = 1, Dict Size = 1', 'Rand Seq Len = 3, Key Len = 1, Dict Size = 2', 
     'Rand Seq Len = 3, Key Len = 1, Dict Size = 3'))
  st.write("### Mean Layer-Wise Metrics Among 50 Examples")
  if params == 'Rand Seq Len = 0, Key Len = 4, Dict Size = 1':
    if option == 'mlp/attn out':
      dict_mean_layers = compute_mean_layers_lookup(f'outputs/lookup/streamlit/layer_wise_analysis_50_outs_cont.json')
    else:
      dict_mean_layers = compute_mean_layers_lookup(f'outputs/lookup/streamlit/layer_wise_analysis_50_resids_cont.json')
  elif params == "Rand Seq Len = 3, Key Len = 1, Dict Size = 1":
    if option == 'mlp/attn out':
      dict_mean_layers = compute_mean_layers_lookup(f'outputs/lookup/streamlit/layer_wise_analysis_50_outs.json')
    else:
      dict_mean_layers = compute_mean_layers_lookup(f'outputs/lookup/streamlit/layer_wise_analysis_50_resids.json')
  elif params == "Rand Seq Len = 3, Key Len = 1, Dict Size = 2":
    if option == 'mlp/attn out':
      dict_mean_layers = compute_mean_layers_lookup(f'outputs/lookup/streamlit/layer_wise_analysis_50_outs_sd_2.json')
    else:
      dict_mean_layers = compute_mean_layers_lookup(f'outputs/lookup/streamlit/layer_wise_analysis_50_resids_sd_2.json')
  else:
    if option == 'mlp/attn out':
      dict_mean_layers = compute_mean_layers_lookup(f'outputs/lookup/streamlit/layer_wise_analysis_50_outs_sd_3.json')
    else:
      dict_mean_layers = compute_mean_layers_lookup(f'outputs/lookup/streamlit/layer_wise_analysis_50_resids_sd_3.json')
  
  plot_layer_lookup_metrics(dict_mean_layers, offset)


# -------------------------------- ORGINIZATIONAL FUNCTIONS ---------------------------------------
def repeated_sequences(offset=0):
  page_names_to_funcs = {
    "Random Sequences": random_sequences,
    "Patterned Sequences": patterned_sequences,
    "Top 100 Sequences": top100_sequences,
    "Bottom 100 Sequences": bottom100_sequences,
    "Logit Lens Analysis": logit_lens,
    "Clustering": clustering,
    }
  demo_name = st.selectbox("Select a repeated sequence visualization", list(page_names_to_funcs.keys()))
  page_names_to_funcs[demo_name](offset)

def lookup_task(offset=0):
  page_names_to_funcs = {
    "Layerwise Analysis": lookup_sequences,
    # "Logit Lens Analysis": logit_lens,
    }
  demo_name = st.selectbox("Select a lookup sequence visualization", list(page_names_to_funcs.keys()))
  page_names_to_funcs[demo_name](offset)
  
# ----------------------------------- PLOTTING FUNCTIONS -------------------------------------
def plot_layer_lookup_metrics(dict_mean_layers, offset=(0, 100)):
  keys = list(dict_mean_layers.keys())
  keys.remove('loss')
  x = list(range(len(dict_mean_layers['loss'])))

  for key in keys:
    fig, ax1 = plt.subplots()
    ax1.scatter(x[offset[0]:offset[1]], dict_mean_layers[key]['running_norm'][offset[0]:offset[1]], label=f'running_norm_layer_{key}', color='b')
    ax1.set_xlabel('Repetitions')
    ax1.set_ylabel(f'layer {key} running norm', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax3 = ax1.twinx()
    
    ax3.scatter(x[offset[0]:offset[1]], dict_mean_layers['loss'][offset[0]:offset[1]], label='loss', color='r')
    ax3.set_ylabel('loss', color='r')
    ax3.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    
    ax2.scatter(x[offset[0]:offset[1]], dict_mean_layers[key]['consec_kls'][offset[0]:offset[1]], label=f'consec_kls_layer_{key}', color='g')
    ax2.set_ylabel(f'layer {key} kl div', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
      
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    lines =  lines1 + lines2 + lines3
    labels =  labels1 + labels2 + labels3
    
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title(f"Layer {key}")
    st.pyplot(fig)

def plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset=(0, 100), curvature=True):
  keys = list(dict_mean_layers.keys())
  keys.remove('loss')
  x = list(range(len(dict_mean_layers['loss'])))
  prefix = "curvature" if curvature else "running_norm"
    
  for key in keys:
    fig, ax1 = plt.subplots()
    ax1.scatter(x[offset[0]:offset[1]], dict_mean_layers[key][offset[0]:offset[1]], label=f'{prefix}_layer_{key}', color='b')
    ax1.set_xlabel('Repetitions')
    ax1.set_ylabel(f'layer {key} {prefix}', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    
    ax2.scatter(x[offset[0]:offset[1]], dict_mean_layers['loss'][offset[0]:offset[1]], label='loss', color='r')
    ax2.set_ylabel('loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
      
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
      
    lines =  lines1 + lines2
    labels =  labels1 + labels2
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title(f"Layer {key}")
    st.pyplot(fig)
    
page_names_to_funcs = {
  "Introduction": intro,
  "Repeated Sequences": repeated_sequences, 
  "Lookup Task": lookup_task
}

demo_name = st.sidebar.selectbox("Select a visualization", list(page_names_to_funcs.keys()))
offset = st.sidebar.slider("Range", 0, 100, (0, 100))


page_names_to_funcs[demo_name](offset)