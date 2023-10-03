import streamlit as st
import pandas as pd
import json
import numpy as np
from utils.plotting_functions import plot_layer_curvature_loss_vs_repetitions
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
# Set page config
st.set_page_config(page_title="Streamlit App", page_icon="🧊", layout="wide")

N_COMPONENTS = 10
N_CLUSTERS = 4
RANDOM_STATE = 0

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

def random_sequences(offset=(0, 100)):
  st.write("## Random Sequences")
  st.write("### Mean Layer-Wise Curvature Random Sequences")
  dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_random.json')
  plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset)
  
def patterned_sequences(offset=(0, 100)):
  st.write("## Patterned Sequences")
  st.write("### Mean Layer-Wise Curvature Patterned Sequences")
  dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_pattern.json')
  plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset)

def top100_sequences(offset=(0, 100)):
  st.write("## Top 100 Sequences")
  st.write("### Mean Layer-Wise Curvature Top 100 Sequences")
  
  dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_top100.json')
  plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset)

def bottom100_sequences(offset=(0, 100)):
  st.write("## Bottom 100 Sequences")
  st.write("### Mean Layer-Wise Curvature Bottom 100 Sequences")
  dict_mean_layers = compute_mean_layers('outputs/repeat/streamlit/layers_loss_bottom100.json')
  plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset)
  
def plot_layer_curvature_loss_vs_repetitions(dict_mean_layers, offset=(0, 100)):
  keys = list(dict_mean_layers.keys())
  keys.remove('loss')
  x = list(range(len(dict_mean_layers['loss'])))

  for key in keys:
    fig, ax1 = plt.subplots()
    ax1.scatter(x[offset[0]:offset[1]], dict_mean_layers[key][offset[0]:offset[1]], label=f'curvature_layer_{key}', color='b')
    ax1.set_xlabel('Repetitions')
    ax1.set_ylabel(f'layer {key} curvature', color='b')
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

@st.cache_data
def compute_mean_layers(file):
  layer_dict = json.load(open(file, "r"))
  for key, value in layer_dict.items():
    layer_dict[key] = np.mean(value, axis=0)
  return layer_dict

page_names_to_funcs = {
  "Introduction": intro,
  "Random Sequences": random_sequences,
  "Patterned Sequences": patterned_sequences,
  "Top 100 Sequences": top100_sequences,
  "Bottom 100 Sequences": bottom100_sequences,
  "Clustering": clustering,
}

demo_name = st.sidebar.selectbox("Select a visualization", list(page_names_to_funcs.keys()))
offset = st.sidebar.slider("Range", 0, 100, (0, 100))

page_names_to_funcs[demo_name](offset)