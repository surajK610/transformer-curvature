import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="Streamlit App", page_icon="🧊", layout="wide")

def intro():  

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
  
def random_sequences():
  import streamlit as st
  

def patterned_sequences():
  pass

def top100_sequences():
  pass

def bottom100_sequences():
  pass

def clustering():
  pass

page_names_to_funcs = {
  "Introduction": intro,
  "Random Sequences": random_sequences,
  "Patterned Sequences": patterned_sequences,
  "Top 100 Sequences": top100_sequences,
  "Bottom 100 Sequences": bottom100_sequences,
  "Clustering": clustering,
}

demo_name = st.sidebar.selectbox("Select a visualization", list(page_names_to_funcs.keys()))
page_names_to_funcs[demo_name]()