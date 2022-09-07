import streamlit as st

st.markdown("# BERTopic Modeling 🎈")
st.sidebar.markdown("#BERTopic Modeling 🎈")

## layout ##

#sidebar
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'BERTopic Model Approaches',
    ('Default', 'Sentence Transformer', 'Guided Topic Modeling', 'Supervised', 'Semi-supervised')
)

add_selectbox = st.sidebar.selectbox(
    'Dimension',
    ('UMAP', 'PCA', 'Truncated SVD')
)

add_selectbox = st.sidebar.selectbox(
    'Visualization',
    ('Topics Word Scores Barchart', 'Intertopic Distance Map', 'Hierarchical Clustering', 'Similarity Heat Map', 'Topic Frequency Distribution')
)

