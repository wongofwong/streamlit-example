import streamlit as st

st.markdown("# Data Manager - Topic Modeling ")
st.sidebar.markdown("# Data Manager - Topic Modeling ")

## layout ##

#sidebar
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'BERTopic Model Approaches',
    ('Default', 'Sentence Transformer', 'Guided Topic Modeling', 'Supervised', 'Semi-supervised')
)

add_selectbox = st.sidebar.selectbox(
    'Dimensionality Reduction',
    ('UMAP', 'PCA', 'Truncated SVD')
)

# add_selectbox = st.sidebar.selectbox(
#     'Visualization',
#     ('Topics Word Scores Barchart', 'Intertopic Distance Map', 'Hierarchical Clustering', 'Similarity Heat Map', 'Topic Frequency Distribution')
# )

viz_options = ['Topics Word Scores Barchart', 'Intertopic Distance Map', 'Hierarchical Clustering']

viz_selected = st.sidebar.selectbox('Visualization', options = viz_options)

if viz_selected == 'Topics Word Scores Barchart':
    st.markdown("[Section 1: Topics Barchart](#section-1)")
elif viz_selected == 'Intertopic Distance Map':
    st.markdown("[Section 2: Topics Clusters](#section-2)")
elif viz_selected == 'Hierarchical Clustering':   
    st.markdown("[Section 3: Hierarchical Clusters](#section-3)")
    
# import streamlit as st
import streamlit.components.v1 as components

# # bootstrap 4 collapse example
# components.html(
#     """
#     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
#     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
#     <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
#     <div id="accordion">
#       <div class="card">
#         <div class="card-header" id="headingOne">
#           <h5 class="mb-0">
#             <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
#             Topic Model Custom Classes
            
       
            
#             </button>
#           </h5>
#         </div>
#         <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
#           <div class="card-body">
#             seed_topic_list=['Minestrone with beef', 'Beef chili', 'Oatmeal with flax seeds and apple', 'Blueberry shortcake', 'Roast beef sandwich', 'Turkey Sandwich', 'Classic Caesar salad', 'Mixed berry shortcake', 'Veggie sandwich', 'Classic Caesar salad with edamame', 'Oatmeal with diced apple', 'Classic minestrone with leafy greens', 'Chocolate chip cookies with pecans', 'Lasagna with meat sauce', 'Strawberry shortcake', 'Vegetarian pizza', 'Chocolate chip cookies', 'Oatmeal with diced apple and pistachios', 'Classic Caesar salad with chicken', 'Vegetarian lasagna with spinach', 'Classic minestrone', 'Chicken and bean chili', 'Vegetarian lasagna', 'Beef Steak', 'Tilapia', 'Banana', 'Green beans', 'pecans', 'Baked Beans', 'Cherries', 'Macaroni and cheese', 'brown rice', 'Brownie ', 'White Bread', 'Carrots', 'green peas', 'cornbread', 'Potato', 'Canned pear halves', 'Corn', 'String cheese', 'Fried Chicken Breast', 'Fried eggs', 'Peanut Butter', 'Pork roast', 'chicken patty', 'Pear ', 'tapioca pudding, vanilla and other flavors, homemade', 'rice pudding, plain, made with egg', 'plain yogurt, lowfat', 'oatmeal, regular cooking', 'macaroni noodles', 'angel hair pasta', 'chocolate pudding, from mix, instant', 'garbanzo beans', 'cottage cheese']
#           </div>
#         </div>
#       </div>
#       <div class="card">
#         <div class="card-header" id="headingTwo">
#           <h5 class="mb-0">
#             <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
#             Topic Model Output Visualizations
#             </button>
#           </h5>
#         </div>
#         <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
#           <div class="card-body">
#             Topic Model Output Visualizations
#           </div>
#         </div>
#       </div>
#     </div>
#     """,
#     height=600,
# )

from bertopic import BERTopic
import pandas as pd


#@st.cache(persist=True)
# @st.experimental_memo(persist="disk")
# def run_topic_model():
#     docs = opsis_labels
#     topic_model = BERTopic(seed_topic_list=opsis_labels)
#     topics, probs = topic_model.fit_transform(docs)
#     return data

# run_topic_model()
df = pd.read_json('clipsubset.json')
captions = df['caption'].values

docs = captions
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

fig_hier = topic_model.visualize_hierarchy(width=2000, height=2000)

fig_term_rank = topic_model.visualize_term_rank()

fig_heatmap = topic_model.visualize_heatmap(width=1500, height=1500)

fig_topics = topic_model.visualize_topics(width=1500, height=800, top_n_topics=10)

fig_docs_clusters = topic_model.visualize_documents(docs, topics = [0, 1, 2, 3, 4, 5, 6, 8, 28, 77, 80, 91, 23, 25])

fig_barchart = topic_model.visualize_barchart(top_n_topics=20, width=200, height=200)

#fig_docs = topic_model.visualize_documents(docs)

fig_freq = topic_model.get_topic_freq()

# from dash import Dash, dcc, html, Input, Output
# import plotly.express as px
# import json

# fig = px.line(
#     x=["a","b","c"], y=[1,3,2], # replace with your own data source
#     title="sample figure", height=325
# )

# app = Dash(__name__)

# app.layout = html.Div([
#     html.H4('Displaying figure structure as JSON'),
#     dcc.Graph(id="graph", figure=fig),
#     dcc.Clipboard(target_id="structure"),
#     html.Pre(
#         id='structure',
#         style={
#             'border': 'thin lightgrey solid', 
#             'overflowY': 'scroll',
#             'height': '275px'
#         }
#     ),
# ])

# import plotly as pt

# import plotly.express as px

# fig = px.line(x=["a","b","c"], y=[1,3,2], title="sample figure")
# print(fig)
# fig.show()

import streamlit as st
import plotly.figure_factory as ff
import numpy as np

st.header("Section 1: Topics Barchart")
st.plotly_chart(fig_barchart, use_container_width=True)

st.header("Section 2: Topics Clusters")
st.plotly_chart(fig_docs_clusters, use_container_width=True)

st.header("Section 3: Hierarchical Clusters")
st.plotly_chart(fig_hier, use_container_width=True)
# st.plotly_chart(fig_term_rank, use_container_width=True)
# st.plotly_chart(fig_heatmap, use_container_width=True)
# st.plotly_chart(fig_topics, use_container_width=True)
# st.plotly_chart(fig_freq, use_container_width=True)
