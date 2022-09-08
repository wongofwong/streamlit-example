import streamlit as st

st.markdown("# Data Manager - BERTopic Modeling ")
st.sidebar.markdown("# Data Manager - BERTopic Modeling ")

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

add_selectbox = st.sidebar.selectbox(
    'Visualization',
    ('Topics Word Scores Barchart', 'Intertopic Distance Map', 'Hierarchical Clustering', 'Similarity Heat Map', 'Topic Frequency Distribution')
)

# import streamlit as st
import streamlit.components.v1 as components

# bootstrap 4 collapse example
components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
      <div class="card">
        <div class="card-header" id="headingOne">
          <h5 class="mb-0">
            <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
            Topic Model Inputs
            
            seed_topic_list=['Minestrone with beef', 'Beef chili', 'Oatmeal with flax seeds and apple', 'Blueberry shortcake', 'Roast beef sandwich', 'Turkey Sandwich', 'Classic Caesar salad', 'Mixed berry shortcake', 'Veggie sandwich', 'Classic Caesar salad with edamame', 'Oatmeal with diced apple', 'Classic minestrone with leafy greens', 'Chocolate chip cookies with pecans', 'Lasagna with meat sauce', 'Strawberry shortcake', 'Vegetarian pizza', 'Chocolate chip cookies', 'Oatmeal with diced apple and pistachios', 'Classic Caesar salad with chicken', 'Vegetarian lasagna with spinach', 'Classic minestrone', 'Chicken and bean chili', 'Vegetarian lasagna', 'Beef Steak', 'Tilapia', 'Banana', 'Green beans', 'pecans', 'Baked Beans', 'Cherries', 'Macaroni and cheese', 'brown rice', 'Brownie ', 'White Bread', 'Carrots', 'green peas', 'cornbread', 'Potato', 'Canned pear halves', 'Corn', 'String cheese', 'Fried Chicken Breast', 'Fried eggs', 'Peanut Butter', 'Pork roast', 'chicken patty', 'Pear ', 'tapioca pudding, vanilla and other flavors, homemade', 'rice pudding, plain, made with egg', 'plain yogurt, lowfat', 'oatmeal, regular cooking', 'macaroni noodles', 'angel hair pasta', 'chocolate pudding, from mix, instant', 'garbanzo beans', 'cottage cheese']
            
            </button>
          </h5>
        </div>
        <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
          <div class="card-body">
            Topic Model Inputs
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
            Topic Model Output Visualizations
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          <div class="card-body">
            Topic Model Output Visualizations
          </div>
        </div>
      </div>
    </div>
    """,
    height=600,
)

from bertopic import BERTopic
import pandas as pd

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

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(
         hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)

import plotly.express as px

fig = px.line(x=["a","b","c"], y=[1,3,2], title="sample figure")
print(fig)
fig.show()

# import plotly.graph_objects as go

# fig = go.Figure(
#     data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
#     layout=go.Layout(
#         title=go.layout.Title(text="A Figure Specified By A Graph Object")
#     )
# )

# fig.show()
# st.plotly_chart(fig, use_container_width=True)
