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

opsis_labels = ["Minestrone with beef",
"Beef chili",
"Oatmeal with flax seeds and apple",
"Blueberry shortcake",
"Roast beef sandwich",
"Turkey Sandwich",
"Classic Caesar salad",
"Mixed berry shortcake",
"Veggie sandwich",
"Classic Caesar salad with edamame",
"Oatmeal with diced apple",
"Classic minestrone with leafy greens",
"Chocolate chip cookies with pecans",
"Lasagna with meat sauce",
"Strawberry shortcake",
"Vegetarian pizza",
"Chocolate chip cookies",
"Oatmeal with diced apple and pistachios",
"Classic Caesar salad with chicken",
"Vegetarian lasagna with spinach",
"Classic minestrone",
"Chicken and bean chili",
"Vegetarian lasagna",
"Beef Steak",
"Tilapia",
"Banana",
"Green beans",
"pecans",
"Baked Beans",
"Cherries",
"Macaroni and cheese",
"brown rice",
"Brownie ",
"White Bread",
"Carrots",
"green peas",
"cornbread",
"Potato",
"Canned pear halves",
"Corn",
"String cheese",
"Fried Chicken Breast",
"Fried eggs",
"Peanut Butter",
"Pork roast",
"chicken patty",
"Pear ",
"tapioca pudding, vanilla and other flavors, homemade",
"rice pudding, plain, made with egg",
"plain yogurt, lowfat",
"oatmeal, regular cooking",
"macaroni noodles",
"angel hair pasta",
"chocolate pudding, from mix, instant",
"garbanzo beans",
"cottage cheese, regular (4% fat)",
"Blueberries",
"couscous",
"rice noodles",
"Strawberries",
"chocolate ice cream",
"vanilla ice cream",
"chili, canned, with beans and beef, less sodium",
"pumpkin cookies",
"dark chocolate covered almonds",
"caramel popcorn",
"cheesecake",
"eclair",
"macaroon",
"ice cream sandwich",
"chocolate cheesecake",
"chocolate coated cookies",
"Rice Krispies treats",
"animal crackers",
"oatmeal cookies",
"chocolate-coated graham crackers",
"peanut butter sandwich cookies",
"molasses cookies",
"sugar wafer cookies",
"mochi",
"key lime pie",
"lava cake",
"wafer cookies",
"gingerbread cookies",
"chocolate covered peanuts",
"windmill cookies",
"angel food cake",
"yogurt coated pretzels",
"butter cookie with filling",
"apple turnover",
"white chocolate macadamia nut cookies",
"Apple pie",
"thumbprint cookie",
"butter waffle cookies",
"ice cream cone, cake or wafer type",
"maraschino cherries",
"chocolate cream pie",
"pizelle cookies",
"shortbread cookies",
"biscotti",
"pecan pie",
"madeleine cookies",
"rice cake, flavored",
"chocolate covered pretzels",
"ice cream bar",
"peanut butter cookies",
"mint cookies with chocolate coating",
"chocolate covered peanut clusters",
"chocolate-covered toffee",
"lemon bars",
"ice cream cone, sugar type",
"yogurt coated shortbread ",
"half moon cookies",
"chocolate mousse",
"Cookie Sandwich",
"oatmeal raisin cookies",
"MNM cookies",
"cream puff",
"Danish pastry with cheese filling",
"Maria cookies",
"vanilla sandwich cookies",
"Lacys cookies",
"meringue, angel cookies",
"brownie-cookie bar",
"snickerdoodle cookies",
"coconut cream pie",
"Danish pastry with fruit filling",
"dark chocolate covered raisins",
"lady fingers",
"chocolate covered strawberries",
"iced sugar cookies",
"fortune cookies",
"Cashews",
"Tuna salad",
"Mashed Potatoes",
"Almonds",
"Sweet potatoes",
"Pancake",
"Cabbage",
"Canadian bacon",
"Chicken drumstick",
"Shrimp",
"Donut",
"Summer squash",
"Potato chips",
"Hot dog",
"Chicken breast",
"Scrambled eggs",
"Chicken thigh",
"Pickles",
"French Fries",
"Cured Ham",
"Beef steak",
"Peach",
"Pork chops",
"Waffles",
"Walnuts",
"Cucumber",
"Dried mango",
"Celery",
"Onion rings ",
"Pinto Beans",
"Asparagus",
"Pistachios",
"Brussels sprouts",
"Pasta salad",
"Peanuts",
"Bell pepper",
"Potato salad",
"Chicken wing ",
"Bagel",
"Broccoli",
"English Muffin",
"Pumpkin pie",
"Cauliflower",
"Apple",
"Raspberries",
"Black beans",
"Chicken salad",
"Cantaloupe",
"Watermelon ",
"Fruit salad",
"Meatloaf",
"Chocolate sandwich cookies",
"Mango",
"Hashed brown potatoes",
"Popcorn",
"Poached eggs",
"Chicken tenders ",
"Pretzel twists or rings",
"Chicken nuggets",
"Cheese cracker",
"Applesauce",
"Refried bean ",
"Egg salad ",
"Mozzarella sticks",
"Tortilla chips",
"Coffee cake ",
"Dried apple",
"Tomato",
"White rice",
"Devil's food cake",
"Brownie",
"Potato chips, ruffled or rippled",
"Orange ",
"chocolate brownie, frosted",
"Yogurt",
"chocolate brownie with nuts",
"Biscuit",
"Raisins ",
"Grapes",
"Bacon",
"Wheat cracker",
"Snack pie",
"Spanish rice",
"Zucchini",
"Clementine",
"Croissant ",
"mushrooms",
"ground beef",
"fig bars",
"melon, honeydew",
"grapefruit",
"avocado",
"pork rinds",
"lima beans",
"almond butter",
"boiled potato",
"ice cream cone",
"macadamia nuts",
"ham lunchmeat",
"paella",
"fish sticks",
"tater tots",
"radish",
"kale",
"apricot",
"mixed nuts",
"ground turkey",
"marshmallow",
"egg white",
"saurkraut",
"scalloped potatoes",
"lamb chops",
"salmon",
"chicken lunchmeat",
"lentils",
"roll",
"ground pork",
"okra",
"strudel",
"bacon bits",
"truffles",
"corn dog",
"rice and beans",
"dried apricot",
"dried prune",
"parsnip",
"jalapeno peppers",
"pork sausage",
"hard boiled eggs",
"salmon patty",
"egg subsitute",
"plum",
"salami",
"jicama",
"beef lunchmeat",
"pepperoni",
"croutons",
"pumpkin seeds",
"turkey light meat",
"beef jerky",
"tortellini",
"kidney beans",
"pineapple",
"tofu, raw",
"guacamole",
"deviled eggs",
"sushi",
"prosciutto",
"eggplant",
"rice pilaf",
"quinoa",
"dried figs",
"onion",
"soft pretzel",
"meatballs",
"snow peas",
"hummus",
"edamame",
"muffin, banana",
"tamale",
"graham crackers",
"spinach dip",
"french toast sticks",
"tuna, canned",
"turkey lunchmeat",
"dates",
"beef brisket",
"artichoke",
"egg yolk",
"falafel",
"bratwurst",
"kimchi",
"muffin, chocolate chip",
"Brazil nuts",
"beef stew",
"tapenade",
"water chestnuts",
"pork ribs",
"Chard",
"colby monteray jack",
"Pita bread",
"blackberries",
"sunflower seed",
"scone",
"bulgur",
"Apple Crisp ",
"canned potato",
"coleslaw",
"scallions",
"papaya",
"dried cranberries",
"fruit cocktail",
"Chicken fried steak",
"ravioli",
"Bamboo Shoots",
"hearts of palm",
"Kiwi",
"queso fresco",
"liverworst spread",
"stuffing",
"salisbury steak",
"wild rice",
"pretzel nuggets",
"bread sticks",
"Mandrian segments",
"sprouts",
"cod ",
"black olives",
"corn nuts",
"beets",
"egg rolls",
"Chorizo, beef",
"roasted potatoes",
"turnip",
"sun-dried tomato",
"summer sausage",
"gnocchi",
"Banana chips",
"taco shell",
"fennel bulb",
"Green olives",
"beef liver",
"Diced ham",
"barley",
"cupcake",
"rice cake",
"Italian sausage",
"whole wheat rotini noodles",
"whole wheat orzo noodles",
"meat ravioli, without sauce",
"egg noodles",
"rotini",
"whole wheat spaghetti noodles",
"rigatoni",
"spaghetti noodles",
"orzo noodles",
"penne",
"cheese ravioli, without sauce",
"lasagna noodles",
"spinach ravioli, without sauce",
"chicken tortellini, without sauce",
"cheese tortellini, without sauce",
"gluten free pasta, lentil",
"linguini",
"farfalle",
"fettucini noodles, without egg",
"ziti"]

#@st.cache(persist=True)
# @st.experimental_memo(persist="disk")
# def run_topic_model():
#     docs = opsis_labels
#     topic_model = BERTopic(seed_topic_list=opsis_labels)
#     topics, probs = topic_model.fit_transform(docs)
#     return data

# run_topic_model()

docs = opsis_labels
topic_model = BERTopic(seed_topic_list=opsis_labels)
topics, probs = topic_model.fit_transform(docs)

fig_hier = topic_model.visualize_hierarchy(width=2000, height=2000)

fig_term_rank = topic_model.visualize_term_rank()

fig_heatmap = topic_model.visualize_heatmap(width=1500, height=1500)

fig_topics = topic_model.visualize_topics(width=1500, height=800,  top_n_topics=100)

fig_barchart = topic_model.visualize_barchart(top_n_topics=100, width=200, height=200)

fig_docs = topic_model.visualize_documents(docs)

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
st.plotly_chart(fig_docs, use_container_width=True)

st.header("Section 3: Hierarchical Clusters")
st.plotly_chart(fig_hier, use_container_width=True)
# st.plotly_chart(fig_term_rank, use_container_width=True)
# st.plotly_chart(fig_heatmap, use_container_width=True)
# st.plotly_chart(fig_topics, use_container_width=True)
# st.plotly_chart(fig_freq, use_container_width=True)
