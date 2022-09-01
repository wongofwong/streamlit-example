import os
import csv
import json
from time import time
from pathlib import Path
from itertools import repeat
import concurrent.futures as cf

import clip
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
# import streamlit.components.v1 as components
from streamlit_ace import st_ace
# import streamlit_modal as modal
from st_clickable_images import clickable_images
from torch import tensor
from torch.cuda import is_available as cuda_is_available

from src import config
from src.odm import connection, get_datasource_options, run_query
from src.annotations import annotate_images, label_to_color, color_key
from src.session_state import init_session_state
from src.embedding import load_clip_model


st.set_page_config(layout="wide")
# Number of images per page (pagination)
N = 10
env = config.Env()
DATA_DIR = Path(env.bucket)
CLIENT = connection(env.mongo_uri)
CLIP_MODEL_VERSION = env.clip_model_version
DEVICE = "cuda" if cuda_is_available() else "cpu"
MODEL, _ = load_clip_model(CLIP_MODEL_VERSION)
DATASET_TYPE_OPTIONS = ["classification", "detection", "segmentation"]
# instantiate Streamlit session state variables
init_session_state()
EXPORT_LIST = st.session_state["export_list"]



def run():
    """
    # Plainsight Data-manager
    """
    with st.sidebar:
        """
        A simple UI for visualizing and filtering dataset and annotations.
        """
        # instructions = st.expander('Instructions')
        # instructions.write(
        #     """
        #     1. Select a datasource from the dropdown menu in the sidebar.
        #     2. Use the Ace editor to enter a MongoDB query; click the Apply button to execute the query
        #     """
        # )
        with st.expander("Select datasource, project, and annotation type", expanded = False):
            dataset_type_filter = st.selectbox(
                label = "Annotation Type",
                options = DATASET_TYPE_OPTIONS
            )
            st.session_state["dataset_type"] = dataset_type_filter

            # check db for options
            datasource_options, project_options, distinct_label_options = get_datasource_options(CLIENT, dataset_type_filter)
            datasource_idx = st.selectbox(
                label = 'Available Datasources',
                options = range(len(datasource_options)),
                format_func=lambda x: datasource_options[x],
                index = st.session_state["datasource_index"]
            )
            st.session_state["datasource_index"] = datasource_idx

            project = st.selectbox(
                label = "Available Projects",
                options = project_options[datasource_idx]
            )
            st.session_state["project_name"] = project
            datasource = datasource_options[datasource_idx]
            st.session_state["datasource_name"] = datasource
            distinct_labels = distinct_label_options[datasource_idx]
            selected_labels = distinct_labels
            lbl_to_clr = None

            # if dataset_type_filter in ["detection", "segmentation"]:
            #     selected_labels = st.multiselect(
            #         label = "Labels to visualize",
            #         options = distinct_labels,
            #         default = distinct_labels
            #     )
            #     lbl_to_clr = label_to_color(selected_labels)
            #     """
            #     Color Key:
            #     """
            #     color_key(lbl_to_clr)     
            
            st.session_state["selected_labels"] = selected_labels
            st.session_state["label_to_color"] = lbl_to_clr

    # """
    # ## Mongo Query
    # """
    with st.expander("Query templates", expanded = False):
        with open("./src/templates/opsis_query_templates.json") as f:
            query_templates = json.load(f)
            st.json(query_templates, expanded = False)
    with st.expander("Mongo Query", expanded = False):
        search_query_string = st_ace(
            value = st.session_state["mongo_query"],
            # placeholder = "MongoDB query",
            theme="dracula", 
            language='json'
        )
        st.session_state["mongo_query"] = search_query_string
    if search_query_string == "":
        search_query = None
        # st.session_state["mongo_query"] = search_query_string
    else:
        search_query = json.loads(search_query_string)
        # st.session_state["mongo_query"] = search_query_string
    # st.session_state["mongo_query"] = search_query_string

    """
    ## Results
    """
    prev, results_title, next = st.columns([1, 10, 1])
    img_docs = run_query(CLIENT, datasource, search_query)
    last_page = len(img_docs) // N
    image_embeddings = []
    label_embeddings = []
    labels = []
    img_file_names = []
    for doc in img_docs:
        image_embeddings.append(doc["imgProperties"]["embedding"])
        label_embeddings.append(doc["labels"][0]["classification"]["embedding"])
        labels.append(doc["labels"][0]["classification"]["label"])
        img_file_names.append(doc["imgProperties"]["fileName"])

    st.session_state["embeddings"] = image_embeddings
    st.session_state["labels"] = labels

    # save to disk for visualization analysis
    # print("Writing embeddings...")
    # with open("label.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(labels)
    # with open("image_embeddings.csv", "w") as f:
    #     w = csv.writer(f, delimiter=";", quoting = csv.QUOTE_NONNUMERIC)
    #     w.writerows(image_embeddings)
    # with open("label_embeddings.csv", "w") as f:
    #     w = csv.writer(f, delimiter=";", quoting = csv.QUOTE_NONNUMERIC)
    #     w.writerows(label_embeddings)
    # with open("img_file_names.csv", "w") as f:
    #     w = csv.writer(f, delimiter=";")
    #     w.writerows(img_file_names)
    # print("Done writing embeddings!")


    # CLIP text search
    with st.sidebar:
        with st.expander("CLIP text search"):
            image_features = np.array(image_embeddings)
            # print(f"Image feature shape: {image_features.shape}")
            clip_query = st.text_input(label = "CLIP search prompt", value = "")

            if clip_query != "":
                text = clip.tokenize([clip_query]).to(DEVICE)
                text_features = MODEL.encode_text(text).cpu().detach().numpy()
                similarities = list((text_features @ image_features.T).squeeze(0))
                sort_idx = np.array(sorted(zip(similarities, range(image_features.shape[0])), key=lambda x: x[0], reverse=True))
                clip_ordering = sort_idx[:,1]
            else:
                clip_ordering = None


    # CLIP label-image agreement
    with st.sidebar:
        with st.expander("CLIP label-image disagreement"):
            if st.checkbox("Sort"):
                image_features = tensor(image_embeddings)
                text_features = tensor(label_embeddings)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarities = (100.0 * image_features @ text_features.T).numpy()
                similarities = np.diag(similarities)
                sort_idx = np.array(sorted(zip(similarities, range(image_features.shape[0])), key=lambda x: x[0], reverse=False))
                clip_ordering = sort_idx[:,1]


    # image pagination
    if next.button("Next"):
        if st.session_state.page_number + 1 > last_page:
            st.session_state.page_number = 0
        else:
            st.session_state.page_number += 1

    if prev.button("Previous"):
        if st.session_state.page_number - 1 < 0:
            st.session_state.page_number = last_page
        else:
            st.session_state.page_number -= 1

    results_title_text = f"""
        <p style="text-align:center"> Page Number: {st.session_state.page_number + 1} of {last_page + 1}</p>
    """
    results_title.markdown(results_title_text, True)

    # Get start and end indices of next page of image docs
    start_idx = st.session_state.page_number * N 
    end_idx = (1 + st.session_state.page_number) * N

    if clip_ordering is not None:
        img_docs_subset = [img_docs[int(i)] for i in clip_ordering][start_idx:end_idx]
    else:
        img_docs_subset = img_docs[start_idx:end_idx]


    # image annotation multithreading 
    images_encoded = []
    titles = []
    with cf.ProcessPoolExecutor(max_workers = min(os.cpu_count() - 1, N)) as executor:
             for res in executor.map(
                                        annotate_images, 
                                        img_docs_subset, 
                                        repeat(dataset_type_filter),
                                        repeat(selected_labels), 
                                        repeat(lbl_to_clr)):
                images_encoded.append(res[0])
                titles.append(res[1])

    # image gallery
    clicked = clickable_images(
        paths = images_encoded, 
        titles = titles,
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "200px"}
    )


    if clicked > -1:
        st.session_state["clicked_image_doc"] = img_docs_subset[clicked]
        st.session_state["clicked_image_title"] = titles[clicked]

    # tagging and export
    with st.sidebar:
        with st.expander("Tag and Export"):
            tag = st.text_input("tag")
            export_name = st.text_input("Name for the JSON export file")
            # export_all = st.checkbox("Export all")
            if bool(tag) and clicked > -1:
                exp = {
                    "fileName" : img_docs_subset[clicked]["imgProperties"]["fileName"],
                    "tag" : tag
                }
                EXPORT_LIST.append(exp)
                st.session_state["export_list"] = EXPORT_LIST
            if bool(export_name) and bool(len(EXPORT_LIST)):
                # if export_all:
                    # EXPORT_LIST.clear()
                    # for doc in img_docs:
                    #     EXPORT_LIST.append({
                    #         "fileName" : doc["imgProperties"]["fileName"],
                    #         "tag" : tag 
                    #     })
                data = json.dumps(EXPORT_LIST)
                st.download_button(
                    label = "Export tags", 
                    data = data, 
                    file_name = f"{export_name}.json", 
                    mime = "application/json",
                    on_click = st.session_state["export_list"].clear)




if __name__ == "__main__":
    run()