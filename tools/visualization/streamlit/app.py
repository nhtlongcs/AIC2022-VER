import os 
import json
import argparse
import os.path as osp 
import streamlit as st
from config import StreamlitConfig


parser = argparse.ArgumentParser(description='Streamlit visualization')
parser.add_argument('--result_folder', type=str,
                    help="Path to folder contains json result files")
parser.add_argument('--query_json', type=str,
                    help="Path to json query file")
parser.add_argument('--video_dir', type=str,
                    help="Path to folder contains all gallery track videos  ")

def json_load(json_path: str):
    data = None
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def main(config):
    st.set_page_config(layout="wide")
    st.title(config.TITLE)

    list_versions = sorted(list(config.version_map.keys()))

    # Choose result version
    st.sidebar.subheader("Choose version")
    version = st.sidebar.radio(label="", options=list_versions)
    result_dict = json_load(config.version_map[version])

    # Choose top k to retrieve
    top_to_show = st.sidebar.slider(
        'Show top-? results', 3, 30, config.TOP_TO_SHOW)

    # Choose query id
    list_qids = list(result_dict.keys())
    display = [] 
    for qid in list_qids:
        display.append(f'{qid}')

    choose_qid = st.selectbox(f"Choose query", options=display)
    list_caps = json_load(config.query_json)[choose_qid]
    list_vid_ids = result_dict[choose_qid]

    # Write out query captions
    st.markdown("### Query captions")
    for cap in list_caps:
        st.write(cap)

    ROWS = top_to_show // config.COLUMNS 

    # Show retrieved video results
    st.markdown("### Video results")
    if st.button('Search'):
        for r in range(ROWS):
            cols = st.beta_columns(config.COLUMNS)
            for c in range(config.COLUMNS):
                vid_order = 3*r + c
                video_name = f'{list_vid_ids[vid_order]}.mp4'
                video_path = osp.join(config.video_dir, video_name)
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                cols[c].video(video_bytes)
                cols[c].text(f'{vid_order+1}. {video_name}')

if __name__ == '__main__':
    args = parser.parse_args()
    config = StreamlitConfig(
        query_json = args.query_json, 
        result_dir = args.result_folder, 
        video_dir = args.video_dir
    )


    main(config)