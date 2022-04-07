import os 
import json
import argparse
import os.path as osp
import streamlit as st

parser = argparse.ArgumentParser(description='Streamlit visualization')
parser.add_argument('--track_json', type=str,
                    help="Path to json query file")
parser.add_argument('--video_dir', type=str,
                    help="Path to folder contains all gallery track videos  ")

def json_load(json_path: str):
    data = None
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def setup(args):
    json_files = os.listdir(args.result_folder)
    version_map = {
        json_name: osp.join(args.result_folder, json_name) for json_name in json_files
    }
    return version_map

args = parser.parse_args()
queries_dict = json_load(args.query_json)
version_map = setup(args)

def main(args):
    st.set_page_config(layout="wide")
    st.title('Traffic Video Event Retrieval via Text Query')

    list_versions = sorted(list(version_map.keys()))

    # Choose result version
    st.sidebar.subheader("Choose version")
    version = st.sidebar.radio(label="", options=list_versions)
    result_dict = json_load(version_map[version])

    # Choose top k to retrieve
    top_to_show = st.sidebar.slider(
        'Show top-? results', 3, 30, 15)

    # Choose query id
    list_qids = list(result_dict.keys())
    display = [] 
    for qid in list_qids:
        display.append(f'{qid}')

    choose_qid = st.selectbox(f"Choose query", options=display)
    query_dict = queries_dict[choose_qid]
    list_caps =  query_dict['nl'] #+ query_dict['nl_other_views']
    list_vid_ids = result_dict[choose_qid] #['pred_ids']

    # Write out query captions
    st.markdown("### Query captions")
    for cap in list_caps:
        st.write(cap)

    COLUMNS = 3
    ROWS = top_to_show // COLUMNS

    # Show retrieved video results
    st.markdown("### Video results")
    if st.button('Search'):
        for r in range(ROWS):
            cols = st.columns(COLUMNS)
            for c in range(COLUMNS):
                vid_order = 3*r + c
                video_name = f'{list_vid_ids[vid_order]}.mp4'
                video_path = osp.join(args.video_dir, video_name)
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                cols[c].video(video_bytes)
                cols[c].text(f'{vid_order+1}. {video_name}')

if __name__ == '__main__':
    main(args)