import os 
import json
import argparse
import os.path as osp
import streamlit as st
from .constants import Constants

parser = argparse.ArgumentParser(description='Streamlit visualization')
parser.add_argument('--result_folder', type=str,
                    help="Path to folder contains json result files")
parser.add_argument('-i', '--root_dir', type=str,
                    help="Path to root dir")
args = parser.parse_args()

CONSTANTS = Constants(args.root_dir)
queries_dict = json.load(open(args.query_json, 'r'))
version_map = {
        json_name: osp.join(args.result_folder, json_name) for json_name in os.listdir(args.result_folder)
    }

action_dict = None
relation_dict = None
color_dict = None
vehicle_dict = None

if osp.isfile(CONSTANTS.STOP_TURN_JSON):
    action_dict = json.load(open(CONSTANTS.STOP_TURN_JSON, 'r'))
if osp.isfile(CONSTANTS.RELATION_JSON):
    relation_dict = json.load(open(CONSTANTS.RELATION_JSON, 'r'))
if osp.isfile(CONSTANTS.COLOR_JSON):
    color_dict = json.load(open(CONSTANTS.COLOR_JSON, 'r'))
if osp.isfile(CONSTANTS.VEHICLE_JSON):
    vehicle_dict = json.load(open(CONSTANTS.VEHICLE_JSON, 'r'))

def main(args):
    st.set_page_config(layout="wide")
    st.title('AIC22: Submission')

    # Choose result version
    st.sidebar.subheader("Choose version")
    list_versions = sorted(list(version_map.keys()))
    version = st.sidebar.radio(label="", options=list_versions)
    result_dict = json.load(open(version_map[version], 'r'))

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
    list_vid_ids = result_dict[choose_qid]
    list_caps =  query_dict['nl'] 
    list_other_caps = query_dict['nl_other_views']

    # Write out query captions
    st.markdown("### Query captions")
    st.markdown("#### Main views")
    for cap in list_caps:
        st.write(cap)
    st.markdown("#### Other views")
    for cap in list_other_caps:
        st.write(cap)

    COLUMNS = 3
    ROWS = top_to_show // COLUMNS


    list_vid_ids = list_vid_ids[:top_to_show]

    
    # Show retrieved video results
    st.markdown("### Video results")
    if st.button('Search'):
        for r in range(ROWS):
            cols = st.columns(COLUMNS)
            for c in range(COLUMNS):
                vid_order = 3*r + c
                if vid_order >= len(list_vid_ids):
                    break
                video_name = f'{list_vid_ids[vid_order]}.mp4'
                video_path = osp.join(args.video_dir, video_name)
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                cols[c].video(video_bytes)
                cols[c].text(f'{vid_order+1}. {video_name}')

if __name__ == '__main__':
    main(args)