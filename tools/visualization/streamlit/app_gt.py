import os 
import json
import argparse
import os.path as osp
import streamlit as st
from .constants import Constants

parser = argparse.ArgumentParser(description='Streamlit visualization')
parser.add_argument('-i', '--root_dir', type=str,
                    help="Path to root dir")
args = parser.parse_args()
           
CONSTANTS = Constants(args.root_dir)

def main():
    st.set_page_config(layout="wide")
    st.title('AIC2022: Ground truth')

    # Choose file
    st.sidebar.subheader("Choose file")
    list_versions = list(CONSTANTS.TRACKS_JSON.keys())
    filename = st.sidebar.radio(label="", options=list_versions)
    all_track_dict = json.load(open(CONSTANTS.TRACKS_JSON[filename], 'r'))

    # Choose top k to retrieve
    top_to_show = st.sidebar.slider(
        'Show top-? results', 3, 30, 15)

    # Choose track id
    list_tids = list(all_track_dict.keys())
    display = [] 
    for qid in list_tids:
        display.append(f'{qid}')

    choose_qid = st.selectbox(f"Choose track", options=display)
    track_dict = all_track_dict[choose_qid]
    list_caps =  track_dict['nl'] 
    list_other_caps = track_dict['nl_other_views']

    # Write out query captions
    st.markdown("### Query captions")
    st.markdown("#### Main views")
    for cap in list_caps:
        st.write(cap)
    st.markdown("#### Other views")
    for cap in list_other_caps:
        st.write(cap)

    # Show retrieved video results
    st.markdown("### Video results")
    if st.button('Search'):
        video_path = osp.join(CONSTANTS.TEST_VIDEO_DIR, f"{choose_qid}.mp4")
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        st.text(f'{choose_qid}')

if __name__ == '__main__':
    main()