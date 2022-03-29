import os 
import os.path as osp 
import streamlit as st

from utils import json_load
from constant import (
    COLUMN, TOP_TO_SHOW,TITLE,
    video_dir,
    test_track_map, test_query_map, test_query,
    version_map
)

st.set_page_config(layout="wide")
st.title(TITLE)

list_versions = sorted(list(version_map.keys()))

st.sidebar.subheader("Choose version")
version = st.sidebar.radio(label="", options=list_versions)
# Choose result file:
# result_name = list_results[0]
# result_name = st.sidebar.selectbox(f"Result versions, default: {result_name}", options=list_results)
result_dict = json_load(version_map[version])

top_to_show = st.sidebar.slider('Show top-? results', 3, 30, TOP_TO_SHOW)

# Choose query
list_qids = list(result_dict.keys())
display = [] 
for qid in list_qids:
    display.append(f'{test_query_map[qid]}.{qid}')

query_o2i = st.selectbox(f"Choose query", options=display)
choose_qid = query_o2i.split('.')[-1]
list_caps = test_query[choose_qid]
list_vids = result_dict[choose_qid]

st.markdown("### Query captions")
for cap in list_caps:
    st.write(cap)
# st.write('\n'.join(list_caps))


ROWS = top_to_show // COLUMN 
i = 0

st.markdown("### Video results")
if st.button('Search'):
    for r in range(ROWS):
        cols = st.beta_columns(COLUMN)
        for c in range(COLUMN):
            vid_order = 3*r + c
            video_name = f'{test_track_map[list_vids[vid_order]]}.mp4'
            video_path = osp.join(video_dir, video_name)
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            cols[c].video(video_bytes)
            cols[c].text(f'{vid_order+1}. {video_name}')

        # video_name = f'{test_track_map[list_vids[3*i]]}.mp4'
        # video_path = osp.join(video_dir, video_name)
        # video_file = open(video_path, 'rb')
        # video_bytes = video_file.read()
        # cols[0].video(video_bytes)
        # cols[0].text(f'{3*i+1}. {video_name}')

        # video_name = f'{test_track_map[list_vids[3*i+1]]}.mp4'
        # video_path = osp.join(video_dir, video_name)
        # video_file = open(video_path, 'rb')
        # video_bytes = video_file.read()
        # cols[1].video(video_bytes)
        # cols[1].text(f'{3*i+2}. {video_name}')

        # video_name = f'{test_track_map[list_vids[3*i+2]]}.mp4'
        # video_path = osp.join(video_dir, video_name)
        # video_file = open(video_path, 'rb')
        # video_bytes = video_file.read()
        # cols[2].video(video_bytes)
        # cols[2].text(f'{3*i+3}. {video_name}')

        # i += 1
        pass