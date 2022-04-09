import os 
import json
import argparse
import os.path as osp
import streamlit as st
from tools.visualization.streamlit.constants import Constants

parser = argparse.ArgumentParser(description='Streamlit visualization')
parser.add_argument('--result_folder', type=str,
                    help="Path to folder contains json result files")
parser.add_argument('-i', '--root_dir', type=str,
                    help="Path to root dir")
parser.add_argument('-s', '--split', type=str,
                    help="Specified split ['test', 'pseudo-test']")
args = parser.parse_args()

CONSTANTS = Constants(args.root_dir)
queries_dict = json.load(open(CONSTANTS.QUERY_JSON[args.split], 'r'))
version_map = {
        json_name: osp.join(args.result_folder, json_name) for json_name in os.listdir(args.result_folder)
    }


srl_dict = None
if osp.isfile(CONSTANTS.SRL_JSON[args.split]):
    srl_dict = json.load(open(CONSTANTS.SRL_JSON[args.split], 'r'))
    

def main(args):
    st.set_page_config(layout="wide")
    st.title('Traffic Video Event Retrieval via Text Query')

    list_versions = sorted(list(version_map.keys()))

    # Choose result version
    st.sidebar.subheader("Choose version")
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

    # SRL dictionary
    if srl_dict:
        srl_track = srl_dict[choose_qid]
        subjects = [
            srl_track[srl_id]['main_subject'] 
            for srl_id in srl_track.keys()
        ]

        
        colors = [
            srl_track[srl_id]['srl'][0]['subject_color'][0]['color'] 
            if len(srl_track[srl_id]['srl'][0]['subject_color']) > 0
            else 'None'
            for srl_id in srl_track.keys()
        ]
        actions = [
            srl_track[srl_id]['srl'][0]['action']
            for srl_id in srl_track.keys()
        ]
        objects = [
            srl_track[srl_id]['srl'][0]['arg_1']
            for srl_id in srl_track.keys()
        ]

        col1, col2, col3 = st.columns([1, 1, 2])

        # Write out query captions
        col1.markdown("### Query captions")
        col1.markdown("#### Main views")
        for cap in list_caps:
            col1.write(cap)
        col1.markdown("#### Other views")
        for cap in list_other_caps:
            col1.write(cap)

        col2.markdown("#### SRL Extraction")
        final_title = []
        for (sbj, clr, act, obj) in zip(subjects, colors, actions, objects):
            sub_title = f'<span style="font-family:sans-serif; color:Green; font-size: 16px;">{sbj}</span>'
            clr_title = f'<span style="font-family:sans-serif; color:Brown; font-size: 16px;">{clr}</span>'
            act_title = f'<span style="font-family:sans-serif; color:Blue; font-size: 16px;">{act}</span>'
            obj_title = f'<span style="font-family:sans-serif; color:Red; font-size: 16px;">{obj}</span>'
            final_title.append(', '.join([sub_title, clr_title, act_title, obj_title]))
        col2.markdown(
            '<br>'.join(final_title), unsafe_allow_html=True
        )
        
        if args.split == 'pseudo-test':
            # Visualize video
            col3.markdown("### Target video")
            video_name = f'{choose_qid}.mp4'
            video_path = osp.join(
                CONSTANTS.VIDEO_DIR[args.split], video_name)
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            col3.video(video_bytes)
            col3.text(video_name)

    else:

        # Write out query captions
        st.markdown("### Query captions")
        st.markdown("#### Main views")
        for cap in list_caps:
            st.write(cap)
        st.markdown("#### Other views")
        for cap in list_other_caps:
            st.write(cap)

    COLUMNS = 3
    ROWS = max(min(len(list_vid_ids), top_to_show) // COLUMNS, 1)

    # View text
    captions_list = []
    if args.split == 'pseudo-test':
        # load query texts
        track_json = json.load(open(CONSTANTS.TRACKS_JSON[args.split], 'r'))
        captions_list = [track_json[i]["nl"]+['-------------'] + track_json[i]['nl_other_views'] for i in list_vid_ids]

    # Show retrieved video results
    st.markdown("### Video results")
    if st.button('Search'):
        for r in range(ROWS):
            cols = st.columns(COLUMNS)
            for c in range(COLUMNS):
                vid_order = 3*r + c
                video_name = f'{list_vid_ids[vid_order]}.mp4'
                video_path = osp.join(CONSTANTS.VIDEO_DIR[args.split], video_name)
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                cols[c].video(video_bytes)
                cols[c].text(f'{vid_order+1}. {video_name}')
                if len(captions_list) > 0:
                    if choose_qid == list_vid_ids[vid_order]:
                        captions = '<br>'.join(captions_list[vid_order])
                        new_title = f'<p style="font-family:sans-serif; color:Green; font-size: 16px;">{captions}</p>'
                        cols[c].markdown(new_title, unsafe_allow_html=True)
                    else:
                        captions = '\n'.join(captions_list[vid_order])
                        cols[c].text(captions)

if __name__ == '__main__':
    main(args)