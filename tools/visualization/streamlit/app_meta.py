import os 
import os.path as osp
import json
import pandas as pd
import argparse
import os.path as osp
from typing import Dict, List 
import numpy as np
import streamlit as st
from tools.visualization.streamlit.constants import Constants

parser = argparse.ArgumentParser(description='Streamlit visualization')
parser.add_argument('-i', '--root_dir', type=str,
                    help="Path to root dir")

parser.add_argument('-s', '--split', type=str,
                    help="Specified split ['test', 'pseudo-test']")

args = parser.parse_args()
           
CONSTANTS = Constants(args.root_dir)

action_dict = None
relation_dict = None
color_dict = None
vehicle_dict = None


if osp.isfile(CONSTANTS.STOP_TURN_JSON[args.split]):
    action_dict = json.load(open(CONSTANTS.STOP_TURN_JSON[args.split], 'r'))

if osp.isfile(CONSTANTS.RELATION_JSON[args.split]):
    relation_dict = json.load(open(CONSTANTS.RELATION_JSON[args.split], 'r'))

if osp.isfile(CONSTANTS.COLOR_JSON[args.split]):
    color_dict = json.load(open(CONSTANTS.COLOR_JSON[args.split], 'r'))
    available_colors = [] 
    for colors in color_dict.values():
        available_colors.extend(colors)
    available_colors = list(set(available_colors))
    
if osp.isfile(CONSTANTS.VEHICLE_JSON[args.split]):
    vehicle_dict = json.load(open(CONSTANTS.VEHICLE_JSON[args.split], 'r'))
    available_vehicles = [] 
    for vehicles in vehicle_dict.values():
        available_vehicles.extend(vehicles)
    available_vehicles = list(set(available_vehicles))

def filter_vehcol(current_ids, thedict: Dict, filters: List):
    if len(filters) == 0:
        return current_ids
    filtered_ids = []
    for current_id in current_ids:
        current_track = thedict[current_id]
        if len(np.intersect1d(current_track, filters)) == len(filters):
            filtered_ids.append(current_id)
    intersected_ids = np.intersect1d(filtered_ids, current_ids)
    return intersected_ids

def filter_action(current_ids, action_dict, filters: List):
    intersected_ids = current_ids
    for filter in filters:
        intersected_ids = np.intersect1d(intersected_ids, action_dict[filter])
    return intersected_ids

def filter_relation(current_ids, relation_dict, filter):
    if filter == 'None':
        return current_ids
    filtered_ids = []
    for current_id in current_ids:
        current_track = relation_dict[current_id]
        if len(current_track[filter]) > 0:
            filtered_ids.append(current_id)
    intersected_ids = np.intersect1d(filtered_ids, current_ids)
    return intersected_ids

def main(args):
    st.set_page_config(layout="wide")
    st.title('AIC2022: Metadata')

    if action_dict:
        action_options = st.multiselect(
            'Action filter',
            ['stop', 'turn_left', 'turn_right', 'straight_views'],
        )
    if relation_dict:
        relation_option = st.selectbox(
            'Relation filter',
            ('None', 'follow', 'followed_by'),
        )
    if color_dict:
        color_options = st.multiselect(
        'Color filter',
        options=available_colors)
    if vehicle_dict:
        vehicle_options = st.multiselect(
        'Vehicle filter',
        options=available_vehicles)

    # Choose top k to retrieve
    top_to_show = st.sidebar.slider(
        'Show top-? results', 3, 30, 15)

    # Filter out
    list_vid_ids = [i[:-4] for i in os.listdir(CONSTANTS.VIDEO_DIR[args.split])]

    if action_dict:
        list_vid_ids = filter_action(
            list_vid_ids, action_dict, action_options)
    if relation_dict:
        list_vid_ids = filter_relation(
            list_vid_ids, relation_dict, relation_option)
    if vehicle_dict:
        list_vid_ids = filter_vehcol(
            list_vid_ids, vehicle_dict, vehicle_options)
    if color_dict:
        list_vid_ids = filter_vehcol(
            list_vid_ids, color_dict, color_options)

    list_vid_ids = list_vid_ids[:top_to_show]

    if len(list_vid_ids) < 0:
        st.warning("No video is found")

    COLUMNS = 3
    ROWS = max(min(len(list_vid_ids), top_to_show) // COLUMNS, 1)

    captions_list = []
    if args.split == 'pseudo-test':
        # load query texts
        track_json = json.load(open(CONSTANTS.TRACKS_JSON[args.split], 'r'))
        captions_list = [track_json[i]["nl"] for i in list_vid_ids]

    # Show retrieved video results
    st.markdown("### Videos")
    if st.button('Search'):
        for r in range(ROWS):
            cols = st.columns(COLUMNS)
            for c in range(COLUMNS):
                vid_order = 3*r + c
                if vid_order >= len(list_vid_ids):
                    break
                video_name = f'{list_vid_ids[vid_order]}.mp4'
                captions = '\n'.join(captions_list[vid_order])
                video_path = osp.join(CONSTANTS.VIDEO_DIR[args.split], video_name)
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                cols[c].video(video_bytes)
                cols[c].text(f'{vid_order+1}. {video_name}')
                cols[c].text(captions)

                

if __name__ == '__main__':
    main(args)