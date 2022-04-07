import os 
import json
import argparse
import os.path as osp
from typing import List 
import numpy as np
import streamlit as st

parser = argparse.ArgumentParser(description='Streamlit visualization')
parser.add_argument('--video_dir', type=str,
                    help="Path to folder contains all gallery track videos  ")
parser.add_argument('--relation_json', type=str, default=None,
                    help="Path to json file contains relations  ")
parser.add_argument('--action_json', type=str, default=None,
                    help="Path to json file contains actions  ")
parser.add_argument('--color_json', type=str, default=None,
                    help="Path to json file contains color identification  ")
parser.add_argument('--vehicle_json', type=str, default=None,
                    help="Path to json file contains vehicle types  ")

args = parser.parse_args()

def json_load(json_path: str):
    data = None
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

action_dict = json_load(args.action_json)
relation_dict = json_load(args.relation_json)

if args.color_json:
    color_dict = json_load(args.color_json)
    availabel_colors = [] 
    for colors in color_dict.values():
        availabel_colors.extend(colors)
    availabel_colors = list(set(availabel_colors))
    
if args.vehicle_json:
    vehicle_dict = json_load(args.vehicle_json)
    availabel_vehicles = [] 
    for vehicles in vehicle_dict.values():
        availabel_vehicles.extend(vehicles)
    availabel_vehicles = list(set(availabel_vehicles))

def filter_vehicle(current_ids, vehicle_dict, filters: List):
    if len(filters) == 0:
        return current_ids
    filtered_ids = []
    for current_id in current_ids:
        current_track = vehicle_dict[current_id]
        if len(np.intersect1d(current_track, filters)) > 0:
            filtered_ids.append(current_id)
    intersected_ids = np.intersect1d(filtered_ids, current_ids)
    return intersected_ids

def filter_color(current_ids, color_dict, filters: List):
    if len(filters) == 0:
        return current_ids
    filtered_ids = []
    for current_id in current_ids:
        current_track = color_dict[current_id]
        if len(np.intersect1d(current_track, filters)) > 0:
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
    st.title('Traffic Video Event Retrieval Metadata visualization' )

    # Choose result version
    st.sidebar.subheader("Choose version")
    action_options = st.multiselect(
        'Action filter',
        ['stop', 'turn_left', 'turn_right', 'straight_views'],
    )

    relation_option = st.selectbox(
        'Relation filter',
        ('None', 'follow', 'followed_by'),
    )

    if args.color_json:
        color_options = st.multiselect(
        'Color filter',
        options=availabel_colors)

    if args.vehicle_json:
        vehicle_options = st.multiselect(
        'Vehicle filter',
        options=availabel_vehicles)

    # Choose top k to retrieve
    top_to_show = st.sidebar.slider(
        'Show top-? results', 3, 30, 15)

    # Filter out
    list_vid_ids = [i[:-4] for i in os.listdir(args.video_dir)]
    list_vid_ids = filter_action(
        list_vid_ids, action_dict, action_options)
    list_vid_ids = filter_relation(
        list_vid_ids, relation_dict, relation_option)

    if args.vehicle_json:
        list_vid_ids = filter_vehicle(
            list_vid_ids, vehicle_dict, vehicle_options)

    if args.color_json:
        list_vid_ids = filter_color(
            list_vid_ids, color_dict, color_options)


    if len(list_vid_ids) < 0:
        st.warning("No video is found")

    COLUMNS = 3
    ROWS = min(len(list_vid_ids), top_to_show) // COLUMNS


    # Show retrieved video results
    st.markdown("### Videos")
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