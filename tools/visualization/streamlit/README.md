# Visualization tools 

## Install 
install [Streamlit](https://docs.streamlit.io/en/stable/)
`pip install streamlit`

## Run

- To run streamlit with arguments, use `--` before flags


- To visualize data ground truth
```
streamlit run app_sub.py -- \
  --result_folder ./results \
  --query_json ./test-queries.json \
  --video_dir ./track_videos
```

- To visualize prediction before submission 
```
streamlit run app_sub.py -- \
  --result_folder ./results \
  --query_json ./test-queries.json \
  --video_dir ./track_videos
```

- To visualize metadata (vehicle action and relation)
```
streamlit run app_meta.py -- \
    --video_dir data/meta/track_visualization/relation/test-convert \
    --relation_json data/meta/relation/test_relation.json \
    --action_json data/meta/action/test_stop_turn.json \
    --color_json tools/visualization/streamlit/results/color_out.json \
    --vehicle_json tools/visualization/streamlit/results/vehicle_out.json
```

## File formats

Directory structure:
```
video_dir
|
--- <track_id1>.mp4
--- <track_id2>.mp4

json_result_dir
|
--- <run1>.json
--- <run2>.json
```

Json result file:
```
{
    <query_id1>: [
        <track_id1>,
        <track_id2>,
    ],
    ...
}
```


Query json file
```
{
  <query_id1>: [
    <caption1>,
    <caption2>,
  ],
   <query_id2>: [
    <caption1>,
    <caption2>,
  ],
  ....
}
```

