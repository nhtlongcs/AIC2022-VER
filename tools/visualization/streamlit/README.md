# Visualization tools 

## Install 
install [Streamlit](https://docs.streamlit.io/en/stable/)
`pip install streamlit`

## Run

- Download video files using this [link](https://drive.google.com/file/d/1LSFgViybP_hjggoAwuFiJ-emzAlfy-b5/view?usp=sharing)
- Firstly, change path in `constants.py`. 

- To run streamlit with arguments, use `--` before flags

- To visualize data ground truth
```
streamlit run app_gt.py -- -i ./data
```

- To visualize prediction before submission 
```
streamlit run app_sub.py -- \
   -i ./data \
   --result_folder <path to submission folder>
```

- To visualize metadata (vehicle action, relation, color)
```
streamlit run app_meta.py -- \
  -i ./data \
  -s pseudo-text
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

