# Visualization tools 

## Install 
install [Streamlit](https://docs.streamlit.io/en/stable/)
`pip install streamlit`

## Run

- To run streamlit with arguments, use `--` before flags
```
streamlit run app.py -- \
  --result_folder ./results \
  --query_json ./test-queries.json \
  --video_dir ./track_videos
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
