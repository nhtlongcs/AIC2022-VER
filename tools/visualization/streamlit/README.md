# Visualization tools 

## Install 
install [Streamlit](https://docs.streamlit.io/en/stable/)
`pip install streamlit`

## Run
streamlit run app.py

## File formats

Json result file:
```
{
    <query_id1>: [
        <video1_id1>,
        <video1_id2>,
    ],
    ...
}
```

Track video directory:
```
<video1_id1>.mp4
<video1_id2>.mp4
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

