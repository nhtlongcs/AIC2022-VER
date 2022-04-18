# Streamlit: Visualization is so lit!

## Install 
- Install [Streamlit](https://docs.streamlit.io/en/stable/)
```
pip install streamlit
```

## Data preparation
- Generate track videos using scripts in `tools/visualization/video_gen`
- Prepare required files which are specified in `constants.py` and change paths. 

## How to run
- To run streamlit with arguments, use `--` before flags
- To visualize prediction before submission 
```
streamlit run app.py -- \
   -i ./data \
   -s <pseudo-test or test>
```