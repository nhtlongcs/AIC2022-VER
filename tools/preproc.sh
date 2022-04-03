DATAPATH=$1
# ALIBABA PART
# generate motion map
echo Extracting motion map ...
python scripts/data/motion_map.py $DATAPATH  && echo DONE || echo Run FAILED, please check

# Generate augment data for training (Optional)

# echo Downloading spacy model ... && python -m spacy download en_core_web_sm # uncomment if you need to install spacy
echo Augmenting data ...
python scripts/data/nlpaug_uts.py $DATAPATH/train_tracks.json && echo DONE || echo Run FAILED, please check
python scripts/data/nlpaug_uts.py $DATAPATH/test_queries.json && echo DONE || echo Run FAILED, please check

# Split data, train and test data into train and test data (Optional)
# By running the following commands, you can split the data into train and test data in same folder.

echo Spliting data ...
python scripts/data/split.py $DATAPATH/train_tracks.json && echo DONE || echo Run FAILED, please check

# Extract train and test's queries into separated parts following the English PropBank Semantic Role Labeling rules.
# SRL PART
# python scripts/srl/extraction.py <input_data_path> <output_metadata_srl_path>
echo Extracting SRL data ...
python scripts/srl/extraction.py $DATAPATH $DATAPATH && echo DONE || echo Run FAILED, please check

SRL_PATH=$DATAPATH/srl
mkdir $SRL_PATH
mkdir $SRL_PATH/action
mkdir $SRL_PATH/color
mkdir $SRL_PATH/veh

echo Extracting SRL data [action] ...
python scripts/srl/action_prep.py $SRL_PATH $SRL_PATH/action && echo DONE || echo Run FAILED, please check
echo Extracting SRL data [color] ...
python scripts/srl/color_prep.py $DATAPATH $SRL_PATH $DATAPATH/extracted_frames $SRL_PATH/color && echo DONE || echo Run FAILED, please check
echo Extracting SRL data [vehicle] ...
python scripts/srl/veh_prep.py $DATAPATH $SRL_PATH $DATAPATH/extracted_frames $SRL_PATH/veh && echo DONE || echo Run FAILED, please check

# Please check this part
mkdir $SRL_PATH/postproc
echo Extracting SRL data [postproc] ...
python scripts/data/convert_order.py $DATAPATH $SRL_PATH/postproc && echo DONE || echo Run FAILED, please check
python scripts/srl/extract_postproc.py $SRL_PATH $SRL_PATH/postproc $SRL_PATH/postproc && echo DONE || echo Run FAILED, please check
