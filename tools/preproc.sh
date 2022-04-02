DATAPATH=$1
# ALIBABA PART
# generate motion map
python scripts/data/motion_map.py $DATAPATH  && echo OK || echo Run FAILED please check

# Generate augment data for training (Optional)

# python -m spacy download en_core_web_sm # uncomment if you need to install spacy
python scripts/data/nlpaug_uts.py $DATAPATH/train_tracks.json && echo OK || echo Run FAILED please check
python scripts/data/nlpaug_uts.py $DATAPATH/test_queries.json && echo OK || echo Run FAILED please check

# Split data, train and test data into train and test data (Optional)
# By running the following commands, you can split the data into train and test data in same folder.

python scripts/data/split.py $DATAPATH/train_tracks.json && echo OK || echo Run FAILED please check

# Extract train and test's queries into separated parts following the English PropBank Semantic Role Labeling rules.
# SRL PART
# python scripts/srl/extraction.py <input_data_path> <output_metadata_srl_path>
python scripts/srl/extraction.py $DATAPATH $DATAPATH && echo OK || echo Run FAILED please check

SRL_PATH=$DATAPATH/srl
mkdir $SRL_PATH
mkdir $SRL_PATH/action
mkdir $SRL_PATH/color
mkdir $SRL_PATH/veh

python scripts/srl/action_prep.py $SRL_PATH $SRL_PATH/action && echo OK || echo Run FAILED please check
python scripts/srl/color_prep.py $DATAPATH $SRL_PATH $DATAPATH/extracted_frames $SRL_PATH/color && echo OK || echo Run FAILED please check
python scripts/srl/veh_prep.py $DATAPATH $SRL_PATH $DATAPATH/extracted_frames $SRL_PATH/veh && echo OK || echo Run FAILED please check

# Please check this part
mkdir $SRL_PATH/postproc
python scripts/data/convert_order.py $DATAPATH $SRL_PATH/postproc && echo OK || echo Run FAILED please check
python scripts/srl/extract_postproc.py $SRL_PATH $SRL_PATH/postproc $SRL_PATH/postproc && echo OK || echo Run FAILED please check
