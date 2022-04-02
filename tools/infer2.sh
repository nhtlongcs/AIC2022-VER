CUDA_VISIBLE_DEVICES=0  python inference_cmplx.py \
                        -c test.yml \
                        -o data.text.json_path=data_sample/meta/test_queries.json \
                        data.track.json_path=data_sample/meta/test_tracks.json \
                        data.track.image_dir=data_sample/meta/extracted_frames/ \
                        data.track.motion_path=data_sample/meta/motion_map
