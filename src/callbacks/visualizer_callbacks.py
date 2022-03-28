import wandb
import os
import cv2
import json
import numpy as np
import os.path as osp
from pytorch_lightning.callbacks import Callback
from torchvision.transforms import functional as TFF
from src.utils.visualization.visualizer import Visualizer

class VisualizerCallback(Callback):

    def __init__(self, motion_path, gt_json_path, query_results_json, mapping_json) -> None:
        super().__init__()
        self.motion_path = motion_path
        self.gt_json_path = gt_json_path
        self.query_results_json = query_results_json
        self.mapping_json = mapping_json
        self.visualizer = Visualizer()

    def on_validation_epoch_start(self, trainer, pl_module):
        # Save mapping for visualization
        os.makedirs('temps', exist_ok=True)
        with open('temps/track_id_mapping.json', 'w') as f:
            json.dump(pl_module.val_dataset.list_of_uuids, f)

    def on_validation_epoch_end(self, trainer, pl_module):

        """
        After finish validation
        """

        with open(self.query_results_json, 'r') as f:
            results = json.load(f)
        with open(self.mapping_json, 'r') as f:
            mapping = json.load(f)

        indexes = [i for i in np.random.randint(0, len(mapping), size=5)]
        my_table = []
        columns = ['id', 'queries', 'prediction', 'groundtruth']
        for index in indexes:
            query_id = mapping[index]
            pred_ids = [mapping[i] for i in results[str(index)]['pred_ids']]
            target_ids = [mapping[i] for i in results[str(index)]['target_ids']]
            scores = results[str(index)]['scores']
            
            # Predictions
            pred_batch = []
            pred_images = [show_motion(i, self.motion_path) for i in pred_ids]
            for idx, (img_show, prob) in enumerate(zip(pred_images, scores)):
                self.visualizer.set_image(img_show)
                self.visualizer.draw_label(
                    f"C: {prob:.4f}", 
                    fontColor=[1,0,0], 
                    fontScale=0.5,
                    thickness=1,
                    outline=None,
                    offset=30
                )
                pred_img = self.visualizer.get_image()
                img_show = TFF.to_tensor(pred_img)
                pred_batch.append(img_show)
            pred_grid_img = self.visualizer.make_grid(pred_batch)

            # Ground truth
            gt_batch = []
            target_images = [show_motion(i, self.motion_path) for i in target_ids]
            for idx, img_show in enumerate(target_images):
                img_show = TFF.to_tensor(img_show)
                gt_batch.append(img_show)
            gt_grid_img = self.visualizer.make_grid(gt_batch)

            # Query texts
            texts = show_texts(query_id, self.gt_json_path)
            query = "\n".join([f"{i}. "+text for i, text in enumerate(texts)])
            record = [
                query_id, 
                query, 
                wandb.Image(pred_grid_img.permute(2,0,1)), 
                wandb.Image(gt_grid_img.permute(2,0,1))
            ]
            my_table.append(record)

        trainer.logger.log_table(
            "val/prediction", data=my_table, columns=columns
        )

def show_motion(track_id, motion_dir):
    motion_image = cv2.imread(osp.join(motion_dir, track_id+'.jpg'))
    motion_image = cv2.resize(motion_image, (200,200))
    motion_image = cv2.cvtColor(motion_image, cv2.COLOR_BGR2RGB)
    return motion_image

def show_texts(track_id, json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    texts = data[track_id]['nl']
    return texts