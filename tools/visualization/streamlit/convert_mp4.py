import os 
import os.path as osp
from tqdm import tqdm

from constant import video_dir

save_dir = video_dir+'_valid'
os.makedirs(save_dir, exist_ok=True)

for fname in tqdm(os.listdir(video_dir)):
    src_path = osp.join(video_dir, fname)
    dst_path = osp.join(save_dir, fname)
    command = f'ffmpeg -i {src_path} -vcodec libx264 {dst_path}'
    os.system(command)
