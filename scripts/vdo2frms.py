import cv2
from tqdm.auto import tqdm
from pathlib import Path
import multiprocessing

N_WORKERS = multiprocessing.cpu_count() // 2


def extract(data):
    vdo_path, out_path = data
    assert vdo_path.exists(), f"Video file {str(vdo_path)} does not exist"

    filename = vdo_path.stem
    path_data = vdo_path.parent
    out_path.mkdir(exist_ok=True, parents=True)

    vidcap = cv2.VideoCapture(str(vdo_path))
    success, image = vidcap.read()
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 1
    with tqdm(total=video_length) as pbar:
        while success:
            path_image = out_path / f"{idx:06d}.jpg"
            if not path_image.exists():
                cv2.imwrite(str(path_image), image)
            success, image = vidcap.read()
            pbar.set_description(f"Processing {path_data.name} {filename}")
            pbar.update(1)
            idx += 1


def parallel_task(task, files):
    with multiprocessing.Pool(N_WORKERS) as pool:
        for imgs in tqdm(pool.imap_unordered(task, files)):
            pass


if __name__ == "__main__":
    TRAINDIR = Path("data/AIC22_Track2_NL_Retrieval/train/")
    VALIDDIR = Path("data/AIC22_Track2_NL_Retrieval/validation/")
    OUTDIR = Path("data/meta/extracted_frames/")

    # for stage, path in [("train", TRAINDIR), ("validation", VALIDDIR)]:
    for stage, path in [("validation", VALIDDIR)]:
        for d in path.glob("*"):
            camera_name = d.name
            files = []
            print(f"Processing [{stage}][{camera_name}]")
            for vdo_path in d.glob("*"):
                if vdo_path.is_dir():
                    vdo_name = vdo_path.name
                    outpath = OUTDIR / "train" / camera_name / vdo_name / "img1"
                    files.append((vdo_path / "vdo.avi", outpath))
            parallel_task(extract, files)
