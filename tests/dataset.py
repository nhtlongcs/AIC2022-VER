from src.datasets import DATASET_REGISTRY
from opt import Opts
import torchvision

if __name__ == "__main__":
    cfg = Opts(
        cfg="/Users/nhtlong/workspace/aic/AIC2022-VER/configs/template.yml"
    ).parse_args()
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_data = DATASET_REGISTRY.get("CityFlowNLDataset")(
        cfg.data, json_path=cfg.data["TRAIN_JSON_PATH"], transform=image_transform
    )
    crop, text, bk, tmp_index = train_data[0]
    print(text)
    print(crop.shape)
    print(bk.shape)
