# python tests/dataset.py
import torchvision
from opt import Opts
from src.datasets import DATASET_REGISTRY
from torch.utils.data import DataLoader

if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    print(DATASET_REGISTRY)
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((288, 288)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    ds = DATASET_REGISTRY.get("CityFlowNLDataset")(
        **cfg.data["args"]["train"],
        data_cfg=cfg.data["args"],
        tok_model_name=cfg.extractors["lang_encoder"]["args"]["pretrained"],
        transform=image_transform,
    )
    dataloader = DataLoader(
        ds, collate_fn=ds.collate_fn, batch_size=8, shuffle=False, num_workers=0
    )
    for i, batch in enumerate(dataloader):
        print(batch["images"].shape)
        print(batch["tokens"]["input_ids"].shape)
        print(batch["texts"])
        print(batch["car_ids"])
        break
