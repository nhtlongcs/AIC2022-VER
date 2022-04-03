import torch
import pytest
from tqdm.auto import tqdm
from pathlib import Path

from src.models import MODEL_REGISTRY
from src.datasets import DATASET_REGISTRY
from src.metrics import METRIC_REGISTRY
from opt import Opts
import torchvision
from src.metrics.metric_wrapper import RetrievalMetric


@pytest.mark.parametrize("model_name", ["UTS"])
def test_evaluate(tmp_path, model_name):
    cfg_path = "tests/configs/default.yml"
    assert Path(cfg_path).exists(), "config file not found"
    cfg = Opts(cfg=cfg_path).parse_args([])

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
    dataloader = torch.utils.data.DataLoader(
        **cfg.data["args"]["train"]["loader"], dataset=ds, collate_fn=ds.collate_fn,
    )
    model = MODEL_REGISTRY.get(model_name)(cfg)
    metric = RetrievalMetric(
        metrics=[
            METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
            if mcfg["args"]
            else METRIC_REGISTRY.get(mcfg["name"])()
            for mcfg in cfg["metric"]
        ],
        **cfg["metric_configs"],
    )
    for i, batch in tqdm(enumerate(dataloader), total=5):
        pairs = model(batch)
        metric.update(pairs, batch)
        if (i % 5 == 0) and (i > 0):
            metric_dict = metric.value()

            # Log string
            log_string = ""
            for metric_name, score in metric_dict.items():
                if isinstance(score, (int, float)):
                    log_string += metric_name + ": " + f"{score:.5f}" + " | "
            log_string += "\n"
            print(log_string)

            # 4. Reset metric
            metric.reset()
            break

