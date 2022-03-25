# python tests/extractors.py
from src.extractors import EXTRCT_REGISTRY

if __name__ == "__main__":
    print(EXTRCT_REGISTRY)
    for name, cls in EXTRCT_REGISTRY:
        args = {
            "EfficientNetExtractor": {"version": 0, "from_pretrained": True},
            "SENetExtractor": {"version": "senet154", "from_pretrained": "imagenet"},
            "LangExtractor": {"pretrained": "bert-base-uncased"},
        }
        model = EXTRCT_REGISTRY.get(name)(**args[name])
