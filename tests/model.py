# python tests/model.py
from src.models import MODEL_REGISTRY
from opt import Opts

if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    print(MODEL_REGISTRY)
    model = MODEL_REGISTRY.get("UTS")(cfg)
    model.prepare_data()
