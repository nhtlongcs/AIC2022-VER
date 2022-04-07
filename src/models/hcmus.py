import torch
import torch.nn as nn
import torch.nn.functional as F
from src.extractors import EXTRCT_REGISTRY

from . import MODEL_REGISTRY
from .abstract import AICBase
from src.utils.losses import FocalLoss


@MODEL_REGISTRY.register()
class HCMUS(AICBase):
    def __init__(self, config):
        super().__init__(config)
        assert "task" in self.cfg.model.keys(), "task must be specified"
        self.task = self.cfg.model["task"]
        assert self.task in [
            "local_clf",
            "retrival",
        ], "task must be specified in [local_clf,retrival], but got {}".format(
            self.task
        )

    def init_model(self):

        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)  # ?
        NUM_COLORS = self.cfg.model["args"]["NUM_COLORS"]
        NUM_VEH_TYPES = self.cfg.model["args"]["NUM_VEHTYPE"]
        self.color_loss = FocalLoss(num_classes=NUM_COLORS)
        self.vehtype_loss = FocalLoss(num_classes=NUM_VEH_TYPES)
        embed_dim = self.cfg.model["args"]["EMBED_DIM"]
        # Define extractors for visual and language
        self.visualExtrct = EXTRCT_REGISTRY.get(
            self.cfg.extractors["img_encoder"]["name"]
        )(**self.cfg.extractors["img_encoder"]["args"])
        self.nlangExtrct = EXTRCT_REGISTRY.get(
            self.cfg.extractors["lang_encoder"]["name"]
        )(**self.cfg.extractors["lang_encoder"]["args"])
        self.visualExtrctBK = EXTRCT_REGISTRY.get(
            self.cfg.extractors["img_encoder"]["name"]
        )(**self.cfg.extractors["img_encoder"]["args"])

        self.img_in_dim = self.visualExtrct.feature_dim

        # Define the latent adaptation for visual backbones
        self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
        self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)

        # Something something
        self.domian_vis_fc_merge = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.vis_car_fc = nn.Sequential(
            nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim // 2)
        )
        self.lang_car_fc = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim // 2)
        )
        self.vis_motion_fc = nn.Sequential(
            nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim // 2)
        )
        self.lang_motion_fc = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim // 2)
        )

        self.domian_lang_fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Define specific loss for each features
        if self.cfg.model["args"]["car_idloss"]:
            self.id_cls = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.cfg.model["args"]["NUM_CLASS"]),
            )
        if self.cfg.model["args"]["mo_idloss"]:
            self.id_cls2 = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.cfg.model["args"]["NUM_CLASS"]),
            )
        if self.cfg.model["args"]["share_idloss"]:
            self.id_cls3 = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.cfg.model["args"]["NUM_CLASS"]),
            )

        if self.cfg.model["args"]["color_loss"]:
            self.color_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, NUM_COLORS),
            )
        if self.cfg.model["args"]["veh_type_loss"]:
            self.vehtype_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, NUM_VEH_TYPES),
            )

    def encode_nlang_feats(self, batch):
        assert "tokens" in batch.keys(), "Input dict must contain tokens"
        lang_embeds = self.nlangExtrct(batch["tokens"])
        lang_embeds = self.domian_lang_fc(lang_embeds)
        return lang_embeds

    def encode_visual_feats(self, batch):
        assert "images" in batch.keys(), "Input dict must contain cropped car images"
        assert "motions" in batch.keys(), "Input dict must contain motion images"
        # Extract visual croped features
        visual_embeddings = self.visualExtrct(batch["images"])
        visual_embeds = self.domian_vis_fc(visual_embeddings)
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)

        # Extract visual motion features
        motion_embeddings = self.visualExtrctBK(batch["motions"])
        motion_embeds = self.domian_vis_fc_bk(motion_embeddings)
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)
        return visual_embeds, motion_embeds

    def normalize_head(
        self,
        embeddings_ls,
    ):
        return map(lambda t: F.normalize(t, p=2, dim=-1), (embeddings_ls))

    def forward(self, batch):
        # 1. Extract features from the batch
        # 1.1 Extract natural language features
        (
            lang_embeds,
            lang_merge_embeds,
            lang_car_embeds,
            lang_mo_embeds,
        ) = self.query_embedding_head(batch)
        # 1.2 Extract visual features
        (
            visual_embeds,
            motion_embeds,
            visual_merge_embeds,
            visual_car_embeds,
            visual_mo_embeds,
        ) = self.track_embedding_head(batch)

        instance_logits = []
        if self.cfg.model["args"]["car_idloss"]:
            cls_logits = self.id_cls(visual_embeds)
            instance_logits.append(cls_logits)
        if self.cfg.model["args"]["mo_idloss"]:
            cls_logits2 = self.id_cls2(motion_embeds)
            instance_logits.append(cls_logits2)
        if self.cfg.model["args"]["share_idloss"]:
            merge_cls_t = self.id_cls3(lang_embeds)
            merge_cls_v = self.id_cls3(visual_merge_embeds)
            instance_logits.append(merge_cls_t)
            instance_logits.append(merge_cls_v)
        if self.cfg.model["args"]["color_loss"]:
            color_logits = self.color_head(visual_embeds)
        if self.cfg.model["args"]["veh_type_loss"]:
            vehtype_logits = self.vehtype_head(visual_embeds)

        pairs = [
            (visual_car_embeds, lang_car_embeds),
            (visual_mo_embeds, lang_mo_embeds),
            (visual_merge_embeds, lang_merge_embeds),
        ]

        return {
            "pairs": pairs[2],
            "all_pairs": pairs,
            "logit_scale": self.logit_scale,
            "instance_logits": instance_logits,
            "color_logits": color_logits,
            "veh_type_logits": vehtype_logits,
        }

    def compute_loss(
        self,
        all_pairs,
        logit_scale,
        instance_logits,
        color_logits,
        veh_type_logits,
        batch,
        **kwargs
    ):
        logit_scale = logit_scale.mean().exp()
        loss = 0
        if self.task == "retrieval":
            for (visual_embeds, lang_embeds) in all_pairs:
                sim_i_2_t = torch.matmul(
                    torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds)
                )
                sim_t_2_i = sim_i_2_t.t()
                loss_t_2_i = F.cross_entropy(
                    sim_t_2_i, torch.arange(batch["images"].size(0), device=self.device)
                )
                loss_i_2_t = F.cross_entropy(
                    sim_i_2_t, torch.arange(batch["images"].size(0), device=self.device)
                )
                loss += (loss_t_2_i + loss_i_2_t) / 2

            for instance_logits in instance_logits:
                loss += 0.5 * F.cross_entropy(instance_logits, batch["car_ids"].long())
        # task == 'local_clf'
        loss += 0.5 * self.color_loss(color_logits, batch["color_lbls"].long())
        loss += 0.5 * self.vehtype_loss(veh_type_logits, batch["vehtype_lbls"].long())
        return loss

    def color_branch(self, batch):
        (
            visual_embeds,
            motion_embeds,
            visual_merge_embeds,
            visual_car_embeds,
            visual_mo_embeds,
        ) = self.track_embedding_head(batch)
        color_logits = self.color_head(visual_embeds)
        return color_logits

    def vehtype_branch(self, batch):
        (
            visual_embeds,
            motion_embeds,
            visual_merge_embeds,
            visual_car_embeds,
            visual_mo_embeds,
        ) = self.track_embedding_head(batch)
        vehtype_logits = self.vehtype_head(visual_embeds)
        return vehtype_logits

    def query_embedding_head(self, batch, inference=False):
        lang_embeds = self.encode_nlang_feats(batch)
        lang_car_embeds = self.lang_car_fc(lang_embeds)
        lang_mo_embeds = self.lang_motion_fc(lang_embeds)
        (lang_merge_embeds, lang_car_embeds, lang_mo_embeds) = self.normalize_head(
            [lang_embeds, lang_car_embeds, lang_mo_embeds]
        )
        if inference:
            return {"ids": batch["ids"], "features": lang_merge_embeds}
        return lang_embeds, lang_merge_embeds, lang_car_embeds, lang_mo_embeds

    def track_embedding_head(self, batch, inference=False):
        visual_embeds, motion_embeds = self.encode_visual_feats(batch)

        visual_car_embeds = self.vis_car_fc(visual_embeds)
        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_merge_embeds = self.domian_vis_fc_merge(
            torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1)
        )

        (
            visual_merge_embeds,
            visual_car_embeds,
            visual_mo_embeds,
        ) = self.normalize_head(
            [visual_merge_embeds, visual_car_embeds, visual_mo_embeds]
        )

        if inference:
            return {"ids": batch["ids"], "features": visual_merge_embeds}

        return (
            visual_embeds,
            motion_embeds,
            visual_merge_embeds,
            visual_car_embeds,
            visual_mo_embeds,
        )

    def predict_step(self, batch, batch_idx):

        assert not (
            "tokens" in batch.keys() and "images" in batch.keys()
        ), "tokens and images are not allowed in batch at same time"

        if "tokens" in batch.keys():
            lang_embeds = self.query_embedding_head(batch, inference=True)
            return lang_embeds
        elif "images" in batch.keys() and "motions" in batch.keys():
            visual_embeds = self.track_embedding_head(batch, inference=True)
            return visual_embeds
        else:
            raise NotImplementedError
