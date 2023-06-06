from segment_anything import sam_model_registry, SamPredictor


class SAM(SamPredictor):
    def __init__(self, sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = super().__init__(sam)
