import os
from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(
        self,
        model_name: str = "ZhengPeng7/BiRefNet",
        fallback_model_name: Optional[str] = None,
    ):
        requested_model_name = os.environ.get("PIXAL3D_REMBG_MODEL", model_name)
        fallback_model_name = (
            fallback_model_name
            or os.environ.get("PIXAL3D_REMBG_FALLBACK_MODEL")
            or "ZhengPeng7/BiRefNet"
        )

        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                requested_model_name, trust_remote_code=True
            )
        except Exception as exc:
            if requested_model_name == fallback_model_name:
                raise

            print(
                f"[Rembg] Failed to load {requested_model_name}: {exc}. "
                f"Falling back to {fallback_model_name}."
            )
            self.model = AutoModelForImageSegmentation.from_pretrained(
                fallback_model_name, trust_remote_code=True
            )

        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    