import json
from typing import Dict, Tuple
import numpy as np
from torch.nn import functional as F
import base64
from PIL import Image
import io
import os
from torch import Tensor
import torch
from scripts.losses.loss import DiceLoss
from scripts.sam_train import SamTrain
from scripts.datasets.constant import DEFAULT_DEVICE

from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling import image_encoder
from segment_anything.modeling.sam import Sam

DICE_FN = DiceLoss(activation=None, reduction="none")


def pick_best_mask(
    pred_multi_mask: Tensor,
    previous_mask: np.ndarray,
    device: str,
) -> Tuple[int, float]:
    previous_mask = torch.as_tensor(previous_mask, device=device)[
        None, None, ...
    ].repeat_interleave(3, dim=1)
    dice = DICE_FN.run_on_batch(input=pred_multi_mask, target=previous_mask)
    chosen_idx = dice.argmin()
    return chosen_idx, dice.min().detach().cpu().item()


class InferenceService:
    def __init__(self, model_path: str = None, device: str = DEFAULT_DEVICE) -> None:
        MODULE_PATH = os.path.abspath(os.path.dirname(__file__))
        self.base_model_path = os.path.join(MODULE_PATH, "../../sam_vit_b_01ec64.pth")
        self.model_path = model_path or os.path.join(
            MODULE_PATH, "../../assets/imp-230603-150046-model-20.pt"
        )
        model: Sam = sam_model_registry["vit_b"](
            checkpoint=self.base_model_path, custom=self.model_path
        )
        model.to(device)
        self.sam_train = SamTrain(sam_model=model)
        self.device = device
        pass

    def mask_preprocess(self, mask: torch.Tensor):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        # Pad
        if not isinstance(mask, torch.Tensor):
            mask = torch.Tensor(mask)

        img_size = self.sam_train.model.prompt_encoder.mask_input_size
        h, w = mask.shape[-2:]
        padh = img_size[0] - h
        padw = img_size[1] - w
        mask = F.pad(mask, (0, padw, 0, padh))
        return mask

    def fake_mask(self, image_shape):
        """Fake cross mask

        Args:
            image_shape (tuple[int]): Image shape to simulate the mask

        Returns:
            np.ndarray: the fake mask
        """
        mask = np.zeros(image_shape[:2])
        h, w = mask.shape
        mask[int(h / 3.0) : int(h * 2.0 / 3.0), :] = 1.0
        mask[:, int(w / 3.0) : int(w * 2.0 / 3.0)] = 1.0
        return mask

    @torch.no_grad()
    def inference(self, image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        # This one resize the edge and apply padding
        img_emb, original_size, input_size = self.sam_train.prepare_img(image=image)
        # Down-size only 1 biggest side
        _, _, _, mask_input_torch = self.sam_train.prepare_prompt(
            original_size=original_size, mask_input=mask[None, ...]
        )
        # Add padding the same way the image receive
        mask_input_torch = self.mask_preprocess(mask_input_torch[0, 0])
        mask_input_torch = mask_input_torch[None, None]

        mask_pred, _, _ = self.sam_train.predict_torch(
            image_emb=img_emb,
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            return_logits=False,
            mask_input=mask_input_torch,
        )

        chosen_idx, _ = pick_best_mask(
            pred_multi_mask=mask_pred,
            previous_mask=mask,
            device=self.device,
        )
        mask_pred = mask_pred[0, chosen_idx]

        return mask_pred.cpu().numpy()

    def decode_image(self, encoded64: str) -> np.ndarray:
        image_decoded = base64.b64decode(encoded64)
        image = Image.open(io.BytesIO(image_decoded))
        imgArray = np.array(image)
        return imgArray

    def encode_image(self, image: np.ndarray) -> bytes:
        image = Image.fromarray(image, mode="RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        encoded_str = base64.b64encode(buffered.getvalue())
        return encoded_str


def parse_bbox_data(data: Dict[str, object]):
    bboxes = data["annotationData"]["bboxes"][0]
    x = int(bboxes["x"])
    y = int(bboxes["y"])
    w = int(bboxes["width"])
    h = int(bboxes["height"])
    return x, y, w, h


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    service = InferenceService(None)
    # # s = service.encode_image(image=np.zeros((512, 512, 3)))
    path = "/Users/hung.nh/Downloads/mockReq.txt"
    with open(path, "r") as out:
        data = json.load(out)

    data["annotationData"]["bboxes"]

    image = service.decode_image(data["image"])
    # mask = service.fake_mask(image.shape)
    # # mask[400:500] = 1.0
    # pred = service.inference(image=image, mask=mask)
    # mask = np.zeros(image.shape[:2])
    x, y, w, h = parse_bbox_data(data)
    # mask[y : y + h, x : x + w] = 1.0
    mask_out = image.copy()
    mask_out[y : y + h, x : x + w, :] = 0.0
    f, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].imshow(mask_out)
    # axes[2].imshow(mask)
    plt.show()

    pass
