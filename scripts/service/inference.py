from typing import Tuple
import numpy as np
import base64
from PIL import Image
import io

from torch import Tensor
import torch
from scripts.losses.loss import DiceLoss
from scripts.sam_train import SamTrain
from scripts.datasets.constant import DEFAULT_DEVICE

from segment_anything.build_sam import sam_model_registry
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
    def __init__(self, model_path: str, device: str = DEFAULT_DEVICE) -> None:
        model: Sam = sam_model_registry["vit_b"](
            checkpoint="./sam_vit_b_01ec64.pth", custom=model_path
        )
        model.to(device)
        self.sam_train = SamTrain(sam_model=model)
        self.device = device
        pass

    def inference(self, image: np.ndarray, mask: np.ndarray):
        img_emb, original_size, input_size = self.sam_train.prepare_img(image=image)
        _, _, _, mask_input_torch = self.sam_train.prepare_prompt(
            original_size=original_size, mask_input=mask[None, ...]
        )
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

        return mask_pred.numpy()

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


if __name__ == "__main__":
    service = InferenceService(None)
    s = service.encode_image(image=np.zeros((512, 512, 3)))
    print(s)
    pass
