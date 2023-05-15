from typing import Optional, Tuple

from tqdm import tqdm
from segment_anything.modeling.sam import Sam
from segment_anything.build_sam import sam_model_registry

import numpy as np
import torch
from torch import Tensor

from segment_anything.utils.transforms import ResizeLongestSide


class SamTrain:
    """A wrapper around SAM, like the SamPredictor,
    to make the training become more simple
    """

    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.mask_transform = ResizeLongestSide(
            target_length=sam_model.image_encoder.img_size // 4
        )

    def predict_torch(
        self,
        # Image emb.
        image_emb: Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
        # Prompt.
        point_coords: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
        boxes: Optional[Tensor] = None,
        mask_input: Optional[Tensor] = None,
        # Output option
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Check the original source at segment_anything/predictor for more info
        """

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Force the sparse-embedding into 0.0 so that it won't affect
        # mask-decoder
        if sparse_embeddings.shape[1] == 0:
            sparse_embeddings = torch.empty(
                (1, *sparse_embeddings.shape[1:]), device=self.device
            )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, input_size, original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def prepare_img(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
            None, :, :, :
        ]

        img_emb, original_size, input_size = self._calculate_emb(
            input_image_torch, image.shape[:2]
        )
        return img_emb, original_size, input_size

    def prepare_prompt(
        self,
        original_size: Tuple[int, ...],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
    ):
        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

        if point_coords is not None:
            coords_torch, labels_torch = self._prepare_point(
                original_size, point_labels, point_coords
            )

        if box is not None:
            box_torch = self._prepare_box(box, original_size)

        if mask_input is not None:
            mask_input_torch = self._prepare_mask_input(mask_input)

        return coords_torch, labels_torch, box_torch, mask_input_torch

    @torch.no_grad()
    def _calculate_emb(
        self,
        transformed_image: Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."

        original_size = original_image_size
        input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        img_emb = self.model.image_encoder(input_image)
        return img_emb, original_size, input_size

    def _prepare_mask_input(self, mask_input):
        """Formatting Mask input
        Args:
            mask_input (np.ndarray | torch.Tensor): shape = [batch_size, H, W]
        """
        mask_input_torch = torch.as_tensor(
            mask_input, dtype=torch.float, device=self.device
        )
        # Add feature dim -> [batch_size, 1, H, W]
        mask_input_torch = mask_input_torch.unsqueeze(1)
        mask_input_torch = self.mask_transform.apply_image_torch(mask_input_torch)
        return mask_input_torch

    def _prepare_box(self, box, original_size):
        box = self.transform.apply_boxes(box, original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
        box_torch = box_torch[None, :]
        return box_torch

    def _prepare_point(self, img_original_size, point_labels, point_coords):
        assert (
            point_labels is not None
        ), "point_labels must be supplied if point_coords is supplied."

        point_coords = self.transform.apply_coords(point_coords, img_original_size)
        coords_torch = torch.as_tensor(
            point_coords, dtype=torch.float, device=self.device
        )
        labels_torch = torch.as_tensor(
            point_labels, dtype=torch.int, device=self.device
        )
        coords_torch = coords_torch.view(-1, *coords_torch.shape[-2:])
        labels_torch = labels_torch.view(-1, *labels_torch.shape[-1:])

        return coords_torch, labels_torch

    @property
    def device(self) -> torch.device:
        return self.model.device


def load_model(checkpoint="./sam_vit_b_01ec64.pth", checkpoint_type="vit_b") -> Sam:
    sam: Sam = sam_model_registry[checkpoint_type](checkpoint=checkpoint)
    return sam


if __name__ == "__main__":
    pass
