from functools import partial
from typing import List, Optional, Tuple, Type
from torch import nn
from torch import Tensor
import torch
from torch.nn import Module
from scripts.sam_train import SamTrain
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.sam import Sam
from segment_anything.modeling.transformer import TwoWayTransformer


class ContextPromptEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        n_context: int,
        activation: Type[Module] = nn.GELU,
    ) -> None:
        super().__init__(
            embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation
        )
        self.n_context = n_context
        self.context_embedding = nn.Embedding(
            # n_context, 64 x 64 represent the image-size
            n_context, self.image_embedding_size[0] * self.image_embedding_size[1]
        )
        pass

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        context_number: int = None,
    ) -> Tuple[Tensor, Tensor]:
        sparse_embeddings, dense_embeddings = super().forward(points, boxes, masks)

        if context_number is None:
            return sparse_embeddings, dense_embeddings

        mask_context: Tensor = self.context_embedding(context_number)
        mask_context = (
            mask_context
            .reshape(-1, self.image_embedding_size[0], self.image_embedding_size[1])
            .unsqueeze(1)
            .expand(-1, self.embed_dim, -1, -1)
        )
        dense_embeddings = dense_embeddings + mask_context
        return sparse_embeddings, dense_embeddings


class ContextSam(Sam):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: ContextPromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__(
            image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std
        )

    pass


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    num_of_context=1,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = ContextSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=ContextPromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            n_context=num_of_context,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,  # 256
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        incompatible_keys = sam.load_state_dict(state_dict, strict=False)
        assert (
            len(incompatible_keys.unexpected_keys) == 0
        ), f"Incompatible keys: {incompatible_keys}"

        for key in list(incompatible_keys.missing_keys):
            assert key.startswith(
                "prompt_encoder.context_embedding."
            ), f"Wrong missing keys: {incompatible_keys}"
            pass
    return sam


def build_sam_context_vit_b(
    checkpoint=None,
    custom=None,
    num_of_context=1,
    check_for_context_weight: bool = False,
) -> ContextSam:
    sam = _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        num_of_context=num_of_context,
    )

    if custom:
        incompatible_keys = sam.load_state_dict(torch.load(custom), strict=False)
        assert (
            len(incompatible_keys.unexpected_keys) == 0
        ), f"Incompatible keys: {incompatible_keys}"

        for key in list(incompatible_keys.missing_keys):
            if check_for_context_weight:
                # Assert that all missing must be of image_encoder
                # -> context weight must provided
                assert key.startswith(
                    "image_encoder."
                ), f"Wrong missing keys: {incompatible_keys}"
            else:
                cond = key.startswith("image_encoder.") or key.startswith(
                    "prompt_encoder."
                )
                assert cond, f"Wrong missing keys: {incompatible_keys}"
            pass
    return sam


class ContextSamTrain(SamTrain):
    def __init__(self, sam_model: ContextSam) -> None:
        super().__init__(sam_model)
        assert isinstance(sam_model.prompt_encoder, ContextPromptEncoder)
        self.n_context = sam_model.prompt_encoder.n_context

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
        # Mask prompt
        mask_input: Optional[Tensor] = None,
        context_numbers: Optional[Tensor] = None,
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
            points=points, boxes=boxes, masks=mask_input, context_number=context_numbers
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


if __name__ == "__main__":
    sam_context = build_sam_context_vit_b(
        checkpoint="./sam_vit_b_01ec64.pth",
        custom="./runs/mask-prop-230509-005503/model-100.pt",
        num_of_context=13,
    )
    assert isinstance(sam_context.prompt_encoder, ContextPromptEncoder), f""

    context_train = ContextSamTrain(sam_model=sam_context)

    pass
