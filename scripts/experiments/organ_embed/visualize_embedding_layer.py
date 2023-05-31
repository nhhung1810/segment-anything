from glob import glob
import os
import natsort
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.datasets.constant import FLARE22_LABEL_ENUM
from scripts.experiments.organ_embed.model import ContextPromptEncoder, ContextSam, build_sam_context_vit_b
import torch

from scripts.render.render_engine import RenderEngine
from scripts.utils import make_directory


def build_model(path) -> torch.nn.Embedding:
    model: ContextSam = build_sam_context_vit_b(
        checkpoint="./sam_vit_b_01ec64.pth",
        custom=path,
        check_for_context_weight=False,
        num_of_context=14,
    )
    model.to('cpu')
    assert isinstance(model.prompt_encoder, ContextPromptEncoder), ""
    context_embedding = model.prompt_encoder.context_embedding
    assert context_embedding.weight.shape[0] == 14, f"Mismatch shape of embedding : {context_embedding.weight.shape}"
    return context_embedding

@torch.no_grad()
def visualize_embedding(emb: torch.nn.Embedding, model_name:str, idx: int, output_dir: str):
    for organ in FLARE22_LABEL_ENUM:
        if organ == FLARE22_LABEL_ENUM.BACK_GROUND: continue
        
        save_name = f"{output_dir}/{organ.name}/{idx:0>4}.png"
        make_directory(save_name, is_file=True)
        img : torch.Tensor = emb(torch.LongTensor([organ.value]))
        img =  img.reshape(64, 64, 1).repeat_interleave(3, -1).cpu().numpy()
        min_img = img.min()
        max_img = img.max()
        img = (img - img.min() / (img.max() - img.min())) * 255.0 
        img = img.astype(np.uint8)
        f, ax = plt.subplots(1, 1, squeeze=True)
        ax.imshow(img)
        ax.set_title(f"{model_name.replace('.pt', '')}-{organ.name}: {min_img:.4f}..{max_img:.4f}")
        f.savefig(save_name)
        plt.close()
    pass



if __name__ == "__main__":
    model_dir = f"./runs/organ-ctx2-230527-215453"
    output_dir = f"{model_dir}/context_emb_visual/"
    make_directory(output_dir)
    paths = list(natsort.natsorted(glob(f"{model_dir}/model-*.pt")))
    for idx, p in enumerate(paths):
        emb = build_model(p)
        name = os.path.basename(p)
        visualize_embedding(emb, name, idx, output_dir)
    pass