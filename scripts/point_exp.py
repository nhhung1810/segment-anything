import os
from tqdm import tqdm
from scripts.render import Renderer
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from utils import (
    make_nested_dir,
    get_data_paths,
    GROUP3,
    load_file_npz,
    load_img,
    GROUP1,
    GROUP2,
)

from point_utils import PointUtils

cache = {}


def extract_index(path):
    index_number = int(os.path.basename(path).replace(".jpg", "").split("_")[-1])
    return index_number


def cache_set_image(predictor: SamPredictor, path, img):
    if not cache.get(path, None):
        predictor.set_image(img)
        cache[path] = [
            predictor.original_size,
            predictor.input_size,
            predictor.get_image_embedding(),
        ]

        return
    else:
        predictor.reset_image()
        [ori, inp, emb] = cache[path]
        predictor.features = emb
        predictor.original_size = ori
        predictor.input_size = inp
        predictor.is_image_set = True
    pass


def point_experiment(
    predictor: SamPredictor, weight_name: str, chosen_class, class_name
):
    point_generator = PointUtils()
    data_path, mask_path = get_data_paths(GROUP1)
    prefix = os.path.dirname(data_path[0])

    out_dir = os.path.join(f"{prefix}-{weight_name}-{class_name}")
    make_nested_dir(out_dir)
    point_coords = None
    point_labels = None
    r = Renderer()

    for path, mask_path in tqdm(
        zip(data_path, mask_path), desc="Prediction...", total=len(data_path)
    ):
        img = load_img(path)
        mask = load_file_npz(mask_path)
        index_number = extract_index(path)

        # Only calculate 1
        if point_coords is None or point_labels is None:
            point_coords, point_labels = point_generator.no_prompt(
                # mask, chosen_class
            )

        # predictor.set_image(img)
        cache_set_image(predictor, path, img)
        predict_mask, pred_iou, _ = predictor.predict(
            point_coords=point_coords, point_labels=point_labels, multimask_output=True
        )

        save_fig = os.path.join(out_dir, f"fig-{index_number}.png")

        r.add(img, None, [point_coords, point_labels], "Raw image")
        r.add(
            img,
            mask.astype(np.float16),
            [None, None],
            "All class GT",
        )
        r.add(
            img,
            (mask == chosen_class).astype(np.float16),
            [point_coords, point_labels],
            "GT",
        )
        r.add_multiple(
            [
                {
                    "img": img,
                    "mask": predict_mask[idx],
                    "points": None,
                    "title": f"Pred {idx}: {pred_iou[idx]:.2f}",
                }
                for idx in range(predict_mask.shape[0])
            ]
        )

        r.show_all(save_path=save_fig)
        r.reset()
        pass


def load_model(
    pre_trained="./sam_vit_b_01ec64.pth", custom_model: str = None
) -> SamPredictor:
    sam = sam_model_registry["vit_b"](checkpoint=pre_trained, custom=custom_model)
    return SamPredictor(sam)


if __name__ == "__main__":
    CHECKPOINT_DIR = "./runs/sam_simple_obj_train-230425-214245"
    model_list = [
        dict(model_name="pretrained", model_path=None),
        dict(model_name="20", model_path=f"{CHECKPOINT_DIR}/model-20.pt"),
        dict(model_name="40", model_path=f"{CHECKPOINT_DIR}/model-40.pt"),
        dict(model_name="60", model_path=f"{CHECKPOINT_DIR}/model-60.pt"),
        dict(model_name="80", model_path=f"{CHECKPOINT_DIR}/model-80.pt"),
        dict(model_name="100", model_path=f"{CHECKPOINT_DIR}/model-100.pt"),
    ]
    for entry in model_list:
        model = load_model(custom_model=entry["model_path"])
        model_name = entry["model_name"]
        point_experiment(model, model_name, chosen_class=1, class_name="liver")
    pass
