from glob import glob
import os
from tqdm import tqdm
from scripts.constants import (
    FLARE22_SMALL_PATH,
    IMAGE_TYPE,
    TRAIN_GROUP1,
    TRAIN_GROUP2,
    TRAIN_GROUP3,
    VAL_GROUP1,
    VAL_GROUP2,
    VAL_GROUP3,
)
from scripts.render import Renderer
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from utils import (
    make_directory,
    load_file_npz,
    load_img,
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


def get_patient_name(path: str):
    return (
        os.path.basename(path)
        .replace(IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER.value, "")
        .replace(IMAGE_TYPE.CHEST_LUNGS_CHEST_MEDIASTINUM.value, "")
        .replace(IMAGE_TYPE.SPINE_BONE.value, "")
        .strip("_")
    )


def get_image_type(path: str):
    path = os.path.basename(path)
    for v in IMAGE_TYPE:
        if v.value in path:
            return v.value
    raise Exception(f"{path}")


def point_experiment(
    predictor: SamPredictor,
    weight_name: str,
    chosen_class,
    class_name,
    group_name,
    out_dir: str,
    data_mask_path,
):
    point_generator = PointUtils()
    [data_paths, mask_paths] = data_mask_path
    prefix = os.path.dirname(data_paths[0])
    patient_name = get_patient_name(prefix)
    type_name = get_image_type(prefix)
    out_dir = os.path.join(
        f"{out_dir}/{patient_name}/{type_name}-{class_name}/{weight_name}"
    )
    make_directory(out_dir)
    point_coords = None
    point_labels = None
    r = Renderer()

    for path, mask_path in tqdm(
        zip(data_paths, mask_paths), desc="Prediction...", total=len(data_paths)
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
            mask,
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


def make_data_mask_path(group_name):
    data_dir = f"{FLARE22_SMALL_PATH}/{group_name}/*"
    mask_dir = os.path.basename(os.path.dirname(group_name))
    mask_dir = f"{FLARE22_SMALL_PATH}/{mask_dir}/masks/*"
    return sorted(list(glob(data_dir))), sorted(list(glob(mask_dir)))


if __name__ == "__main__":
    CHECKPOINT_DIR = "runs/sam-fix-iou-230427-144454"
    groups = [VAL_GROUP1, VAL_GROUP2, VAL_GROUP3]
    model_list = [
        dict(model_name="pretrained", model_path=None),
        dict(model_name="20", model_path=f"{CHECKPOINT_DIR}/model-20.pt"),
        dict(model_name="40", model_path=f"{CHECKPOINT_DIR}/model-40.pt"),
        dict(model_name="60", model_path=f"{CHECKPOINT_DIR}/model-60.pt"),
        dict(model_name="80", model_path=f"{CHECKPOINT_DIR}/model-80.pt"),
        dict(model_name="100", model_path=f"{CHECKPOINT_DIR}/model-100.pt"),
        dict(model_name="120", model_path=f"{CHECKPOINT_DIR}/model-120.pt"),
        dict(model_name="140", model_path=f"{CHECKPOINT_DIR}/model-140.pt"),
        dict(model_name="160", model_path=f"{CHECKPOINT_DIR}/model-160.pt"),
        dict(model_name="180", model_path=f"{CHECKPOINT_DIR}/model-180.pt"),
        dict(model_name="200", model_path=f"{CHECKPOINT_DIR}/model-200.pt"),
    ]

    for entry in model_list:
        for group_name in groups:
            model = load_model(custom_model=entry["model_path"])
            model_name = entry["model_name"]
            point_experiment(
                model,
                model_name,
                chosen_class=1,
                class_name="liver",
                group_name=group_name,
                out_dir=f"{FLARE22_SMALL_PATH}/fix-iou-grad-acc-4-no-point/",
                data_mask_path=make_data_mask_path(group_name),
            )
    pass
