import torch
import os
from tqdm import tqdm
from scripts.render import Renderer
from segment_anything import SamPredictor, sam_model_registry
from scipy.ndimage import gaussian_filter
import numpy as np
from utils import \
    make_nested_dir, get_data_paths, GROUP2, GROUP3, \
    load_file_npz, load_img, argmax_dist, mask_out, resize


def load_model() -> SamPredictor:
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    return predictor


def positive_center_point(mask, class_number):
    filter_center_candidate = gaussian_filter(mask == class_number, 10)
    #  Precise positive
    [row, col] = np.argwhere(filter_center_candidate > 0.0)[0]
    return row, col


def positive_random_point(mask, class_number, center: list = None):
    [row, col] = center
    positive = np.argwhere((mask == class_number).astype(np.int16) > 0.0)
    choices = np.random.RandomState(seed=1810).choice(np.arange(
        positive.shape[0]),
                                                      size=10)
    c = argmax_dist(positive[choices], row, col)
    positive = positive[c:c + 1][:, ::-1]
    return positive


def negative_random_with_constrain(mask):
    filter_negative = gaussian_filter(mask == 0, 3)
    filter_negative = mask_out(filter_negative,
                               xmin=200,
                               xmax=450,
                               ymin=100,
                               ymax=450,
                               to_value=0.0)
    negative = np.argwhere(filter_negative > 0.0)
    choices = np.random.RandomState(seed=1810).choice(np.arange(
        negative.shape[0]),
                                                      size=1)
    # Pickup the choice and swap row with col
    negative = negative[choices][:, ::-1]
    return negative


def make_point_from_mask(mask, class_number):
    coors = np.argwhere(mask == class_number)

    if coors.shape[0] == 0: return None, None

    row, col = positive_center_point(mask, class_number)
    positive = positive_random_point(mask, class_number, [row, col])
    negative = negative_random_with_constrain(mask)

    # Make the col/row and label
    coors = np.array([[col, row], *positive, *negative])
    label = np.array(
        [1, *np.ones(positive.shape[0]), *np.zeros(negative.shape[0])])

    # In Cartesian Coordinate, number of row is y-axis
    return coors, label

def point_experiment():
    data_path, mask_path = get_data_paths(GROUP3)
    chosen_class = 1
    predictor = load_model()

    prefix = os.path.dirname(data_path[0])
    out_dir = os.path.join(f"{prefix}-fig-class-{chosen_class}")
    make_nested_dir(out_dir)
    point_coords = None
    point_labels = None
    r = Renderer()

    for path, mask_path in tqdm(zip(data_path, mask_path),
                                desc="Prediction...",
                                total=len(data_path)):
        img = load_img(path)
        mask = load_file_npz(mask_path)

        # Only calculate 1
        if point_coords is None or point_labels is None:
            point_coords, point_labels = make_point_from_mask(
                mask, chosen_class)

        index_number = int(
            os.path.basename(path).replace(".jpg", "").split('_')[-1])

        predictor.set_image(img)
        predict_mask, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
            )

        save_fig = os.path.join(out_dir, f"fig-{index_number}.png")
        
        r.add(img, None, [point_coords, point_labels], 'Raw image')
        r.add(
            img,
            (mask == chosen_class).astype(np.float16), 
            [point_coords, point_labels], 
            'GT'
        )
        r.add_multiple([{
                'img': img,
                'mask': predict_mask[idx],
                'points': None,
                'title': f'Predict mask {idx}'
            } for idx in range(predict_mask.shape[0])])
        
        r.show_all(save_path=save_fig)
        pass



def mask_experiment(scale_factor):
    chosen_class = 1
    data_path, mask_path = get_data_paths(GROUP2)
    predictor = load_model()
    mask_input_size = [4*x for x in predictor.model.prompt_encoder.image_embedding_size]
    torch.randn(1, *mask_input_size, dtype=torch.float32)
    prefix = os.path.dirname(data_path[0])
    
    out_dir = os.path.join(
        f"{prefix}-video-segment-with-gt")
    try:
        next_num = max([int(os.path.basename(name)) for name in os.listdir(out_dir)])
        next_num = next_num + 1
    except Exception:
        next_num = 0
        pass

    out_dir = os.path.join(out_dir, f"{next_num}")
    make_nested_dir(out_dir)

    point, label = np.array([[400, 300]]), np.array([1.])
    predict_masks = None

    r = Renderer()
    _mask = None

    for path, mask_path in tqdm(zip(data_path, mask_path),
                                desc="Prediction...",
                                total=len(data_path)):
        img = load_img(path)
        mask = load_file_npz(mask_path)
        # _mask = None

        index_number = int(
            os.path.basename(path).replace(".jpg", "").split('_')[-1])
        
        predictor.set_image(img)

        _mask = resize(mask == chosen_class) * scale_factor
        # _mask = resize(predict_masks[2], [256, 256]) \
        #     if predict_masks is not None else None
        
        # if _mask is not None:
        #     print(_mask.shape)
        #     pass
 
        predict_masks, _, _ = predictor.predict(
            # point_coords=point,
            # point_labels=label,
            multimask_output=True,
            mask_input=_mask,
            return_logits=False,
        )

        save_fig = os.path.join(out_dir, f"fig-{index_number}.png")

        r.add(img, None, [point, label], 'Raw image')
        r.add(
            img,
            (mask == chosen_class).astype(np.float16),
            [point, label], 
            'GT'
        )
        r.add_multiple([{
                'img': None,
                'mask': predict_masks[idx],
                'points': None,
                'title': f'Predict mask {idx}'
            } for idx in range(predict_masks.shape[0])]
            )
        
        r.add(None, np.array(_mask[0]), None, f'Input mask - scale - {scale_factor}')
        
        r.show_all(save_path=save_fig)
        r.reset()

        point = None
        label = None



if __name__ == "__main__":
    # point_experiment()
    for scale_factor in [1.0]:
        print(f"Start exp - {scale_factor}")
        try:
            mask_experiment(scale_factor=scale_factor)
        except Exception as msg:
            print(f"Exp {scale_factor} with excep: {msg}")
    pass