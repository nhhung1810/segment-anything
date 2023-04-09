
import itertools
import os
from pathlib import Path
from time import time_ns
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils import make_nested_dir, get_data_paths, GROUP1, GROUP2, GROUP3, load_file_npz, load_img, argmax_dist



def load_model() -> SamPredictor:
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    return predictor


def mask_out(mask, xmin, xmax, ymin, ymax, to_value):
    _mask = np.ones(mask.shape) == 1.0
    _mask[xmin:xmax, ymin:ymax] = False
    mask[_mask] = to_value
    return mask

def positive_center_point(mask, class_number):
    filter_center_candidate = gaussian_filter(mask == class_number, 10)
    #  Precise positive
    [row, col] = np.argwhere(filter_center_candidate > 0.0)[0]
    return row, col

def positive_random_point(mask, class_number, center:list=None):
    [row, col] = center
    positive = np.argwhere((mask == class_number).astype(np.int16) > 0.0)
    choices = np.random.RandomState(seed=1810).choice(np.arange(positive.shape[0]), size=10)
    c = argmax_dist(positive[choices], row, col)
    positive = positive[c:c+1][:, ::-1]
    return positive

def negative_random_with_constrain(mask):
    filter_negative = gaussian_filter(mask == 0, 3)
    filter_negative = mask_out(
        filter_negative, xmin=200, xmax=450, ymin=100, ymax=450, to_value=0.0)
    negative = np.argwhere(filter_negative > 0.0)
    choices = np.random.RandomState(seed=1810).choice(np.arange(negative.shape[0]), size=1)
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
    coors = np.array([
        [col, row],
        *positive,
        *negative
    ])
    label = np.array([
        1,
        *np.ones(positive.shape[0]),
        *np.zeros(negative.shape[0])
    ]
    )
    
    # In Cartesian Coordinate, number of row is y-axis
    return coors, label

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def render_img(show_im, save_fig_path, point_coords: np.ndarray, point_labels: np.ndarray):
    # Convention: show_im: [img, gt, mask1, mask2, ...]
    n_img = len(show_im)
    n_row = int(np.floor(np.sqrt(n_img)))
    n_row, n_col = n_row, n_row + 1

    f, axes = plt.subplots(n_row, n_col)
    for idx_1 in range(n_row):
        for idx_2 in range(n_col):
            idx = idx_1 * n_col + idx_2
            if idx >= n_img: break
            
            if idx != 0:
                axes[idx_1, idx_2].imshow(show_im[0])
                show_mask(show_im[idx], axes[idx_1, idx_2])
            else:
                axes[idx_1, idx_2].imshow(show_im[0])
                
                # Positive value
                msk = point_labels == 1
                if msk.any():
                    xs = point_coords[msk][:, 0]
                    ys = point_coords[msk][:, 1]
                    axes[idx_1, idx_2].scatter(xs, ys, c='g')
                # Negative value
                msk = point_labels == 0
                if msk.any():
                    xs = point_coords[msk][:, 0]
                    ys = point_coords[msk][:, 1]
                    axes[idx_1, idx_2].scatter(xs, ys, c='r')

            if idx == 0:
                axes[idx_1, idx_2].set_title('Image with interactive point')
            elif idx == 1:
                axes[idx_1, idx_2].set_title('GTruth')
            else:
                axes[idx_1, idx_2].set_title(f'Mask proposal - {idx}')
                pass
    if not save_fig_path:
        f.show()
    else:
        f.savefig(save_fig_path)
        # Prevent leaking memory
        plt.close()
    pass

if __name__ == "__main__":
    data_path, mask_path = get_data_paths(GROUP3)
    chosen_class = 1
    predictor = load_model()
    
    prefix = os.path.dirname(data_path[0])
    out_dir = os.path.join(f"{prefix}-fig-class-{chosen_class}")
    make_nested_dir(out_dir)
    point_coords = None
    point_labels = None
    
    for path, mask_path in tqdm(zip(data_path, mask_path), desc="Prediction...", total=len(data_path)):
        img = load_img(path)
        mask = load_file_npz(mask_path)
        
        # Only calculate 1
        if point_coords is None or point_labels is None:
            _, point_coords, point_labels = make_point_from_mask(mask, chosen_class)
        
        index_number = int(os.path.basename(path).replace(".jpg", "").split('_')[-1])

        predictor.set_image(img)
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
            )

        show_im = [
            img, 
            (mask == chosen_class).astype(np.float16),
            *[masks[idx].astype(np.float16) for idx in range(masks.shape[0])]
            ]
        save_fig_path = os.path.join(out_dir, f"fig-{index_number}.png")
        render_img(show_im=show_im, save_fig_path=save_fig_path, point_coords=point_coords, point_labels=point_labels)
        pass
    pass