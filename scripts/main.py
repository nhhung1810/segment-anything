import torch
import os
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

from utils import make_nested_dir, get_data_paths, GROUP1, GROUP3, load_file_npz, load_img, argmax_dist


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


def show_mask(mask, ax, directly=False):
    if directly:
        ax.imshow(mask)
        return
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def render_img(show_im, save_fig_path, point_coords: np.ndarray,
               point_labels: np.ndarray):
    if point_coords is None:
        point_coords = np.array([])
        pass
    if point_labels is None:
        point_labels = np.array([])
        pass
    # Convention: show_im: [img, gt, mask1, mask2, ...]
    n_img = len(show_im)
    if n_img > 3:
        n_row = int(np.floor(np.sqrt(n_img)))
        n_row, n_col = n_row, n_row + 1
    else:
        n_row, n_col = 1, 3

    f, axes = plt.subplots(n_row, n_col, squeeze=False)
    for idx_1 in range(n_row):
        for idx_2 in range(n_col):
            idx = idx_1 * n_col + idx_2
            if idx >= n_img: break

            if idx != 0:
                # axes[idx_1, idx_2].imshow(show_im[0])
                show_mask(show_im[idx], axes[idx_1, idx_2], directly=True)
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
                add_text = "with point" if point_coords.shape[0] > 0 else ""
                axes[idx_1, idx_2].set_title(f'Image {add_text}')
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


def point_experiment():
    data_path, mask_path = get_data_paths(GROUP3)
    chosen_class = 1
    predictor = load_model()

    prefix = os.path.dirname(data_path[0])
    out_dir = os.path.join(f"{prefix}-fig-class-{chosen_class}")
    make_nested_dir(out_dir)
    point_coords = None
    point_labels = None

    for path, mask_path in tqdm(zip(data_path, mask_path),
                                desc="Prediction...",
                                total=len(data_path)):
        img = load_img(path)
        mask = load_file_npz(mask_path)

        # Only calculate 1
        if point_coords is None or point_labels is None:
            _, point_coords, point_labels = make_point_from_mask(
                mask, chosen_class)

        index_number = int(
            os.path.basename(path).replace(".jpg", "").split('_')[-1])

        predictor.set_image(img)
        masks, _, _ = predictor.predict(point_coords=point_coords,
                                        point_labels=point_labels,
                                        multimask_output=True)

        show_im = [
            img, (mask == chosen_class).astype(np.float16),
            *[masks[idx].astype(np.float16) for idx in range(masks.shape[0])]
        ]
        save_fig_path = os.path.join(out_dir, f"fig-{index_number}.png")
        render_img(show_im=show_im,
                   save_fig_path=save_fig_path,
                   point_coords=point_coords,
                   point_labels=point_labels)
        pass


def mask_experiment():
    data_path, mask_path = get_data_paths(GROUP1)
    chosen_class = 1
    predictor = load_model()
    # mask_input_size = [4*x for x in predictor.model.prompt_encoder.image_embedding_size]
    # pseudo_mask = torch.randn(1, *mask_input_size, dtype=torch.float32)
    prefix = os.path.dirname(data_path[0])
    out_dir = os.path.join(f"{prefix}-video-segment")
    make_nested_dir(out_dir)

    point, label = np.array([[400, 300]]), np.array([1.])
    masks = None

    for path, mask_path in tqdm(zip(data_path, mask_path),
                                desc="Prediction...",
                                total=len(data_path)):
        img = load_img(path)
        mask = load_file_npz(mask_path)
        _mask = None

        index_number = int(
            os.path.basename(path).replace(".jpg", "").split('_')[-1])
        

        predictor.set_image(img)
        _mask = T.Resize(size = (256 , 256))(torch.Tensor(masks[2])[None, ...]) \
            if masks is not None else None
        if _mask is not None:
            print(_mask.shape)
            pass
        masks, _, _ = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True,
            mask_input=_mask,
            return_logits=False,
        )

        show_im = [
            img, (mask == chosen_class).astype(np.float16),
            *[masks[idx].astype(np.float16) for idx in range(masks.shape[0])],
        ]

        if _mask is not None: show_im.append(np.array(_mask[0]))
        save_fig_path = os.path.join(out_dir, f"fig-{index_number}.png")
        render_img(show_im=show_im,
                   save_fig_path=save_fig_path,
                   point_coords=point,
                   point_labels=label
        )
        
        # NOTE: After the first run -> point will be off
        # point = None
        # label = None


if __name__ == "__main__":
    # point_experiment()
    mask_experiment()
    pass