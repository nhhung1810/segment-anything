import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def load_ct_info(file_path):
    sitk_image = sitk.ReadImage(file_path)
    if sitk_image is None:
        res = {}
    else:
        origin = sitk_image.GetOrigin()  # original used list(reversed(, dont know why
        spacing = sitk_image.GetSpacing()  # original used list(reversed(, dont know why
        direction = sitk_image.GetDirection()
        subdirection = [direction[8], direction[4], direction[0]]
        res = {
            "sitk_image": sitk_image,
            "npy_image": sitk.GetArrayFromImage(sitk_image),
            "origin": origin,
            "spacing": spacing,
            "direction": direction,
            "subdirection": subdirection,
        }
    return res


def change_axes_of_image(npy_image, orientation):
    """default orientation=[1, -1, -1]"""
    if orientation[0] < 0:
        npy_image = np.flip(npy_image, axis=0)
    if orientation[1] > 0:
        npy_image = np.flip(npy_image, axis=1)
    if orientation[2] > 0:
        npy_image = np.flip(npy_image, axis=2)
    return npy_image


def save_ct_from_sitk(sitk_image, save_path, sitk_type=None, use_compression=False):
    if sitk_type is not None:
        sitk_image = sitk.Cast(sitk_image, sitk_type)
    sitk.WriteImage(sitk_image, save_path, use_compression)


def save_ct_from_npy(
    npy_image,
    save_path,
    origin=None,
    spacing=None,
    direction=None,
    sitk_type=None,
    use_compression=False,
):
    sitk_image = sitk.GetImageFromArray(npy_image)
    if origin is not None:
        sitk_image.SetOrigin(origin)
    if spacing is not None:
        sitk_image.SetSpacing(spacing)
    if direction is not None:
        sitk_image.SetDirection(direction)
    save_ct_from_sitk(sitk_image, save_path, sitk_type, use_compression)

def postprocess(pred_file, gt_file, out_dir):
    # filenames = os.listdir(gt_file)
    os.makedirs(out_dir, exist_ok=True)

    print("Processing prediction files")
    # for test_filename in tqdm(filenames):
    raw_image_path = gt_file
    # pred_test_filename = gt_file.replace('.nii.gz', '.npy')
    pred_image_path = pred_file

    if not os.path.isfile(pred_image_path):
        return
    assert os.path.isfile(pred_image_path), f"Missing {pred_image_path}"

    raw_image_dict = load_ct_info(raw_image_path)
    pred_image_dict = {
        'mask': np.load(pred_image_path).transpose(2,0,1)
    }

    pred_image_dict["mask"] = change_axes_of_image(
        pred_image_dict["mask"], raw_image_dict["subdirection"]
    )

    out_test_filename = os.path.basename(gt_file).replace('_0000.nii.gz', '.nii.gz')
    dest_image_path = os.path.join(out_dir, out_test_filename)

    save_ct_from_npy(
        npy_image=pred_image_dict["mask"],
        save_path=dest_image_path,
        origin=raw_image_dict["origin"],
        spacing=raw_image_dict["spacing"],
        direction=raw_image_dict["direction"],
        sitk_type=sitk.sitkUInt8,
    )

if __name__ == "__main__":
    TEST_MASK_PATH = (
        "dataset/FLARE22-version1/FLARE22_LabeledCase50/labels/FLARE22_Tr_0008.nii.gz"
    )

    PRED_MASK_PATH = (
        "runs/sam-one-point-230501-012024/inference/FLARE22_Tr_0008.npy"
    )


    postprocess(pred_file=PRED_MASK_PATH, gt_file=TEST_MASK_PATH, out_dir=os.path.dirname(PRED_MASK_PATH))
    pass
