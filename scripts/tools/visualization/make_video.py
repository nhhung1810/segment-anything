import os

import cv2
import numpy as np


SUBMISSION_GT_PATH = "runs/submission/gt/FLARE22_Tr_0008.nii.gz"
SUBMISSION_PRED_PATH = "runs/submission/pred/FLARE22_Tr_0008.nii.gz"


class VideoWriter:
    def __init__(self, video_info, saved_path):
        self.video_info = video_info
        self.saved_path = saved_path

        os.makedirs(self.saved_path, exist_ok=True)

        video_name = self.video_info["name"]
        out_path = os.path.join(self.saved_path, video_name)

        self.FPS = self.video_info["fps"]
        self.WIDTH = self.video_info["width"]
        self.HEIGHT = self.video_info["height"]
        self.NUM_FRAMES = self.video_info["num_frames"]
        self.outvid = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.FPS,
            (self.WIDTH, self.HEIGHT),
        )

    def write_frame(self, frame):
        self.outvid.write(frame)


def normalize_min_max(array):
    if array.max() == 0:
        return array

    norm_array = (array - array.min()) / array.max()
    return norm_array


def write_video(path, name):
    image = nib.load(path).get_fdata()

    frames = []
    for idx in range(image.shape[-1]):
        frame = normalize_min_max(image[:, :, idx])
        frame = (frame * 255).astype(np.uint8)
        frame = np.stack([frame, frame, frame], axis=2).astype(np.uint8)
        frames.append(frame)
        pass

    width, height, depth = image.shape
    writer = VideoWriter(
        {
            "name": name,
            "fps": 10,
            "width": width,
            "height": height,
            "num_frames": depth,
        },
        "runs/submission/visualize/",
    )

    for frame in frames:
        writer.write_frame((frame).astype(np.uint8))
        pass

    writer.outvid.release()
    writer.outvid = None

    return image.shape


if __name__ == "__main__":
    import nibabel as nib

    gt_path = SUBMISSION_GT_PATH
    pred_path = SUBMISSION_PRED_PATH
    filename = os.path.splitext(os.path.basename(gt_path))[0]
    gt_shape = write_video(gt_path, name=f"{filename}-gt.mp4")
    pred_shape = write_video(pred_path, name=f"{filename}-pred.mp4")
    assert gt_shape == pred_shape

    pass
