from argparse import ArgumentParser
from glob import glob
import os
import subprocess
from tqdm import tqdm
from scripts.datasets.constant import FLARE22_LABEL_ENUM, IMAGE_TYPE, TEST_NON_PROCESSED
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.render.data_class import ImageData, MaskData, OneImageRenderData
from scripts.render.render_engine import RenderEngine
from scripts.utils import make_directory


def visualize(image_file: str, gt_file: str, save_dir: str):
    p = FLARE22_Preprocess()
    r = RenderEngine()
    volume, masks = p.run_with_config(
        image_file=image_file,
        gt_file=gt_file,
        config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
    )
    for idx in tqdm(range(masks.shape[0]), desc="Rendering..."):
        # mask = masks[idx]
        (
            r.add(
                OneImageRenderData(
                    image=ImageData(volume[idx]).format(),
                    mask=MaskData(
                        mask=masks[idx],
                        legend_dict={
                            v.value: v.name.replace(" ", "_").lower()
                            for v in FLARE22_LABEL_ENUM
                        },
                    ),
                )
            )
            .show(save_path=f"{save_dir}/{idx:0>4}.png")
            .reset()
        )
        pass
    pass


parser = ArgumentParser("Run visualizing...")
parser.add_argument("-img", "--image_file", type=str, default="auto")
parser.add_argument(
    "-gt",
    "--gt_file",
    type=str,
    default=None,
    help="Ground truth or prediction label that follow the GT format",
)
parser.add_argument("-o", "--output_dir", type=str, default="auto")
parser.add_argument("-vid", "--make_vid", type=bool, default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    TEST_IMAGE_DIR = os.path.join(TEST_NON_PROCESSED, "images")
    save_dir = "./runs/visualize/test_folder/"
    gt_file: str = args.gt_file
    assert gt_file is not None and os.path.exists(gt_file), ""

    # Auto imply the source image file here
    image_file = args.image_file
    if image_file == "auto":
        gt_symbol: str = os.path.basename(gt_file).replace(".nii.gz", "")
        image_file = list(
            filter(lambda x: gt_symbol in x, list(glob(f"{TEST_IMAGE_DIR}/*.nii.gz")))
        )[0]
        pass
    assert os.path.exists(image_file), f"{image_file}"

    # Auto imply the output dir at the
    output_dir = args.output_dir
    if output_dir == "auto":
        output_dir = os.path.join(os.path.dirname(gt_file), "visual")
        pass

    is_make_video = args.make_vid

    print(
        f"""
        image-file   : {image_file}
        label-file   : {gt_file}
        output-dir   : {output_dir}
        """
    )

    make_directory(output_dir)
    visualize(
        image_file=image_file,
        gt_file=gt_file,
        save_dir=output_dir,
    )
    print(output_dir)
    if is_make_video:
        cmd = f"""ffmpeg -y -framerate 2 -pattern_type glob -i "{output_dir}/*.png" -c:v libx264 -pix_fmt yuv420p "{output_dir}/out.mp4" """
        subprocess.call(
            [cmd],
            shell=True,
        )
        print(f"{output_dir}/out.mp4")

        pass
    pass
