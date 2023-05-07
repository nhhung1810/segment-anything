from collections import defaultdict
import subprocess
import natsort

from tqdm import tqdm


from scripts.datasets.constant import VAL_METADATA, FLARE22_LABEL_ENUM
from scripts.datasets.flare22_loader import FileLoader
from scripts.render import Renderer
from scripts.utils import load_file_npz, load_img, make_directory, omit


PATH = "/dataset/FLARE22-version1"
VAL_MASK = f"{PATH}/ValMask"
VAL_IMAGE = f"{PATH}/ValImageProcessed"


if __name__ == "__main__":
    render = Renderer(
        legend_dict={
            value.value: value.name.lower().replace("_", " ")
            for value in FLARE22_LABEL_ENUM
        }
    )
    file_loader = FileLoader(metadata_path=VAL_METADATA)
    patient_data = defaultdict(list)

    SKIP_SAVE_FIG = False
    FRAMERATE = 10

    for data in file_loader.data:
        patient_data[data["name"]].append(omit(data, ["name"]))
        pass

    for pname, pvalues in tqdm(patient_data.items(), desc="Visualizing..."):
        _dir = make_directory(f"./runs/visualize/{pname}")
        for ct_slice in tqdm(
            pvalues, total=len(pvalues), leave=False, desc="Generating mask overlay..."
        ):
            if not SKIP_SAVE_FIG:
                idx = ct_slice["id_number"]
                img = load_img(ct_slice["img_path"])
                mask = load_file_npz(ct_slice["mask_path"])
                render.add(img=img, mask=mask, title=f"{pname}-{idx}").show_all(
                    save_path=f"{_dir}/{idx:0>4}.png"
                ).reset()
                pass

        # After all save-fig -> run make video
        # Auto-overwrite is applied, please aware about this
        cmd = f"""\
        ffmpeg -y -framerate {FRAMERATE} -pattern_type glob -i "{_dir}/*.png" -c:v\
        libx264 -pix_fmt yuv420p "{_dir}/{pname}.mp4"
        """

        subprocess.run(
            [cmd],
            shell=True,
        )
        pass
    pass
