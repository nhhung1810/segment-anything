import itertools
from typing import Dict, List, Tuple
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scripts.datasets.color import get_color


class Renderer:
    def __init__(self, legend_dict: Dict[int, object]) -> None:
        self.legend_dict = legend_dict
        self.data = []
        pass

    def build_class_name_map(self, masks: np.ndarray):
        result = {}
        for class_num in np.unique(masks):
            try:
                label = self.legend_dict[class_num]
            except Exception:
                label = f"{class_num}"
            result[class_num] = label

        return result

    def overlay_mask(self, mask, ax: Axes, no_overlay=False):
        if no_overlay:
            ax.imshow(mask)
            return

        h, w = mask.shape[-2:]
        masks = mask.reshape(h, w)
        result = np.ones([h, w, 4])

        if self.legend_dict is not None:
            handle_legend = []
        class_label_map = self.build_class_name_map(masks)
        for class_num in np.unique(masks):
            (r, b, g, _) = get_color(class_num) if class_num != 0 else (0, 0, 0, 0)
            color = (r, b, g, 0.6)
            if self.legend_dict is not None:
                handle_legend.append(
                    Line2D(
                        xdata=[0],
                        ydata=[0],
                        color=color,
                        lw=2,
                        label=f"{class_label_map[class_num]}",
                    )
                )
            result[masks == class_num] = color
            pass

        if self.legend_dict is not None:
            ax.legend(handles=handle_legend, bbox_to_anchor=(1.5, 1), loc="upper right")
            # ax.legend(handles=handle_legend, bbox_to_anchor=(3, 1), loc="upper right")
        ax.imshow(result)
        pass

    def add_multiple(self, arrays: List[Dict]):
        for data in arrays:
            self.add(
                img=data.get("img", None),
                mask=data.get("mask", None),
                points=data.get("points", None),
                title=data.get("title", None),
            )
            pass
        pass

    def add(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        points: Tuple[np.ndarray, np.ndarray] = None,
        bbox: Tuple[np.ndarray, str] = None,
        title: str = "",
    ):
        assert img is not None or mask is not None, "Both img or mask is None"

        _img = self._format_img(img)
        _mask = self._format_mask(mask)
        _bbox = self._format_bbox(bbox)

        self.data.append(
            {
                "img": _img,
                "mask": _mask,
                "points": points,
                "title": title or "Untitled",
                "bbox": _bbox,
            }
        )
        return self

    def _format_mask(self, mask):
        if mask is None:
            return None

        assert mask.ndim <= 3, f"Mask dim <= 3, while get {mask.shape}"
        # Reshape mask to dim of 2
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                _mask = mask[0]
            elif mask.shape[-1] == 1:
                _mask = mask[:, :, 0]
        else:
            _mask = mask.copy()

        return _mask

    def _format_img(self, img):
        if img is None:
            return None

        # Check for img data
        assert img.ndim < 4, "Out of control"
        _img = img.copy()
        if img.ndim == 2:
            _img = _img[:, :, None]

        if img.ndim == 3:
            assert (
                _img.shape[-1] == 1 or _img.shape[-1] == 3
            ), f"Invalid shape {img.shape} transform to {_img.shape}"
            pass
        return _img

    def _format_bbox(self, bbox):
        return bbox

    def show_all(self, save_path: str = None):
        assert len(self.data) > 0, "There is no data to be rendered"
        f, ax, n_row, n_col = self._get_valid_subplot(len(self.data))
        for i1, i2 in itertools.product(range(n_row), range(n_col)):
            # for i1 in range(n_row):
            #     for i2 in range(n_col):
            idx = i1 * n_col + i2
            if idx >= len(self.data):
                break

            _data = self.data[idx]
            if _data["img"] is not None:
                ax[i1, i2].imshow(_data["img"])
                pass

            if _data["mask"] is not None:
                self.overlay_mask(
                    _data["mask"], ax[i1, i2], no_overlay=_data["img"] is None
                )
                pass

            if _data["points"] is not None:
                [pcoors, plabels] = _data["points"]
                self._render_point_label(
                    ax[i1, i2], pcoors, plabels, chosen_value=1, color_str="g"
                )
                self._render_point_label(
                    ax[i1, i2], pcoors, plabels, chosen_value=0, color_str="r"
                )
                pass

            if _data["bbox"] is not None:
                bbox = _data["bbox"][0]
                bbox_label = _data["bbox"][1]
                rect = patches.Rectangle(
                    (bbox["x"], bbox["y"]),
                    bbox["w"],
                    bbox["h"],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax[i1, i2].add_patch(rect)
                pass

            ax[i1, i2].set_title(_data["title"])
            pass
        pass

        if save_path:
            f.savefig(
                save_path,
                # bbox_inches="tight"
            )
            plt.close()
            return self
        else:
            f.show()

        return self

    def _render_point_label(self, ax, pcoors, plabels, chosen_value=1, color_str="g"):
        if pcoors is None:
            return
        if plabels is None:
            return
        msk = plabels == chosen_value
        if msk.any():
            xs = pcoors[msk][:, 0]
            ys = pcoors[msk][:, 1]
            ax.scatter(xs, ys, c=color_str)

    def _get_valid_subplot(self, n):
        if n == 1:
            f, axes = plt.subplots(1, 1, squeeze=False)
            return f, axes, 1, 1

        if n <= 3:
            n_row, n_col = 1, n
        else:
            n_row = int(np.floor(np.sqrt(n)))
            n_row, n_col = n_row, n_row + 1

        f, axes = plt.subplots(n_row, n_col, squeeze=False)

        return f, axes, n_row, n_col

    def reset(self):
        self.data.clear()
        plt.close()
