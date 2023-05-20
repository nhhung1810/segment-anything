from scripts.datasets.color import get_color
from scripts.render.data_class import (
    BBoxData,
    MaskData,
    MultiMasksData,
    OneImageRenderData,
    PointData,
)
from scripts.utils import make_directory


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


from typing import List


class RenderEngine:
    def __init__(self) -> None:
        self.buffer: List[OneImageRenderData] = []
        pass

    def add(self, data: OneImageRenderData):
        self.buffer.append(data)
        return self

    def reset(self):
        self.buffer.clear()
        return self

    def overlay_mask(self, mask, ax: Axes, no_overlay=False, legend_dict: dict = None):
        # Implicitly handle a multi-class mask
        if no_overlay:
            ax.imshow(mask)
            return

        h, w = mask.shape[-2:]
        masks = mask.reshape(h, w)
        result = np.ones([h, w, 4])

        if legend_dict is not None:
            handle_legend = []
        for class_num in np.unique(masks):
            (r, b, g, _) = get_color(class_num) if class_num != 0 else (0, 0, 0, 0)
            color = (r, b, g, 0.6)
            if legend_dict is not None:
                handle_legend.append(
                    Line2D(
                        xdata=[0],
                        ydata=[0],
                        color=color,
                        lw=2,
                        label=f"{legend_dict[class_num]}",
                    )
                )
            result[masks == class_num] = color
            pass

        if legend_dict is not None:
            ax.legend(handles=handle_legend, bbox_to_anchor=(1.5, 1), loc="upper right")
        ax.imshow(result)
        pass

    def make_valid_subplot(self, n):
        if n <= 3:
            n_row, n_col = 1, n
        else:
            n_row = int(np.floor(np.sqrt(n)))
            n_row, n_col = n_row, n_row + 1

        f, axes = plt.subplots(n_row, n_col, squeeze=False)
        return f, axes, n_row, n_col

    def render_overlay(self, mask: MaskData, ax: Axes):
        self.overlay_mask(mask.mask, ax=ax, legend_dict=mask.legend_dict)
        pass

    def render_multi_mask(self, masks: MultiMasksData, ax: Axes):
        # Treat each mask as a binary mask
        n_class, h, w = masks.masks.shape
        legends = masks.legend

        handle_legend = []
        for class_num in range(n_class):
            result = np.zeros([h, w, 4])
            color = get_color(class_num, alpha=0.6)
            if legends is not None:
                handle_legend.append(
                    Line2D(
                        xdata=[0],
                        ydata=[0],
                        color=color,
                        lw=2,
                        label=f"{legends[class_num]}",
                    )
                )
            # This will produce multiple overlay mask
            result[masks.masks[class_num] > 0.0] = color
            if legends is not None:
                ax.legend(
                    handles=handle_legend, bbox_to_anchor=(1.3, 1), loc="upper right"
                )
            ax.imshow(result)
            print()

        pass

    def render_points(self, points: List[PointData], ax: Axes):
        pass

    def render_bbox(self, bboxes: List[BBoxData], ax: Axes):
        pass

    def show_one_render_data(self, data: OneImageRenderData, ax: Axes):
        if data.image:
            ax.imshow(data.image.image)

        # We can only use this one at a time
        if data.mask and not data.multi_masks:
            self.render_overlay(data.mask, ax)
        elif data.multi_masks:
            self.render_multi_mask(data.multi_masks, ax)

        if data.points:
            self.render_points(data.points, ax)

        if data.bboxes:
            self.render_bbox(data.bboxes, ax)

        pass

    def show(self, save_path: str):
        _ = make_directory(save_path, is_file=True)
        assert len(self.buffer) > 0, "No data to be rendered"
        n_images = len(self.buffer)
        f, axes, n_row, n_col = self.make_valid_subplot(n_images)
        for i1 in range(n_row):
            for i2 in range(n_col):
                idx = i1 * n_col + i2
                data = self.buffer[idx]
                ax: Axes = axes[i1, i2]
                self.show_one_render_data(data, ax)

        if save_path:
            f.savefig(save_path)
            plt.close()
            return self
        else:
            f.show()
            return self

        return self
