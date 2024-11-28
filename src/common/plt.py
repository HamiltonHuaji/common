from common.imports import *

@dataclass(kw_only=True)
class PlotImage:
    image: torch.Tensor | np.ndarray
    title: str | None = None
    cmap: str = 'viridis'
    value_range: Tuple[float, float] = (0, 1) # (0, 1): will be plotted as is

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    def from_tensor(cls, image: torch.Tensor | np.ndarray, title: str | None = None, cmap: str = 'viridis', value_range: Tuple[float, float] = (0, 1)):
        if isinstance(image, torch.Tensor):
            if ("bgr" in image.names):
                image = image.align_to(..., "height", "width", "bgr").rename(None)
                if image.size(-1) >= 3:
                    image = image[..., [2, 1, 0]]
            elif ("rgb" in image.names):
                image = image.align_to(..., "height", "width", "rgb").rename(None)
        return cls(image=image, title=title, cmap=cmap, value_range=value_range)

def plot_images(images, max_cols=4, axis_size=(4, 3)):
    images = [
        PlotImage.from_dict(image)
        if isinstance(image, dict)
        else (
            PlotImage.from_tensor(image)
            if isinstance(image, (torch.Tensor, np.ndarray))
            else image
        )
        for image in images
    ]

    n = len(images)
    n_rows = int(np.sqrt(n))
    n_cols = (n - 1) // n_rows + 1
    assert n_rows * n_cols >= n
    if n_cols > max_cols:
        n_cols = max_cols
        n_rows = (n - 1) // n_cols + 1

    per_axis_size_x, per_axis_size_y = axis_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * per_axis_size_x, n_rows * per_axis_size_y), layout='constrained')
    axes = axes.flatten() if n > 1 else [axes]

    for plot_image, ax in zip(images, axes):
        image = plot_image.image.detach().clone().cpu().numpy() if isinstance(plot_image.image, torch.Tensor) else plot_image.image
        image = image.squeeze()

        if image.dtype != np.float32 and image.dtype != np.int32:
            image = image.astype(np.float32)

        # deal with H,W,3 and 3,H,W
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        # deal with value_range
        if plot_image.value_range != (0, 1):
            image = (image - plot_image.value_range[0]) / (plot_image.value_range[1] - plot_image.value_range[0])

        # if 1-channel image, show per axis colorbar
        if image.ndim == 2:
            f = ax.imshow(image, cmap=plot_image.cmap)
            plt.colorbar(f, ax=ax)
        else:
            image = np.clip(image, 0, 1)
            ax.imshow(image)

        if plot_image.title:
            ax.set_title(plot_image.title)
    # plt.tight_layout()
