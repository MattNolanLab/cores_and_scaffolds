from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.transforms import BboxBase, BboxTransformTo, blended_transform_factory
from numpy.typing import NDArray

DEFAULT_FIG_W: float = 3.3
DEFAULT_FIG_H: float = 2.5


A4: NDArray[float] = np.array([8.3, 11.7])
A4_L: NDArray[float] = np.array([11.7, 8.3])
A5: NDArray[float] = np.array([5.8, 8.3])
A5_L: NDArray[float] = np.array([8.3, 5.8])
SLIDE: NDArray[float] = np.array([13.3, 7.5])


def asp(g, *args, **kwargs):
    return plt.gcf().add_subplot(g, *args, **kwargs)


class PLASTICITY_COLOURS:
    all: str = 'grey'
    recurrent: str = '#347979'
    input: str = '#E77979'
    output: str = '#F4D03F'


def set_font_size(size: float):
    font_params = {
        'font.size': size,
        'axes.titlesize': size,
        'axes.labelsize': size,
        'xtick.labelsize': size,
        'ytick.labelsize': size,
        'legend.fontsize': size,
        'figure.titlesize': size,
        'figure.labelsize': size,
    }
    mpl.rcParams.update(font_params)


class PanelFigure:
    """OOP GridSpec-like object
    Used in older figures and for exploration.

    TODO: deprecate this
    """
    def __init__(
        self,
        cols: int = 1,
        rows: int = 1,
        w: float = DEFAULT_FIG_W,
        h: float = DEFAULT_FIG_H,
        dpi: int = 300,
        enable_share_axes: bool = False,
    ):
        self.cols: int = cols
        self.rows: int = rows

        self.n: int = self.cols * self.rows

        plt.figure(figsize=(w * self.cols, h * self.rows), dpi=dpi)

        self.index: int = 0

        self.ylims: List[Tuple[float, float]] = []
        self.xlims: List[Tuple[float, float]] = []

        self.enable_share_axes: bool = enable_share_axes

    def store_lims(self):
        self.ylims.append(plt.gca().get_ylim())
        self.xlims.append(plt.gca().get_xlim())

    def __call__(self, sharex=None, sharey=None):
        self.index += 1
        if self.index > 0:
            self.store_lims()
        plt.subplot(self.rows, self.cols, self.index, sharex=sharex, sharey=sharey)

        # axis handling
        if self.enable_share_axes:
            if not self.is_left():
                plt.ylabel('')
                plt.yticks([])
            if not self.is_bottom():
                plt.xlabel('')
                plt.xticks([])

            if self.index == self.n:
                self.share_axes()

    @property
    def row(self) -> int:
        return (self.index - 1) // self.cols

    @property
    def col(self) -> int:
        return (self.index - 1) % self.cols

    def share_axes(self):
        self.store_lims()  # Assumes that this will be called at the end
        min_ylim, max_ylim = (min([lim[0] for lim in self.ylims]), max([lim[1] for lim in self.ylims]))
        min_xlim, max_xlim = (min([lim[0] for lim in self.xlims]), max([lim[1] for lim in self.xlims]))

        for i in range(1, self.n + 1):
            plt.subplot(self.rows, self.cols, i)
            plt.ylim(min_ylim, max_ylim)
            plt.xlim(min_xlim, max_xlim)

    def is_left(self):
        return self.col == 0

    def is_bottom(self):
        return self.row == self.rows - 1

    def is_top(self):
        return self.row == 0


def make_paper_theme() -> None:
    plt.style.use('default')
    plt.style.use('ieee')

    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color',
        ['black', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    )

    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    plt.rcParams['legend.fontsize'] = 15

    plt.rcParams['font.family'] = 'Arial'


def make_panel_fig(cols: int = 1, rows: int = 1) -> None:
    plt.figure(figsize=(DEFAULT_FIG_W * cols, DEFAULT_FIG_H * rows))


def make_exploration_theme() -> None:
    plt.style.use('default')
    mpl.rcParams['figure.dpi'] = 100


def strip_plot_axes():
    ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    if hasattr(ax, 'zaxis'):
        ax.set_zticks([])
        ax.set_zlabel('')


def set_xaxis_engineering_labels():
    plt.gca().xaxis.set_major_formatter(ticker.EngFormatter())


def set_yaxis_engineering_labels():
    plt.gca().yaxis.set_major_formatter(ticker.EngFormatter())


def set_xaxis_n_ticks(n: int):
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(n))


def set_yaxis_n_ticks(n: int):
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(n))


def hide_y():
    plt.ylabel('')
    plt.gca().tick_params(axis='y', labelleft=False)


def hide_x():
    plt.ylabel('')
    plt.gca().tick_params(axis='x', labelbottom=False)


def plot_track_vertical_base(rz_min=60 + 30, rz_max=80 + 30):
    ax = plt.gca()
    ax.axhspan(rz_min, rz_max, alpha=0.15, color='green')
    ax.axhspan(0, 30, alpha=0.15, color='black')
    blackbox2_end: int = 60 + 60 + 20 + 30 + 30
    ax.axhspan(60 + 60 + 20 + 30, blackbox2_end, alpha=0.15, color='black')


def make_figure(w: float = 1.0, h: float = 1.0):
    return plt.figure(figsize=(3.3 * w, 2.5 * h))


def despine_topright():
    [plt.gca().spines[k].set_visible(False) for k in ['top', 'right']]


def despine_all():
    [plt.gca().spines[k].set_visible(False) for k in ['top', 'right', 'bottom', 'left']]


def set_xticks_track():
    plt.xticks(_get_track_boundaries(), _get_track_boundaries())


def set_yticks_track():
    plt.yticks(_get_track_boundaries(), _get_track_boundaries())


def _get_track_boundaries() -> np.ndarray:
    """Returns relevant track boundaries.

    E.g. plt.xticks(get_track_boundaries(), get_track_boundaries())
    """
    return np.array([0, 100, 200])


class AxesBbox(BboxBase):
    def __init__(self, axes):
        super().__init__()
        self._points = None
        self._bboxes = [ax.bbox for ax in axes]
        self.set_children(*self._bboxes)

    def get_points(self):
        if self._invalid:
            self._points = self.union(self._bboxes).get_points()
            self._invalid = 0
        return self._points


def suptitle(*args, **kwargs):
    fig = plt.gcf()
    label = fig.suptitle(*args, **kwargs)
    trans = blended_transform_factory(BboxTransformTo(AxesBbox(fig.axes)), label.get_transform())
    label.set_transform(trans)
    return label


def supxlabel(*args, **kwargs):
    fig = plt.gcf()
    label = fig.supxlabel(*args, **kwargs)
    trans = blended_transform_factory(BboxTransformTo(AxesBbox(fig.axes)), label.get_transform())
    label.set_transform(trans)
    return label


def supylabel(*args, **kwargs):
    fig = plt.gcf()
    label = fig.supylabel(*args, **kwargs)
    trans = blended_transform_factory(label.get_transform(), BboxTransformTo(AxesBbox(fig.axes)))
    label.set_transform(trans)
    return label


def get_midpoint_x() -> float:
    fig = plt.gcf()
    return (fig.subplotpars.right + fig.subplotpars.left) / 2


def get_midpoint_y() -> float:
    fig = plt.gcf()
    return (fig.subplotpars.top + fig.subplotpars.bottom) / 2


def tight_layout(*args, left: float = 0.0, bottom: float = 0.0, right: float = 1.0, top: float = 1.0, **kwargs):
    plt.tight_layout(*args, rect=[left, bottom, right, top], **kwargs)


def set_tick_decimals(decimals: int, axis: str = 'both') -> None:
    fmt = ticker.FormatStrFormatter(f'%.{decimals}f')
    if axis in ('x', 'both'):
        plt.gca().xaxis.set_major_formatter(fmt)
    if axis in ('y', 'both'):
        plt.gca().yaxis.set_major_formatter(fmt)


def set_y_tick_decimals(decimals: int) -> None:
    set_tick_decimals(decimals, axis='y')


def set_x_tick_decimals(decimals: int) -> None:
    set_tick_decimals(decimals, axis='x')


def y_displacement(control_point: float, percent_ticks: list[float] = [-25, 0, 25]):
    tick_locations = [((v * 2) + control_point) for v in percent_ticks]
    tick_labels = [f'{p:g}' for p in percent_ticks]
    plt.yticks(tick_locations, tick_labels)


class GridSpec(plt.GridSpec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_last_row(self, i: int):
        return i == (self.nrows - 1)

    def is_last_col(self, i: int):
        return i == (self.ncols - 1)

    def is_first_row(self, i: int):
        return i == 0

    def is_first_col(self, i: int):
        return i == 0
