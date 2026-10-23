"""Table layout of coronal slices: one panel per slice, shared registered frame and scale."""
import math
import numpy as np
from scipy.spatial import cKDTree

def grid_columns(n):
    """Rows of four up to 12 slices (3 x 4), then rows of eight (2 x 8 for 16)."""
    return max(1, min(n, 4 if n <= 12 else 8))


def slice_grid(cells, slices, columns=None, pad=0.04, title=None):
    """Return per-cell display coordinates, panels and the overall grid size.

    Every panel spans the union bounding box of the displayed slices in registered
    (rx, ry) units, so slices keep their registered position and relative size.
    Display y increases downward, matching the registered orientation (dorsal up).
    """
    columns = grid_columns(len(slices)) if columns is None else max(1, min(columns, len(slices)))
    rows = math.ceil(len(slices)/columns)
    # Labels are a fixed pixel size, so smaller panels need a taller title band.
    title = max(0.08, 0.02*columns) if title is None else title
    order = {s: i for i, s in enumerate(slices)}
    x0, x1 = float(cells.rx.min()), float(cells.rx.max())
    y0, y1 = float(cells.ry.min()), float(cells.ry.max())
    margin = pad*max(x1-x0, y1-y0)
    header = title*(y1-y0)
    cell_w, cell_h = (x1-x0) + 2*margin, (y1-y0) + header + 2*margin
    index = cells.slice.map(order).to_numpy()
    col, row = index % columns, index // columns
    px = col*cell_w + margin + (cells.rx.to_numpy()-x0)
    py = row*cell_h + header + margin + (cells.ry.to_numpy()-y0)
    counts = cells.slice.value_counts()
    panels = [dict(label=s, n=int(counts.get(s, 0)), x=float((i % columns)*cell_w), y=float((i//columns)*cell_h),
                   w=float(cell_w), h=float(cell_h), header=float(header)) for i, s in enumerate(slices)]
    return px, py, panels, float(columns*cell_w), float(rows*cell_h)


def dot_size(cells, sample=20000, seed=0):
    """Median nearest-neighbour spacing within slices, in registered units."""
    distances = []
    rng = np.random.default_rng(seed)
    for _, part in cells.groupby('slice', observed=True):
        points = part[['rx', 'ry']].to_numpy()
        if len(points) < 2:
            continue
        query = points[rng.choice(len(points), min(len(points), sample), replace=False)]
        distances.append(cKDTree(points).query(query, k=2)[0][:, 1])
    return float(np.median(np.concatenate(distances))) if distances else 20.0


def quantize(px, py, width, height):
    """Unsigned 16-bit display positions and the unit step that restores them."""
    step = max(1.0, math.ceil(max(width, height)/65535))
    return (np.round(px/step).astype(np.uint16), np.round(py/step).astype(np.uint16), step)
