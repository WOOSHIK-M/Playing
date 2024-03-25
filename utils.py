import numpy as np

from collections import namedtuple
from config import *

# type alias
GridLoc = namedtuple("GridLoc", "row, col")
RealXY = namedtuple("RealXY", "x, y")
ChargeInfo = namedtuple("ChargeInfo", "x, y, quantity")


def discrete_to_continuous(loc: GridLoc) -> RealXY:
    """Convert discrete location to coordinate on continuous."""
    return RealXY(
        x=(loc.col + 0.5) * BIN_WIDTH - FIELD_WIDTH / 2,
        y=(loc.row + 0.5) * BIN_HEIGHT - FIELD_HEIGHT / 2,
    )


def continuous_to_discrete(xy: RealXY) -> GridLoc:
    """Convert coordinate to discrete location."""
    row_idx = (xy.y + FIELD_HEIGHT / 2) / FIELD_HEIGHT * N_ROWS
    col_idx = (xy.x + FIELD_WIDTH / 2) / FIELD_WIDTH * N_COLS
    return GridLoc(row=int(row_idx), col=int(col_idx))
