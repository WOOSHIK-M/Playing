from electric_field import Atom
from utils import GridLoc, ChargeInfo


# N_ROWS = 11
ONE_ATOM = [
    Atom(
        name="atom_1", 
        loc=GridLoc(row=5, col=5), 
        width=10, 
        height=10,
        charge_info=[ChargeInfo(x=0, y=0, quantity=1)],
    )
]

# N_ROWS = 51
TWO_ATOMS = [
    Atom(
        name="atom_1", 
        loc=GridLoc(row=25, col=20), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
    ),
    Atom(
        name="atom_2", 
        loc=GridLoc(row=25, col=30), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
    )
]

# N_ROWS = 51
FOUR_ATOMS = [
    Atom(
        name="atom_1", 
        loc=GridLoc(row=20, col=25), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
    ),
    Atom(
        name="atom_2", 
        loc=GridLoc(row=25, col=20), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
    ),
    Atom(
        name="atom_3", 
        loc=GridLoc(row=30, col=25), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
    ),
    Atom(
        name="atom_4", 
        loc=GridLoc(row=25, col=30), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
    )
]

# N_ROWS = 51
SIX_ATOMS = [
    Atom(
        name="atom_1", 
        loc=GridLoc(row=18, col=20), 
        width=15, 
        height=15,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
        group_name="group_1",
    ),
    Atom(
        name="atom_2", 
        loc=GridLoc(row=25, col=20), 
        width=15, 
        height=15,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
        group_name="group_1",
    ),
    Atom(
        name="atom_3", 
        loc=GridLoc(row=32, col=20), 
        width=15, 
        height=15,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
        group_name="group_1",
    ),
    Atom(
        name="atom_4", 
        loc=GridLoc(row=18, col=30), 
        width=15, 
        height=15,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
        group_name="group_2",
    ),
    Atom(
        name="atom_5", 
        loc=GridLoc(row=25, col=30), 
        width=15, 
        height=15,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
        group_name="group_2",
    ),
    Atom(
        name="atom_6", 
        loc=GridLoc(row=32, col=30), 
        width=15, 
        height=15,
        charge_info=[
            ChargeInfo(x=3, y=3, quantity=1),
            ChargeInfo(x=-3, y=3, quantity=1),
            ChargeInfo(x=3, y=-3, quantity=1),
            ChargeInfo(x=-3, y=-3, quantity=1),
        ],
        group_name="group_2",
    ),
]


# N_ROWS = 51
TEM_ATOMS = [
    Atom(
        name="atom_1", 
        loc=GridLoc(row=11, col=22), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
    Atom(
        name="atom_2", 
        loc=GridLoc(row=18, col=22), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
    Atom(
        name="atom_3", 
        loc=GridLoc(row=25, col=22), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
    Atom(
        name="atom_4", 
        loc=GridLoc(row=32, col=22), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
    Atom(
        name="atom_5", 
        loc=GridLoc(row=39, col=22), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
    Atom(
        name="atom_6", 
        loc=GridLoc(row=11, col=28), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
    Atom(
        name="atom_7", 
        loc=GridLoc(row=18, col=28), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
    Atom(
        name="atom_8", 
        loc=GridLoc(row=25, col=28), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
    Atom(
        name="atom_9", 
        loc=GridLoc(row=32, col=28), 
        width=10, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
    Atom(
        name="atom_10", 
        loc=GridLoc(row=10, col=5), 
        width=40, 
        height=10,
        charge_info=[
            ChargeInfo(x=-3, y=0, quantity=1),
            ChargeInfo(x=3, y=0, quantity=1),
        ],
    ),
]


MIXED_SIZE_ATOMS = [
    Atom(
        name="Big", 
        loc=GridLoc(row=15, col=22), 
        width=60, 
        height=30,
        charge_info=[
            ChargeInfo(x=-15, y=0, quantity=1),
            ChargeInfo(x=15, y=0, quantity=1),
        ],
    ),
    Atom(
        name="Middle_1", 
        loc=GridLoc(row=40, col=10), 
        width=12, 
        height=15,
        charge_info=[
            ChargeInfo(x=-4, y=0, quantity=1),
            ChargeInfo(x=4, y=0, quantity=1),
        ],
    ),
    Atom(
        name="Middle_2", 
        loc=GridLoc(row=30, col=10), 
        width=15, 
        height=12,
        charge_info=[
            ChargeInfo(x=-6, y=0, quantity=1),
            ChargeInfo(x=6, y=0, quantity=1),
        ],
    ),
    Atom(
        name="Small_1", 
        loc=GridLoc(row=40, col=20), 
        width=5, 
        height=5,
        charge_info=[
            ChargeInfo(x=-1, y=0, quantity=1),
            ChargeInfo(x=1, y=0, quantity=1),
        ],
    ),
    Atom(
        name="Small_2", 
        loc=GridLoc(row=35, col=20), 
        width=5, 
        height=5,
        charge_info=[
            ChargeInfo(x=-1, y=0, quantity=1),
            ChargeInfo(x=1, y=0, quantity=1),
        ],
    ),
    Atom(
        name="Small_3", 
        loc=GridLoc(row=30, col=20), 
        width=10, 
        height=2,
        charge_info=[
            ChargeInfo(x=-4, y=0, quantity=1),
            ChargeInfo(x=4, y=0, quantity=1),
        ],
    ),
    Atom(
        name="Small_4", 
        loc=GridLoc(row=25, col=20), 
        width=6, 
        height=7,
        charge_info=[
            ChargeInfo(x=-2, y=0, quantity=1),
            ChargeInfo(x=2, y=0, quantity=1),
        ],
    ),
]


import pandas as pd

from utils import *
from config import *

def read_csv(fpath: str) -> list[Atom]:
    """Parse atom info from csv."""
    df = pd.read_csv(fpath)

    atoms = []
    for row in df.iloc:
        if row["PlacedLayer"] != "TOP":
            continue

        center_x = row["left"] + row["width"] / 2 - FIELD_WIDTH / 2
        center_y = row["top"] + row["height"] / 2 - FIELD_HEIGHT / 2
        charge_info = [
            ChargeInfo(
                x - FIELD_WIDTH / 2 - center_x, 
                y - FIELD_HEIGHT / 2 - center_y, 
                1.0,
            ) 
            for x, y in map(eval, eval(row["pins"]))
        ]
        atom = Atom(
            name=row["Name"],
            loc=continuous_to_discrete(xy=RealXY(x=center_x, y=center_y)),
            width=row["width"],
            height=row["height"],
            is_fixed=row["Fixed"],
            charge_info=charge_info,
            # group_name=row["sheet_number"],
            group_name=row["community_id"],
        )
        atoms.append(atom)
    return atoms


EXPERT_ATOMS = read_csv("data/expert.csv")
