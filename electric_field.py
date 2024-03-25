import itertools
import math

from scipy.ndimage import gaussian_filter

from utils import *


class PositiveCharge:
    """A base positive charge class."""

    def __init__(
        self, 
        name: str, 
        xy: RealXY, 
        quantity: float,
        group_name: str,
    ) -> None:
        """Initialize."""
        self.name = name
        self.xy = xy
        self.quantity = quantity
        self.group_name = group_name
    
    @property
    def atom_name(self) -> str:
        """Get the name which belongs to."""
        return "_".join(self.name.split("_")[:-1])

    @staticmethod
    def compute_electric_potential_energy(
        xy: RealXY,
        charges: list["PositiveCharge"],
    ) -> float:
        """Compute electric potential at given location.
        
        Formula : V = kQ/r
        """
        v = 0.0
        for c in charges:
            r = math.sqrt((xy.x - c.xy.x) ** 2 + (xy.y - c.xy.y) ** 2)
            v += c.quantity / (r + 1.0)
        return v
    
    @staticmethod
    def compute_electrostatic_force(c1: "PositiveCharge", c2: "PositiveCharge") -> tuple[float, float]:
        """Compute electrostatic force between two charges.
        
        Formula: f = kq1q2 / r^2
        """
        if c1.group_name == c2.group_name:
            return 0.0, 0.0
        
        r = math.sqrt((c1.xy.x - c2.xy.x) ** 2 + (c1.xy.y - c2.xy.y) ** 2)
        
        # f = kq1q2 / r^2
        force = c1.quantity * c2.quantity / (r + 1.0) ** 2

        # compute force vectors
        theta = math.atan2(c1.xy.y - c2.xy.y, c1.xy.x - c2.xy.x)
        force_x = force * math.cos(theta)
        force_y = force * math.sin(theta)
        return np.round(force_x, 10), np.round(force_y, 10)
    
    def __repr__(self) -> str:
        """Display info of this."""
        return (
            f"(Charge Class) name: {self.name}, "
            f"loc: ({self.xy.x:.2f}, {self.xy.y:.2f}), "
            f"quantity: {self.quantity}"
        )


class Atom:
    """A base atom class with containing charges.
    
    It is represented by rectangle basically.
    """

    def __init__(
        self,
        name: str,
        loc: GridLoc,
        width: float,
        height: float,
        charge_info: list[ChargeInfo],
        group_name: str = "Group_Name",
        is_fixed: bool = False,
        is_boundary: bool = False,
    ) -> None:
        """Initialize."""
        self.name = name
        self.loc = loc
        self.width = width
        self.height = height
        self.group_name = group_name

        self.is_fixed = is_fixed
        self.is_boundary = is_boundary

        self.relative_xy = [(x, y) for x, y, _ in charge_info]
        self.charges = [
            PositiveCharge(
                name=f"{self.name}_{idx}",
                xy=RealXY(x=self.center.x + x, y=self.center.y + y),
                quantity=quantity * math.sqrt((self.width * self.height) / len(charge_info)) / 10,
                group_name=group_name,
            ) 
            for idx, (x, y, quantity) in enumerate(charge_info)
        ]

    @property
    def is_movable(self) -> bool:
        """Check if it is movable."""
        return not self.is_fixed and not self.is_boundary
    
    @property
    def center(self) -> RealXY:
        """Get a coordinate of center."""
        return discrete_to_continuous(loc=self.loc)
    
    @property
    def bottom_left(self) -> RealXY:
        """Get a coordinate of bottom left."""
        xy = self.center
        return RealXY(x=xy.x - self.width / 2, y=xy.y - self.height / 2)

    @property
    def top_right(self) -> RealXY:
        """Get a coordinate of top right."""
        xy = self.center
        return RealXY(x=xy.x + self.width / 2, y=xy.y + self.height / 2)

    @staticmethod
    def intersect_between(atom_i: "Atom", atom_j: "Atom") -> bool:
        """Check if two rectangles are intersectd."""
        if atom_i.bottom_left.x > atom_j.top_right.x or atom_i.top_right.x < atom_j.bottom_left.x:
            return False
        if atom_i.bottom_left.y > atom_j.top_right.y or atom_i.top_right.y < atom_j.bottom_left.y:
            return False
        return True
    
    def move_to_(self, loc: GridLoc, move_charges_too: bool = True) -> None:
        """Move it to the given location."""
        self.loc = loc
        if move_charges_too:
            for charge, (x, y) in zip(self.charges, self.relative_xy):
                charge.xy = RealXY(x=self.center.x + x, y=self.center.y + y)

    def __repr__(self) -> str:
        """Display info of this."""
        return (f"(Atom class) name: {self.name}, loc: {self.loc}, size: {self.width, self.height}")


class ElectricField:
    """A electric field class."""

    def __init__(self, atoms: list[Atom]) -> None:
        """Initialize."""
        self.atoms = atoms
        self.movable_atoms = [atom for atom in self.atoms if atom.is_movable]
        self.charges = [c for atom in self.atoms for c in atom.charges]

        self.d_atoms = {a.name: a for a in self.atoms}
        self.d_charges = {c.name: c for c in self.charges}

        # boundary atoms
        self.boundary_atoms = self.make_boundary_atoms()
        self.boundary_charges = [c for atom in self.boundary_atoms for c in atom.charges]
        self.base_epf = self.get_electric_potential_field(self.boundary_charges)

    def get_electric_potential_field(self, charges: list[PositiveCharge]) -> np.ndarray:
        """Get electric potential field.
        
        Formula: V = kQ / r
        """
        n_bins = N_ROWS * N_COLS
        field = np.array(
            [
                PositiveCharge.compute_electric_potential_energy(
                    xy=discrete_to_continuous(loc=GridLoc(row=row, col=col)),
                    charges=charges,
                )
                for row, col in map(divmod, range(n_bins), [N_COLS] * n_bins)
            ]
        ).reshape((N_ROWS, N_COLS))
        return field

    @property
    def electric_potential_field(self) -> np.ndarray:
        """Get electric potential field."""
        field = self.get_electric_potential_field(self.charges) + self.base_epf
        field = gaussian_filter(field, sigma=1)
        return field

    def evaluate(self) -> float:
        """Evaluate the current electric field."""
        d_forces = self.compute_electrostatic_forces()
        return -sum(math.sqrt(x ** 2 + y ** 2) for x, y in d_forces.values())
    
    def compute_electrostatic_forces(self) -> dict[str, tuple[float, float]]:
        """Compute electrostatic forces of all charges."""
        d_c_forces = {c.name: np.zeros(2) for c in self.charges}

        # compute electrostatic forces of each charge
        for c1, c2 in itertools.combinations(self.charges, 2):
            forces = PositiveCharge.compute_electrostatic_force(c1, c2)
            d_c_forces[c1.name] += forces
            d_c_forces[c2.name] -= forces

        # TODO : it can be accelerated by np.gradient (dV/dr = -qE)
        # compute electrostatic forces with boundaries
        for c1 in self.charges:
            for c2 in self.boundary_charges:
                forces = PositiveCharge.compute_electrostatic_force(c1, c2)
                d_c_forces[c1.name] += forces
        
        return d_c_forces

    def make_boundary_atoms(self) -> None:
        """Add atoms to boudnary to make movable atoms locate inner field."""        
        rows, cols = np.indices((N_ROWS, N_COLS))
        edge_indices =  set(zip(rows[0], cols[0]))  # top
        edge_indices |= set(zip(rows[:, -1], cols[:, -1]))  # right
        edge_indices |= set(zip(rows[-1], cols[-1]))  # bottom
        edge_indices |= set(zip(rows[:, 0], cols[:, 0]))  # left

        atoms = [
            Atom(
                name=f"boundary_{idx}",
                loc=GridLoc(row=row, col=col),
                width=BIN_WIDTH,
                height=BIN_HEIGHT,
                charge_info=[ChargeInfo(x=0, y=0, quantity=3.0)],
                is_fixed=True,
                is_boundary=True,
            )
            for idx, (row, col) in enumerate(sorted(edge_indices))
        ]
        return atoms

    def __repr__(self) -> str:
        """Display info of this."""
        return f"(Electric field) n_atoms: {len(self.atoms)}, n_charges: {len(self.charges)}"
