import math
import random
import numpy as np

from scipy.stats import entropy
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

from pathlib import Path


FIELD_WIDTH = FIELD_HEIGHT = 100
N_OBJECTS = 30
N_FIXED = 5

N_ROWS = N_COLS = 100

class Location:
    """Basic location class."""

    def __init__(self, x: float, y: float) -> None:
        """Initialize."""
        self.x, self.y = x, y

    def get_idx(
        self, 
        n_rows: int,
        n_cols: int,
        width: float,
        height: float,
        match_to_bin: bool = True,
    ) -> tuple[int]:
        """Get bin inidcies."""
        row_idx = ((self.y + height / 2) / height * n_rows)
        col_idx = ((self.x + width / 2) / width * n_cols)
        if match_to_bin:
            row_idx = np.floor(row_idx).astype(np.int64)
            col_idx = np.floor(col_idx).astype(np.int64)
        return row_idx, col_idx


class Charge:
    """A base charge."""

    def __init__(
        self, 
        loc: Location,
        size: float = 1.0,
    ) -> None:
        """Initialize."""
        self.loc = loc
        self.size = size

    @staticmethod
    def compute_potential_energy(charges: list["Charge"], loc: Location) -> float:
        """Compute potential energy from a charge at loc."""
        force_x = force_y = 0.0
        for c in charges:
            theta = math.atan2(loc.y - c.loc.y, loc.x - c.loc.x)
            r = math.sqrt((loc.x - c.loc.x) ** 2 + (loc.y - c.loc.y) ** 2)
            if not r:
                return -1

            # V = kQ/r
            force = c.size / r

            force_x += force * math.cos(theta) if c.loc.x != loc.x else 0.0
            force_y += force * math.sin(theta) if c.loc.y != loc.y else 0.0
        return math.sqrt(force_x ** 2 + force_y ** 2)

    def __repr__(self) -> str:
        """."""
        return f"size: {self.size}, loc: {self.loc.x, self.loc.y}"


class Rectangle:
    """A basic rectangle class."""

    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        """Initialize."""
        self.center = Location(x=x, y=y)
        self.width = width
        self.height = height
        
        self.charges = [Charge(loc=Location(x=x, y=y), size=self.size)]
        self.fixed = False

    @property
    def size(self) -> float:
        """Get size."""
        return self.width * self.height
    
    @property
    def bottom_left(self) -> Location:
        """."""
        return Location(x=self.center.x - self.width / 2, y=self.center.y - self.height / 2)

    @property
    def top_right(self) -> Location:
        """."""
        return Location(x=self.center.x + self.width / 2, y=self.center.y + self.height / 2)

    def move_to_(self, loc: Location) -> None:
        """Move this rectangle and its charges."""
        self.center = loc
        for c in self.charges:
            c.loc = self.center

    def draw_rectangle(self, fig: go.Figure) -> go.Figure:
        """."""
        fig.add_shape(
            x0=(self.bottom_left.x / FIELD_WIDTH + 0.5) * N_COLS,
            x1=(self.top_right.x / FIELD_WIDTH + 0.5) * N_COLS,
            y0=(self.bottom_left.y / FIELD_HEIGHT + 0.5) * N_ROWS,
            y1=(self.top_right.y / FIELD_HEIGHT + 0.5) * N_ROWS,
            line=dict(color="Black" if self.fixed else "RoyalBlue"),
            fillcolor="Gray" if self.fixed else "LightSkyBlue",
            opacity=0.8 if self.fixed else 0.5,
        )
        return fig


class ElectricField:
    """A electric field class."""

    def __init__(self) -> None:
        """Initialize."""
        self.objs = [
            Rectangle(
                # x=-FIELD_WIDTH / 2, y=-FIELD_HEIGHT / 2,
                # x=0.0, y=0.0,
                x=random.random() * FIELD_WIDTH - FIELD_WIDTH / 2,
                y=random.random() * FIELD_HEIGHT - FIELD_HEIGHT / 2,
                width=random.random() * 10 + 1,
                height=random.random() * 10 + 1,
            )
            for _ in range(N_OBJECTS)
        ]
        self.objs = sorted(self.objs, key=lambda rect: rect.size, reverse=True)
        for obj in self.objs[:N_FIXED]:
            obj.fixed = True
        self.movable_objs = [obj for obj in self.objs if not obj.fixed]

        self.charges = []
        for obj in self.objs:
            self.charges += obj.charges
    
        self.bin_width, self.bin_height = FIELD_WIDTH / N_COLS, FIELD_WIDTH / N_ROWS

    @property
    def potential_field(self) -> np.ndarray:
        """Compute the potential energy of each location."""
        field = np.array(
            [
                Charge.compute_potential_energy(self.charges, self.compute_bin_location(idx))
                for idx in range(N_ROWS * N_COLS)
            ]
        ).reshape((N_ROWS, N_COLS))
        field[field == -1] = field.max()
        field = np.log(np.log(field + 1) + 1)
        field /= field.max()
        return field

    def compute_bin_location(self, idx: int) -> Location:
        """Compute a location of the center of bin."""
        row_idx, col_idx = divmod(idx, N_COLS)    
        return Location(
            x=(col_idx + 0.5) * self.bin_width - FIELD_WIDTH / 2, 
            y=(row_idx + 0.5) * self.bin_height - FIELD_HEIGHT / 2,
        )


class SimulatedAnnealing:
    """Do SA."""

    def __init__(self) -> None:
        """Initialize."""
        self.electric_field = ElectricField()

        self.save_dir = Path("save")
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def run(self) -> None:
        """Optimize electric field."""
        init_field, init_reward = self.evaluate_electric_field()
        self.dump_img(init_field, init_reward, title="0")

        idx = 1

        cur_reward = init_reward
        while True:
            obj = random.choice(self.electric_field.movable_objs)   
            
            ori_loc = obj.center
            obj.move_to_(
                Location(
                    x=random.random() * FIELD_WIDTH - FIELD_WIDTH / 2,
                    y=random.random() * FIELD_HEIGHT - FIELD_HEIGHT / 2,
                )
            )

            _, tmp_reward = self.evaluate_electric_field()
            if tmp_reward >= cur_reward:
                obj.move_to_(ori_loc)
            else:
                cur_reward = tmp_reward
                print(f"# of epochs: {idx}, potential energy: {cur_reward:.4f}")
                self.dump_img(*self.evaluate_electric_field(), title=f"{idx}")
                idx += 1

            if idx == 100:
                break
        
        self.make_gif()

    def evaluate_electric_field(self) -> tuple[np.ndarray, float]:
        """Return field and entropy."""
        field = self.electric_field.potential_field
        return field, entropy(field, axis=None)

    def dump_img(self, field: np.ndarray, reward: float, title: str = "Default") -> None:
        """Dump img."""
        objs = self.electric_field.objs
        charges = self.electric_field.charges

        fig = make_subplots(
            rows=1, 
            cols=2, 
            specs=[[{'is_3d': False}, {'is_3d': True}]],
            horizontal_spacing=0.04,
            subplot_titles=["2D Contour Graph", "3D Contour Grpah"],
        )

        # 2d - contour
        fig.add_trace(
            go.Contour(
                x=np.linspace(0, N_COLS - 1, N_COLS) + 0.5,
                y=np.linspace(0, N_ROWS - 1, N_ROWS) + 0.5,
                z=field, 
                showscale=False,
            ), 
            row=1, 
            col=1,
        )

        # 2d - scatter
        c_locs = [
            c.loc.get_idx(N_ROWS, N_COLS, FIELD_WIDTH, FIELD_HEIGHT, match_to_bin=False) 
            for c in charges
        ]
        c_y, c_x = np.array(c_locs).transpose()
        c_labels = [f"{c.size * 100:.0f}" for c in charges]
        fig.add_trace(
            go.Scatter(
                x=c_x, 
                y=c_y,
                text=c_labels,
                mode="markers+text",
                marker=dict(color="black"),
                textposition="top right",
                textfont=dict(color="white"),
            ),
            row=1,
            col=1,
        )
        # 2d - rectangle
        for obj in objs:
            fig = obj.draw_rectangle(fig)

        fig.update_yaxes(range=[0, N_ROWS], dtick=1, row=1, col=1)
        fig.update_xaxes(range=[0, N_COLS], dtick=1, row=1, col=1)

        # 3d
        z = np.array([row[::-1] for row in field[::-1]])
        fig.add_trace(
            go.Surface(
                x=[str(i) for i in range(N_ROWS)][::-1],
                y=[str(i) for i in range(N_COLS)][::-1],
                z=z,
                showscale=False,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title_text=f"# of epochs: {title}, Entropy: {reward:.4f}", 
            width=1800, 
            height=900,
        )
        pio.write_image(fig, self.save_dir / f"{title}.png")

    def make_gif(self) -> None:
        """."""
        import imageio
        from PIL import Image

        # List image files
        image_files = list(self.save_dir.iterdir())
        image_files = sorted(image_files, key=lambda x: int(x.stem))
        images = [Image.open(x) for x in image_files]
        
        im = images[0]
        im.save('result.gif', save_all=True, append_images=images[1:],loop=0xff, duration=300)
        print("Done")

SimulatedAnnealing().run()