import math
import random
import shutil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from scipy.stats import entropy
from plotly.subplots import make_subplots
from PIL import Image
from pathlib import Path


# 6301 setting
FIELD_WIDTH = 105.3
FIELD_HEIGHT = 62.30000074101214
N_ROWS = 100
N_COLS = int(N_ROWS * FIELD_WIDTH / FIELD_HEIGHT + 1)

SAVE_DIR = "save"
TICKER = "copycat"

SAVE_DIR += f"_{TICKER}"
BIN_WIDTH, BIN_HEIGHT = FIELD_WIDTH / N_COLS, FIELD_WIDTH / N_ROWS

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

    @staticmethod
    def discrete_to_continous(loc: tuple[int, int]) -> "Location":
        """Convert discrete action to continuous coordinates."""
        return Location(
            x=(loc[1] + 0.5) * (FIELD_WIDTH / N_COLS) - FIELD_WIDTH / 2,
            y=(loc[0] + 0.5) * (FIELD_HEIGHT / N_ROWS) - FIELD_HEIGHT / 2,
        )


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
            # a charge belongs to its bin, no computation
            if abs(loc.x - c.loc.x) < BIN_WIDTH / 2 and abs(loc.y - c.loc.y) < BIN_HEIGHT / 2:
                continue

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

    def __init__(
        self, 
        name: str,
        x: float, 
        y: float, 
        width: float, 
        height: float,
        fixed: bool = False,
        is_boundary: bool = False,
    ) -> None:
        """Initialize."""
        self.name = name

        self.center = Location(x=x, y=y)
        self.width = width
        self.height = height
        
        self.charges = [Charge(loc=Location(x=x, y=y), size=self.size)]
        self.fixed = fixed
        self.is_boundary = is_boundary

    @staticmethod
    def intersect_between(rect_i: "Rectangle", rect_j: "Rectangle") -> bool:
        """Check if two rectangles are intersectd."""
        if rect_i.bottom_left.x > rect_j.top_right.x or rect_i.top_right.x < rect_j.bottom_left.x:
            return False
        if rect_i.bottom_left.y > rect_j.top_right.y or rect_i.top_right.y < rect_j.bottom_left.y:
            return False
        return True

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
            fillcolor="LightGray" if self.fixed else "LightSkyBlue",
            opacity=0.8 if self.fixed else 0.5,
        )
        return fig


class ElectricField:
    """A electric field class."""

    def __init__(self) -> None:
        """Initialize."""
        
        import pandas as pd
        
        df = pd.read_csv("data/expert.csv")
        self.objs = [
            Rectangle(
                name=row["Name"],
                x=row["left"] + row["width"] / 2 - FIELD_WIDTH / 2,
                y=row["top"] + row["height"] / 2 - FIELD_HEIGHT / 2,
                width=row["width"],
                height=row["height"],
                fixed=row["Fixed"],
            )
            for row in df.iloc
            if row["PlacedLayer"] == "TOP"
        ]
        self._add_boundary_charges(b_charge_width=5, b_charge_height=5)
            
        # get movable objects
        self.movable_objs = [obj for obj in self.objs if not obj.fixed]

        self.charges = []
        for obj in self.objs:
            self.charges += obj.charges

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
            x=(col_idx + 0.5) * BIN_WIDTH - FIELD_WIDTH / 2, 
            y=(row_idx + 0.5) * BIN_HEIGHT - FIELD_HEIGHT / 2,
        )
    
    def _add_boundary_charges(self, b_charge_width: float, b_charge_height: float) -> None:
        """Add boundary objects."""
        # top
        self.objs += [
            Rectangle(
                name="boundary_top",
                x=x_coord - FIELD_WIDTH / 2 + FIELD_WIDTH % b_charge_width - 1,
                y=FIELD_HEIGHT / 2,
                width=b_charge_width,
                height=1,
                fixed=True,
                is_boundary=True,
            )
            for x_coord in range(0, np.round(FIELD_WIDTH + b_charge_width).astype(np.int64), b_charge_width)
        ]
        # bottom
        self.objs += [
            Rectangle(
                name="boundary_bottom",
                x=x_coord - FIELD_WIDTH / 2 + FIELD_WIDTH % b_charge_width - 1,
                y=-FIELD_HEIGHT / 2,
                width=b_charge_width,
                height=1,
                fixed=True,
                is_boundary=True,
            )
            for x_coord in range(0, np.round(FIELD_WIDTH + b_charge_width).astype(np.int64), b_charge_width)
        ]
        # left
        self.objs += [
            Rectangle(
                name="boundary_left",
                x=-FIELD_WIDTH / 2,
                y=y_coord - FIELD_HEIGHT / 2 + FIELD_HEIGHT % b_charge_height - 1,
                width=1,
                height=b_charge_height,
                fixed=True,
                is_boundary=True,
            )
            for y_coord in range(0, np.round(FIELD_HEIGHT + b_charge_height).astype(np.int64), b_charge_height)
        ]
        # right
        self.objs += [
            Rectangle(
                name="boundary_right",
                x=FIELD_WIDTH / 2,
                y=y_coord - FIELD_HEIGHT / 2 + FIELD_HEIGHT % b_charge_height - 1,
                width=1,
                height=b_charge_height,
                fixed=True,
                is_boundary=True,
            )
            for y_coord in range(0, np.round(FIELD_HEIGHT + b_charge_height).astype(np.int64), b_charge_height)
        ]


class SimulatedAnnealing:
    """Do SA."""

    def __init__(self) -> None:
        """Initialize."""
        self.electric_field = ElectricField()

        self.save_dir = Path(SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # resolve overlapping first
        # for obj in self.electric_field.objs:
        #     if obj.fixed:
        #         continue
        #     self._move_to_non_overlapped(obj)

    def run(
        self,
        init_temp: float = 100.0,
        threshold: float = 5.0,
        cooling_factor: float = 0.95,
        n_iters: int = 500,
    ) -> None:
        """Optimize electric field."""
        init_field, init_reward = self.evaluate_electric_field()
        infos, n_optimized = [(init_temp, init_reward)], 1
        self.dump_img(init_field, init_reward, infos, title="0")

        cur_temp, cur_reward, opt_reward = init_temp, init_reward, init_reward
        while cur_temp > threshold:
            for _ in range(n_iters):
                obj = random.choice(self.electric_field.movable_objs)   
                
                ori_loc = obj.center
                obj.move_to_(self._move_to_non_overlapped(obj))

                _, tmp_reward = self.evaluate_electric_field()

                if self.is_allowed(cur_temp, cur_reward, tmp_reward):
                    infos.append((cur_temp, tmp_reward))
                    cur_reward = tmp_reward                    

                    # update the best !
                    if tmp_reward < opt_reward:   
                        opt_reward = tmp_reward 
                        self.dump_img(
                            *self.evaluate_electric_field(), 
                            infos=infos,
                            title=f"{n_optimized}"
                        )
                        n_optimized += 1
                else:
                    obj.move_to_(ori_loc)
        
            print(
                f"Current Temperature: {cur_temp:.4f}, "
                f"Potential Energy: {cur_reward:.6f}, "
                f"Optimum: {opt_reward:.6f}"
            )
            cur_temp *= cooling_factor

        self.dump_img(
            *self.evaluate_electric_field(),
            infos=infos,
            title=f"{n_optimized}",
        )
        self.make_gif()
    
    def  _move_to_non_overlapped(self, obj: Rectangle) -> Location:
        """."""
        non_boundaries = [
            tg_obj
            for tg_obj in self.electric_field.objs 
            if not tg_obj.is_boundary and tg_obj.name != obj.name
        ]

        available_x = (obj.width - FIELD_WIDTH)
        available_y = (obj.height - FIELD_HEIGHT)

        while True:
            x = random.random() * available_x - available_x / 2
            y = random.random() * available_y - available_y / 2
            
            obj.move_to_(Location(x=x, y=y))
            if not any(
                Rectangle.intersect_between(rect_i=obj, rect_j=tg_obj) for tg_obj in non_boundaries
            ): 
                break
        return Location(x=x, y=y)

    def is_allowed(
        self,
        cur_temp: float,
        cur_reward: float,
        tmp_reward: float,
    ) -> bool:
        """Determine whether the current change is allowed."""
        if tmp_reward < cur_reward:
            return True
        
        delta_e = (tmp_reward - cur_reward) * 1e5
        prob = np.exp(-delta_e / cur_temp)
        # print(
        #     f"Temp: {cur_temp:.2f}, Cur_R: {cur_reward:.6f}, "
        #     f"Tmp_R: {tmp_reward:.6f}, Prob: {prob}"
        # )
        return random.random() < prob

    def evaluate_electric_field(self) -> tuple[np.ndarray, float]:
        """Return field and entropy."""
        field = self.electric_field.potential_field
        return field, entropy(field, axis=None)

    def dump_img(
        self, 
        field: np.ndarray, 
        reward: float, 
        infos: list[tuple[float, float]],
        title: str = "Default") -> None:
        """Dump img."""
        objs = self.electric_field.objs
        charges = self.electric_field.charges

        # set figure specs
        fig = make_subplots(
            rows=2, 
            cols=2, 
            specs=[
                [{'is_3d': False}, {'is_3d': True}],
                [{'colspan': 2, 'is_3d': False}, None],
            ],
            row_heights=[0.5, 0.3],
            horizontal_spacing=0.04,
            vertical_spacing=0.1,
            subplot_titles=[
                "2D Placement",  # (1, 1)
                "3D Contour Grpah",  # (1, 2)
                "Learning Curve",  # (2, 1-2)
            ],
        )

        # 2d - contour / charges / box
        fig = self.draw_potential_field(fig, field=field)
        fig = self.draw_objects_and_charges(fig, objs=objs, charges=charges)

        # 3d
        fig = self.draw_surface(fig, field=field)

        # learning curve
        fig = self.make_learning_curve(fig, infos)

        fig.update_layout(
            title_text=f"# of improved: {title}, Potential energy: {reward:.6f}", 
            width=2400, 
            height=1600,
        )
        
        fpath = self.save_dir / f"{title}.png"
        pio.write_image(fig, fpath)
        shutil.copy(src=fpath, dst=f"optimal_{TICKER}.png")
    
    def draw_potential_field(self, fig: go.Figure, field: np.ndarray) -> go.Figure:
        """Draw contour graph."""
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
        return fig

    def draw_objects_and_charges(
        self, 
        fig: go.Figure, 
        objs: list[Rectangle], 
        charges: list[Charge]
    ) -> go.Figure:
        """Draw charges."""
        c_locs = [
            c.loc.get_idx(N_ROWS, N_COLS, FIELD_WIDTH, FIELD_HEIGHT, match_to_bin=False) 
            for c in charges
        ]
        c_y, c_x = np.array(c_locs).transpose()
        text_labels = ["" if o.is_boundary else f"{o.name}" for o in objs]
        fig.add_trace(
            go.Scatter(
                x=c_x, 
                y=c_y,
                text=text_labels,
                mode="markers+text",
                marker=dict(color="black"),
                textposition="top right",
                textfont=dict(color="white"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        for obj in objs:
            fig = obj.draw_rectangle(fig)
        return fig

    def draw_surface(self, fig: go.Figure, field: np.ndarray) -> go.Figure:
        """Draw objects."""  
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
        return fig

    def make_learning_curve(self, fig: go.Figure, infos: list[float]) -> go.Figure:
        """Draw leanring curve."""
        if not infos:
            return fig

        x, y = np.array(infos).transpose()
        
        # add scatters
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers", 
                marker=dict(opacity=0.2, color="LightSkyBlue"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        # add average line
        df = pd.DataFrame({"x": x, "y": y}).groupby(by="x").mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=df["x"], 
                y=df["y"], 
                mode="lines", 
                line=dict(color="SkyBlue"),
                showlegend=False,
            ),
            row=2, 
            col=1,
        )

        fig.update_xaxes(
            autorange="reversed", 
            type="log",
            title_text="Temperature", 
            row=2, 
            col=1,
        )
        fig.update_yaxes(title_text="Potential Energy", row=2, col=1)
        return fig

    def make_gif(self) -> None:
        """Make a .gif file."""
        # List image files
        image_files = list(self.save_dir.iterdir())
        image_files = sorted(image_files, key=lambda x: int(x.stem))
        images = [Image.open(x) for x in image_files]
        
        im = images[0]
        im.save(f'result_{TICKER}.gif', save_all=True, append_images=images[1:],loop=0xff, duration=300)
        print("Done")

SimulatedAnnealing().run()