import numpy as np

import plotly.colors as pco
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image

from electric_field import ElectricField
from utils import (
    FIELD_HEIGHT, 
    FIELD_WIDTH, 
    N_COLS, 
    N_ROWS,
)


class Plotter:
    """A logger class."""

    @staticmethod
    def plot_electric_density_field(
        electric_field: ElectricField,
        title: str = "Electric Potnetial",
        fpath: str = "electric_density_field.png",
    ) -> None:
        """Plot contour and surface graph."""
        field = electric_field.electric_potential_field

        # make base
        fig = make_subplots(
            rows=1, 
            cols=2,
            specs=[[{'is_3d': False}, {'is_3d': True}]],
            horizontal_spacing=0.04,
            subplot_titles=["Placement", "3D Contour Graph"],
        )

        # draw potential field
        fig.add_trace(
            go.Contour(
                x=np.linspace(0, N_COLS - 1, N_COLS) + 0.5,
                y=np.linspace(0, N_ROWS - 1, N_ROWS) + 0.5,
                z=field,
                showscale=False,
                opacity=0.5,
            ),
            row=1, 
            col=1,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # draw atoms
        group_names = {atom.group_name for atom in electric_field.atoms}

        color_maps = sns.color_palette("husl", len(group_names))
        d_group_colors = {
            gname: pco.label_rgb(color) 
            for gname, color in zip(group_names, color_maps)
        }
        for atom in electric_field.atoms:
            fig.add_shape(
                x0=(atom.bottom_left.x / FIELD_WIDTH + 0.5) * N_COLS,
                x1=(atom.top_right.x / FIELD_WIDTH + 0.5) * N_COLS,
                y0=(atom.bottom_left.y / FIELD_HEIGHT + 0.5) * N_ROWS,
                y1=(atom.top_right.y / FIELD_HEIGHT + 0.5) * N_ROWS,
                line=dict(color="Black" if atom.is_fixed else "green"),
                fillcolor="LightGray" if atom.is_fixed else d_group_colors[atom.group_name],
                opacity=0.7,
                row=1,
                col=1,
            )
        
        # draw boundary atoms
        draw_args = dict(line=dict(width=0.0), fillcolor="LightGray")
        fig.add_shape(x0=0, x1=N_COLS, y0=0, y1=1, **draw_args)  # bottom
        fig.add_shape(x0=0, x1=1, y0=0, y1=N_ROWS, **draw_args)  # left
        fig.add_shape(x0=0, x1=N_COLS, y0=N_ROWS-1, y1=N_ROWS, **draw_args)  # top
        fig.add_shape(x0=N_COLS - 1, x1=N_COLS, y0=0, y1=N_ROWS, **draw_args)  # right

        # draw charges
        charges = [c for atom in electric_field.atoms for c in atom.charges if not atom.is_boundary]
        fig.add_trace(
            go.Scatter(
                x=[(c.xy.x / FIELD_WIDTH + 0.5) * N_COLS for c in charges], 
                y=[(c.xy.y / FIELD_HEIGHT + 0.5) * N_ROWS for c in charges], 
                mode="markers+text",
                # text=[c.name for c in charges],
                # textposition="top right",
                # textfont=dict(color="Black"),
                showlegend=False,
            ),
            row=1, 
            col=1,
        )

        # draw electrostatic force as arrow
        d_c_forces = electric_field.compute_electrostatic_forces()
        for c_name, (force_x, force_y) in d_c_forces.items():
            c = electric_field.d_charges[c_name]
            cx = (c.xy.x / FIELD_WIDTH + 0.5) * N_COLS
            cy = (c.xy.y / FIELD_HEIGHT + 0.5) * N_ROWS
            fig.add_trace(
                go.Scatter(
                    x=[cx, cx + force_x * 1000],
                    y=[cy, cy + force_y * 1000],
                    marker=dict(
                        color="Black",
                        size=8,
                        symbol="arrow-bar-up",
                        angleref="previous",
                    ),
                    showlegend=False,
                )
            )

        # draw 3d contour surface
        fig.update_yaxes(range=[0, N_ROWS], dtick=1, row=1, col=1)
        fig.update_xaxes(range=[0, N_COLS], dtick=1, row=1, col=1)
        fig.add_trace(
            go.Surface(
                x=[str(i) for i in range(N_COLS)][::-1],
                y=[str(i) for i in range(N_ROWS)][::-1],
                z=np.array([row[::-1] for row in field[::-1]]),
                showscale=False,
            ),
            row=1,
            col=2,
        )
        # fig.update_scenes(zaxis=dict(range=[0, None]), row=1, col=2)

        fig.update_layout(
            title_text=title, 
            width=FIELD_WIDTH * 11, 
            height=FIELD_HEIGHT * 6,
            margin=dict(l=20, r=20, b=20, t=70),
            scene_aspectmode='auto',
        )
        pio.write_image(fig, fpath)

    @staticmethod
    def make_gif(dirpath: Path) -> None:
        """Make a gif of optimization process."""
        image_files = list(dirpath.glob("[0-9]*.png"))
        image_files = sorted(image_files, key=lambda x: int(x.stem))
        images = [Image.open(x) for x in image_files]
        
        im = images[0]
        im.save(
            dirpath / "result.gif",
            save_all=True, 
            append_images=images[1:],
            loop=0xff, 
            duration=300,
        )
        print(f"Done: {dirpath}")
