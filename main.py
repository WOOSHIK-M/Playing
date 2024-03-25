import copy
import random
import shutil

import numpy as np

from pathlib import Path

from electric_field import ElectricField, Atom
from specs import *
from plotter import Plotter
from utils import GridLoc, N_COLS, N_ROWS, FIELD_WIDTH, FIELD_HEIGHT


class SimulatedAnnealing:
    """Do simulated annealing to optimize."""
    
    def __init__(
        self,
        electric_field: ElectricField,
        init_temp: float = 1e-1,
        threshold: float = 1e-4,
        cooling_factor: float = 0.95,
        n_iters: int = 2000,
        save_dir: str = "test",
    ) -> None:
        """Initialize."""
        self.electric_field = electric_field

        self.init_temp = init_temp
        self.threshold = threshold
        self.cooling_factor = cooling_factor
        self.n_iters = n_iters
        
        rows, cols = np.indices((N_ROWS, N_COLS))
        self.available_actions = [
            GridLoc(row=row, col=col) 
            for row, col in zip(rows.reshape(-1), cols.reshape(-1))
        ]

        self.ready_to_run(save_dir=save_dir)

    def run(self, is_eval: bool = False) -> float:
        """Do optimization."""
        if is_eval:
            return self.init_reward

        n_cooled, cur_temp, cur_reward = 1, self.init_temp, self.init_reward
        while cur_temp > self.threshold:
            for _ in range(self.n_iters):
                atom = self.get_an_atom_to_move()
                ori_loc = atom.loc

                tmp_loc = self.get_a_valid_location_to_move(atom)
                atom.move_to_(tmp_loc)
                tmp_reward = self.electric_field.evaluate()

                # do an action or not ?
                if self.is_allowed(cur_temp, cur_reward, tmp_reward):
                    cur_reward = tmp_reward
                
                    # update the best !
                    if tmp_reward > self.opt_reward:
                        self.opt_reward = tmp_reward
                        self.opt_field = copy.deepcopy(self.electric_field)
                        Plotter.plot_electric_density_field(
                            electric_field=self.electric_field,
                            title=(f"Optimal Score: {self.opt_reward:.6f}"),
                            fpath=self.save_dir / "optimal.png",
                        )
                else:
                    atom.move_to_(ori_loc)
            
            print(
                f"Current Temperature: {cur_temp:.6f}, "
                f"Potential Energy: {cur_reward:.6f}, "
                f"Optimum: {self.opt_reward:.6f}"
            )
            fpath = self.save_dir / f"{n_cooled}.png"
            Plotter.plot_electric_density_field(
                electric_field=self.electric_field,
                title=(
                    f"# of iterations: {n_cooled}, "
                    f"Current Score: {cur_reward:.6f}, "
                    f"Optimal Score: {self.opt_reward:.6f}"
                ),
                fpath=fpath,
            )
            shutil.copy(src=fpath, dst=self.save_dir / "last.png")
            n_cooled += 1
            cur_temp *= self.cooling_factor

            if cur_reward < self.opt_reward:
                self.electric_field = copy.deepcopy(self.opt_field)

        # save the optimal
        Plotter.plot_electric_density_field(
            electric_field=self.opt_field,
            title=(
                f"# of iterations: {n_cooled}, "
                f"Current Score: {self.opt_reward:.6f}, "
                f"Optimal Score: {self.opt_reward:.6f}"
            ),
            fpath=fpath,
        )
        Plotter.make_gif(self.save_dir)
        return self.opt_reward
    
    def is_allowed(
        self,
        cur_temp: float,
        cur_reward: float,
        tmp_reward: float,
    ) -> bool:
        """Determine whether the current change is allowed."""
        if tmp_reward > cur_reward:
            return True
        
        delta_e = (cur_reward - tmp_reward)
        prob = np.exp(-delta_e / cur_temp)
        print(
            f"Temp: {cur_temp:.6f}, "
            f"Cur_R: {cur_reward:.6f}, Tmp_R: {tmp_reward:.6f}, "
            f"delta_e: {delta_e:.6f}, Prob: {prob:.4f}"
        )
        return random.random() < prob
        
    def get_an_atom_to_move(self) -> Atom:
        """Get an atom to move."""
        return random.choice(self.electric_field.movable_atoms)
    
    def get_a_valid_location_to_move(self, atom: Atom) -> GridLoc:
        """Get a valid action."""
        ori_loc = atom.loc
        tg_atoms = [
            a 
            for a in self.electric_field.atoms 
            if atom.name != a.name and not a.is_boundary
        ]
        while True:
            loc = random.choice(self.available_actions)
            atom.move_to_(loc=loc, move_charges_too=False)

            # it is out of field
            if (
                atom.top_right.x > FIELD_WIDTH / 2
                or atom.top_right.y > FIELD_HEIGHT / 2
                or atom.bottom_left.x < -FIELD_WIDTH / 2
                or atom.bottom_left.y < -FIELD_HEIGHT / 2
            ):
                continue

            # overlapped with others
            if any(Atom.intersect_between(atom, tg_atom) for tg_atom in tg_atoms):
                continue

            break
    
        # revert atom locations
        atom.move_to_(loc=ori_loc)
        return loc

    def ready_to_run(self, save_dir: str) -> None:
        """Ready to optimize field."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # do init
        self.init_reward = self.opt_reward = self.electric_field.evaluate()
        self.opt_field = copy.deepcopy(self.electric_field)

        # logging the init info
        print(f"Init reward : {self.init_reward:.6f}")

        fpath = self.save_dir / "0.png"
        Plotter.plot_electric_density_field(
            electric_field=self.electric_field,
            title=(
                f"# of iterations: 0, "
                f"Current Score: {self.init_reward:.6f}, "
                f"Optimal Score: {self.opt_reward:.6f}"
            ),
            fpath=fpath,
        )
        shutil.copy(src=fpath, dst=self.save_dir / "optimal.png")

SimulatedAnnealing(
    electric_field=ElectricField(atoms=EXPERT_ATOMS),
    save_dir="expert_sheet_number",
).run()
