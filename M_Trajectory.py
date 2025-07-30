from potential_gradient import Molecules # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import pandas as pd
from os.path import expanduser

def simulation(home, new_home):
    """Run code on a directory containing unmodified .data files and store the new files in a separate folder"""
    if not Path(home).exists():
        print("ERROR: Provided home path does not exist.")
        exit(1)
    for data_file in Path(home).rglob("*.data"):
            mols = M_Trajectory(data_file)
            mols.compute()
            mols.modify_data(new_home)

def simulation_2(home, new_home):
    """Run nearest-neighbour code ver. on a directory containing unmodified .data files and store the new files in a separate folder"""
    if not Path(home).exists():
        print("ERROR: Provided home path does not exist.")
        exit(1)
    for data_file in Path(home).rglob("*.data"):
            mols = M_Trajectory(data_file)
            mols.compute_NN()
            mols.modify_data(new_home)

class M_Trajectory(Molecules):
    """Molecules subclass used for calculating intramolecular forces from .data files pertaining to Hydrogen molecule trajectories"""

    def extract_coords(self, filename):
        """Extract position data from .data file"""
        positions = []
        lat = []

        with open(filename, 'r') as f:
            for line in f:
                # Extract force lines
                if line.strip().startswith("atom"):
                    # Example line: atom 1 2 3 H 0 0 0.1 -0.2 0.3
                    parts = line.split()
                    positions.append(parts[1:4])
                elif line.strip().startswith("lattice"):
                    parts = line.split()[1:4]
                    parts_float = [float(part) for part in parts]
                    lat.append(parts_float)
        
        lattice = np.array([lat[0], lat[1], lat[2]])
        return positions, lattice
    
    
    def modify_data(self, folder):
        """Modify the .data files by subtracting intramolecular forces"""
        fname = Path(self.filename).name
        fpath = f"{folder}/{fname}"
        new_data = []
        i = -1

        with open(self.filename, "r") as f:
            for line in f:
                if line.strip().startswith("atom"):
                    i += 1
                    parts = line.split()
                    parts[-3] = "{:.6f}".format(float(parts[-3]) - self.Fx[i])
                    parts[-2] = "{:.6f}".format(float(parts[-2]) - self.Fy[i])
                    parts[-1] = "{:.6f}".format(float(parts[-1]) - self.Fz[i])
                    new_line = "{:<5} {:>12} {:>12} {:>12} {:>2} {:>15} {:>12} {:>11} {:>12} {:>12}\n".format(*parts)
                    new_data.append(new_line)
                elif line.strip().startswith("energy"):
                    parts = line.split()
                    parts[1] = "{:.6f}".format(float(parts[1]) - self.E)
                    new_line = "{:<6} {:>13}\n".format(*parts)
                    new_data.append(new_line)
                else:
                    new_data.append(line)

        with open(fpath, "w") as f:
            for line in new_data:
                f.write(line)
        
    def save_forces(self, folder):
        """Save force data to a file"""
        filename = f"{folder}/PP/forces.txt"

        with open(filename, "w") as f:
            f.write("Atom,Fx (eV/A),Fy (eV/A),Fz (eV/A),Updates\n")
            for i in range(self.N):
                f.write(f"{i+1},{self.Fx[i]},{self.Fy[i]},{self.Fz[i]},{self.count[i]}\n")


    def modify_data_test(self, folder):
        """Modify the .data files to test which atoms are causing issues"""
        fname = "mod input with counts.txt"
        fpath = f"{folder}/{fname}"
        new_data = []
        i = -1

        with open(self.filename, "r") as f:
            for line in f:
                if line.strip().startswith("atom"):
                    i += 1
                    parts = line.split()
                    A = str(float(parts[-3]) - self.Fx[i])
                    B = str(float(parts[-2]) - self.Fy[i])
                    C = str(float(parts[-1]) - self.Fz[i])
                    new_line = f"{i+1},{A},{B},{C},{self.count[i]}\n"
                    new_data.append(new_line)
                elif line.strip().startswith("energy"):
                    parts = line.split()
                    energy = parts[1]
                    new_data.append(f"energy,{energy}\n")

        with open(fpath, "w") as f:
            f.write("atom,Fx,Fy,Fz,updates\n")
            for line in new_data:
                f.write(line)




if __name__ == "__main__":
    folder = expanduser("") # folder containing raw input.data files
    newfolder = expanduser("") # folder to contain the modified input.data files
