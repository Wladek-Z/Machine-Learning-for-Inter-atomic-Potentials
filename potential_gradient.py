import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import pandas as pd
from itertools import product
from scipy.spatial import KDTree

r_ci = 1 # Angstrom
r_c = 1.5 # Angstrom
E_c = -12.534 # Cutoff energy per atom, eV
r_min = 0.753 # Equilibrium bond length, Angstrom

C = np.zeros(13)

# Coefficients obtained from curve fitting in fit_potentials.py. Also stored in ELJ12.txt
C[0] = -5.65429887e+01
C[1] = 3.17871633e+02
C[2] = -1.36366521e+03
C[3] = 3.28161439e+03
C[4] = -5.10301803e+03
C[5] = 5.38631149e+03
C[6] = -3.91966493e+03
C[7] = 1.95568837e+03
C[8] = -6.48949185e+02
C[9] = 1.32615401e+02
C[10] = -1.33895098e+01
#
C[12] = 8.60247231e-02

func_V = lambda x: C[0] + C[1] / x + C[2] / x**2 + C[3] / x**3 + C[4] / x**4 + C[5] / x**5 + C[6] / x**6 + C[7] / x**7 + C[8] / x**8 + C[9] / x**9 + C[10] / x**10 + C[11] / x**11 + C[12] / x**12 - 2 * E_c
func_dV = lambda x: -(C[1] / x**2 + 2 * C[2] / x**3 + 3 * C[3] / x**4 + 4 * C[4] / x**5 + 5 * C[5] / x**6 + 6 * C[6] / x**7 + 7 * C[7] / x**8 + 8 * C[8] / x**9 + 9 * C[9] / x**10 + 10 * C[10] / x**11 + 11 * C[11] / x**11 + 12 * C[12] / x**13)

# Functions required for calculations in sim
def cutoff(r, V):
    """Cutoff function for separation vectors"""
    x = (r - r_ci) / (r_c - r_ci) # variable transformation
    func_c = ((15 - 6 * x) * x - 10) * x**3 + 1

    if r < r_ci:
        return V
    elif r < r_c:
        return V * func_c
    return 0

def deriv_cutoff(r, V, dV):
    """Cutoff function for force calculation. Note that r in this function is x everywhere else, unfortunately"""
    x = (r - r_ci) / (r_c - r_ci) # variable transformation
    func_c = ((15 - 6 * x) * x - 10) * x**3 + 1
    func_dc = 30 * x**2 * (2 * x - x**2 - 1)

    if r < r_ci:
        return dV
    elif r < r_c:
        return dV * func_c + V * func_dc
    return 0

def force(x):
    """Force obtained from given ELJ potential - x: Separation vector"""
    return -deriv_cutoff(x, func_V(x), func_dV(x))

def ELJ(x):
    """Extended Lennard-Jones potential - x: Separation vector"""
    return cutoff(x, func_V(x))


def simulation(home, new_home):
    """Run code on a directory filled with individual folders containing CASTEP .cell files"""
    root_folder = home
    target_filename = "H2.cell"

    with open(new_home, "a") as f:
        f.write("Separation (A),Energy (eV),Force (eV/A),Fx (eV/A),Fy (eV/A),Fz (eV/A)\n")

        for file_path in Path(root_folder).rglob(target_filename):
            folder_name = file_path.parent.name
            match = re.match(r"sep_([-+]?\d*\.?\d+)", folder_name)

            if match:
                sep = match.group(1)
                H2 = Molecules(file_path)
                H2.compute()
                E = H2.E
                F = H2.F[0]
                Fx = H2.Fx[0]
                Fy = H2.Fy[0]
                Fz = H2.Fz[0]
                
                f.write(f"{sep},{E},{F},{Fx},{Fy},{Fz}\n")

def load_data(path):
    """load data from a directory containing folder containing DFT energy/forces data and simulated energy/forces data"""
    data1 = pd.read_csv(f"{path}energies.txt", ",")
    seps = data1["Separation (A)"].values
    energies = data1["Energy (eV)"].values

    data2 = pd.read_csv(f"{path}forces.txt", ",")
    forces = data2["Force (eV/A)"].values
    fx = data2["Fx (eV/A)"].values
    fy = data2["Fy (eV/A)"].values
    fz = data2["Fz (eV/A)"].values

    C_sort = np.argsort(seps)
    seps = seps[C_sort]
    energies = energies[C_sort]
    forces = forces[C_sort]
    fx = fx[C_sort]
    fy = fy[C_sort]
    fz = fz[C_sort]

    data3 = pd.read_csv(f"{path}/sim data/sim_VF.txt")
    seps_sim = data3["Separation (A)"].values
    energies_sim = data3["Energy (eV)"].values 
    forces_sim = data3["Force (eV/A)"].values
    fx_sim = data3["Fx (eV/A)"].values
    fy_sim = data3["Fy (eV/A)"].values
    fz_sim = data3["Fz (eV/A)"].values

    S_sort = np.argsort(seps_sim)
    seps_sim = seps_sim[S_sort]
    energies_sim = energies_sim[S_sort]
    forces_sim = forces_sim[S_sort]
    fx_sim = fx_sim[S_sort]
    fy_sim = fy_sim[S_sort]
    fz_sim = fz_sim[S_sort]

    return np.array([seps, seps_sim]), np.array([energies, energies_sim]), np.array([forces, forces_sim]), np.array([fx, fx_sim]), np.array([fy, fy_sim]), np.array([fz, fz_sim])

def graph_results(filepath):
    """Produce multiple plots to compare results from the simulation with CASTEP"""
    x, E, F, Fx, Fy, Fz = load_data(filepath)

    experiment  = "two tilted molecules/molecular separation"
    #xlabel = r"Separation [$\AA$]"
    xlabel = "Molecular separation scale factor"

    # Force data taken from atom at the origin, relative coordinate system is reversed
    # We keep absolute magnitude for the full force
    Fx *= -1
    Fy *= -1
    Fz *= -1

    mask0 = (x[0] >= 1.5) & (x[0] <= 3.4)
    mask1 = (x[1] >= 1.5) & (x[1] <= 3.4)

    E0 = E[0][mask0]
    E1 = E[1][mask1]
    F0 = F[0][mask0]
    F1 = F[1][mask1]
    Fx0 = Fx[0][mask0]
    Fy0 = Fy[0][mask0]
    Fz0 = Fz[0][mask0]
    Fx1 = Fx[1][mask1]
    Fy1 = Fy[1][mask1]
    Fz1 = Fz[1][mask1]
    x0 = x[0][mask0]
    x1 = x[1][mask1]

    # Calculate residuals
    res_E = E0 - E1
    res_F = F0 - F1

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Graph of simulation energy against CASTEP energy
    ax[0].plot(x1, E1, color='orange', label='simulation')
    ax[0].scatter(x0, E0, 2, color='indigo', label='CASTEP')
    ax[0].axvline(r_min, linestyle="--", color="black")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("Energy [eV]")
    ax[0].set_title(f"Energy - {experiment}")
    ax[0].legend()

    ax[1].plot(x1, F1, color='orange', label='simulation')
    ax[1].scatter(x0, F0, 2, color='indigo', label='CASTEP')
    ax[1].axvline(r_min, linestyle="--", color="black")
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(r"Force [eV/$\AA$]")
    ax[1].set_title(f"Force - {experiment}")
    ax[1].legend()
    
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(x1[x1 <= 2], res_E[x1 <= 2], 4, color='red')
    ax[0].axhline(linestyle="--", color="black")
    ax[0].axvline(r_min, linestyle="--", color="black")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("Energy [eV]")
    ax[0].set_title(f"Energy Residuals - {experiment}")

    ax[1].scatter(x1[x1 <= 2], res_F[x1 <= 2], 4, color='red')
    ax[1].axhline(linestyle="--", color="black")
    ax[1].axvline(r_min, linestyle="--", color="black")
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(r"Force [eV/$\AA$]")
    ax[1].set_title(f"Force Residuals - {experiment}")
    
    plt.show()

    # Calculate force residuals in each cartesian component for molecule at the origin
    res_Fx = Fx0 - Fx1
    res_Fy = Fy0 - Fy1
    res_Fz = Fz0 - Fz1

    plt.scatter(x1[x1 <= 2], res_Fx[x1 <= 2], 4, color='royalblue', label='x-component')
    plt.scatter(x1[x1 <= 2], res_Fy[x1 <= 2], 4, color='deeppink', label='y-component')
    plt.scatter(x1[x1 <= 2], res_Fz[x1 <= 2], 4, color='limegreen', label='z-component')
    plt.xlabel(xlabel)
    plt.ylabel(r"Force Residual [eV/$\AA$]")
    plt.axhline(linestyle="--", color="black")
    plt.axvline(r_min, linestyle="--", color="black")
    plt.title(f"Decomposed Force Residuals - {experiment}")
    plt.legend()
    plt.show()


class Molecules():
    """Class to store information/methods for H2 molecules in a box"""

    def __init__(self, filename):
        """Generate the cartesian coordinate positions of N H2 atoms based on CASTEP files"""
        pos, lattice = self.extract_coords(filename)
        self.N = len(pos)
        self.N_PBC = self.N * 27 # 3x3x3 supercell for nearest-cell periodic boundary conditions
        self.filename = filename
        self.count = np.zeros(self.N_PBC, dtype=int)  # Count of updates for each atom

        # Convert pos to np.array of floats if not already
        pos = np.array(pos, dtype=float)
        self.positions = np.zeros((self.N_PBC, 3))
        self.positions[:self.N, :] = pos

        shifts = [np.dot([i, j, k], lattice) for i, j, k in product([-1, 0, 1], repeat=3)
          if not (i == j == k == 0)]

        for idx, shift in enumerate(shifts):
            j = (idx + 1) * self.N
            k = j + self.N
            self.positions[j:k, :] = self.positions[:self.N, :] + shift

        self.X = self.positions[:, 0]
        self.Y = self.positions[:, 1]
        self.Z = self.positions[:, 2]

    def extract_coords(self, filename):
        """Extract position data from .cell file"""
        positions = []
        params = []
        in_pos_section = False
        in_lat_section = False

        with open(filename, 'r') as f:
            for line in f:

                # Detect start of lattice section
                if "%BLOCK lattice_abc" in line:
                    in_lat_section = True
                    continue

                # Detect end of lattice section
                if in_lat_section:
                    parts = line.split()
                    params.append(parts)
                elif in_lat_section and "ENDBLOCK lattice_abc" in line:
                    in_lat_section = False

                # Detect start of forces section
                if "%BLOCK positions_abs" in line:
                    in_pos_section = True
                    continue

                # Detect end of position section
                if in_pos_section and "ENDBLOCK positions_abs" in line:
                    break

                # Extract position lines
                if in_pos_section and line.strip().startswith("H"):
                    # Example line: H 1 0 2
                    parts = line.split()
                    positions.append(parts[1:])

        lattice = self.lattice_from_parameters(params[0], params[1])

        return positions, lattice
    
    def lattice_from_parameters(self, abc, angs):
        """Function to convert lattice parameters + angles into a set of basis vectors"""
        a = float(abc[0])
        b = float(abc[1])
        c = float(abc[2])
        alpha_deg = float(angs[0])
        beta_deg = float(angs[1])
        gamma_deg = float(angs[2])

        # Convert angles to radians
        alpha = np.radians(alpha_deg)
        beta  = np.radians(beta_deg)
        gamma = np.radians(gamma_deg)

        # Cosines and sine of gamma
        cos_alpha = np.cos(alpha)
        cos_beta  = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        # Lattice vectors
        a1 = [a, 0.0, 0.0]
        a2 = [b * cos_gamma, b * sin_gamma, 0.0]
        
        # a3_x and a3_y
        a3_x = c * cos_beta
        a3_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        
        # a3_z via volume formula
        a3_z = c * np.sqrt(
            1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 
            2 * cos_alpha * cos_beta * cos_gamma) / sin_gamma

        a3 = [a3_x, a3_y, a3_z]

        # 3×3 matrix where rows are the lattice vectors
        return np.array([a1, a2, a3])

    def compute(self):
        """Compute the total energy and force on each hydrogen atom"""
        # Initialise force vectors
        self.Fx = np.zeros(self.N_PBC)
        self.Fy = np.zeros(self.N_PBC)
        self.Fz = np.zeros(self.N_PBC)

        # Initialise total energy
        self.E = 0 

        for i in range(self.N):  # atoms in the central cell
            for j in range(self.N_PBC):  # only unique pairs
                vec, sep = self.get_sep(i, j)
                if (sep <= r_c) and (i != j):
                    V = self.potential(sep)
                    self.E += V

                    Fx, Fy, Fz = self.force(vec, sep)
                    self.Fx[i] += Fx
                    self.Fy[i] += Fy
                    self.Fz[i] += Fz
                    self.count[i] += 1  # Increment count for atom i

        self.E /= 2  # Each pair counted twice
        # Only report forces for atoms in the central cell
        self.F = np.sqrt(self.Fx[:self.N]**2 + self.Fy[:self.N]**2 + self.Fz[:self.N]**2)

    
    def compute_NN(self):
        """Compute total force and energy due to each atom, but only to the nearest-neighbour"""
        coords_all = np.array([self.X, self.Y, self.Z]).T
        coords_central_cell = np.array([self.X[:self.N], self.Y[:self.N], self.Z[:self.N]]).T

        # Find the nearest-neigbours of each atom in the central cell
        kdtree = KDTree(coords_all)
        _, points = kdtree.query(coords_central_cell, 2)

        atom = points[:,1]

        # Initialise force vectors
        self.Fx = np.zeros(self.N_PBC)
        self.Fy = np.zeros(self.N_PBC)
        self.Fz = np.zeros(self.N_PBC)

        # Initialise total energy
        self.E = 0 

        # Perform calculations only on nearest-neighbour
        for i in range(self.N):  
            vec, sep = self.get_sep(i, atom[i])

            V = self.potential(sep)
            self.E += V

            Fx, Fy, Fz = self.force(vec, sep)
            self.Fx[i] += Fx
            self.Fy[i] += Fy
            self.Fz[i] += Fz

        self.E /= 2
        # Only report forces for atoms in the central cell
        self.F = np.sqrt(self.Fx[:self.N]**2 + self.Fy[:self.N]**2 + self.Fz[:self.N]**2)

    def track_seps(self, folder):
        """Track the separations of atoms in the trajectory"""
        fname = "separation.txt"
        fpath = f"{folder}/{fname}"
        self.separations = [[] for i in range(self.N_PBC)]

        for i in range(self.N):  # atoms in the central cell
            for j in range(self.N_PBC):  
                _, sep = self.get_sep(i, j)
                if (sep <= r_c) and (i != j):
                    self.separations[i].append(sep)

                    
        data = []
        for i in range(self.N):
            line = f"{i+1}," + ",".join(f"{sep}" for sep in self.separations[i]) + "\n"
            data.append(line)

        with open(fpath, "w") as f:
            f.write("Atom,Separations (A)\n")
            for line in data:
                f.write(line)
    
    def show_V_F(self, j):
        """Display the total energy and force on atom j"""
        print(f"Total energy (eV): {self.E} \n")
        print(f"Force (eV / Angstrom): x = {self.Fx[j]}, y = {self.Fy[j]}, z = {self.Fz[j]}")


    def potential(self, x):
        """Calculate potential energy between two hydrogen atoms"""
        return cutoff(x, func_V(x))
    
    
    def force(self, v, x):
        """Calculate force on a single hydrogen atom and decompose into cartesian components"""
        if x == 0:
            return 0.0, 0.0, 0.0  # avoid division by zero
        
        F = force(x)  # scalar force magnitude
        
        unit_v = v / x
        Fx = F * unit_v[0]
        Fy = F * unit_v[1]
        Fz = F * unit_v[2]

        return Fx, Fy, Fz


    def show_p(self):
        """Display the positions of all H2 molecules"""
        for i in range(self.N):
            print(f"x1[{i}] = {self.X[i]}, y1[{i}] = {self.Y[i]}, z1[{i}] = {self.Z[i]} \n")
            print(f"x2[{i}] = {self.X[i+self.N]}, y2[{i}] = {self.Y[i+self.N]}, z2[{i}] = {self.Z[i+self.N]} \n")
            

    def get_sep(self, j, k):
        """Return the separation vector and magnitude between two hydrogen atoms - remember correct order for correct sign: looking for atom j"""
        v = np.array([self.X[j], self.Y[j], self.Z[j]]) - np.array([self.X[k], self.Y[k], self.Z[k]])
        norm = np.linalg.norm(v)
        return v, norm


    def save_to_file(self, folder):
        """Save position, force, and total energy data to txt file"""
        file = input("Enter filename to save results: ")
        filename = f"{folder}/{file}.txt"

        with open(filename, "w") as f:
            f.write(f"Total energy: {self.E} eV\n")

            f.write("Position Data\n")
            for i in range(self.N):
                f.write(f"H {self.X[i]} {self.Y[i]} {self.Z[i]}\n")

            f.write("Force Data (eV/Angstrom)|  x   y   z\n")
            for i in range(self.N):
                f.write(f"{self.Fx[i]} {self.Fy[i]} {self.Fz[i]}\n")


if __name__ == "__main__":
    file_home = ""
    file_new_home = ""
    f_path = ""
    folder = ""
    print("Write some code to use the functions")
    
