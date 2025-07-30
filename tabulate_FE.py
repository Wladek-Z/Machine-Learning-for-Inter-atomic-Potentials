import re
import numpy as np
from pathlib import Path

def extract_FE(filename):
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(
                r'Final energy, E\s*=\s*([-+]?\d*\.?\d+(?:[DEde][+-]?\d+)?)', line)
            if match:
                value = match.group(1).replace('D', 'E').replace('d', 'E')
                return float(value)
    return None

root_folder = "" # directory containing many folders filled with CASTEP data. The folders must follow the naming convention "sep_{num}", where num is a float corresponding to the bond length of the molecule
target_filename = "H2.castep"

with open(f"{root_folder}/energies.txt", "a") as f:
    f.write("Separation (Angstrom),Final Energy (eV)\n")

    for file_path in Path(root_folder).rglob(target_filename):
        folder_name = file_path.parent.name
        match = re.match(r"sep_([-+]?\d*\.?\d+)", folder_name)

        if match:
            num = match.group(1)
            energy = extract_FE(f'{root_folder}/sep_{num}/H2.castep')
            f.write(f"{num},{energy}\n")
        
