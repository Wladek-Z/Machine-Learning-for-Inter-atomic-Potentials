# Machine-Learning-for-Inter-atomic-Potentials
Select python programmes used to 1) calculate the short-range contributions towards the total forces and energy of a hydrogen structure and 2) use the obtained numerical results to modify NNP training data obtained from DFT simulations, for further use with n2p2

The main subject of these programmes is the Molecules class (for processing .cell files obtained from CASTEP) and its subclass, M_Trajectory (for processing .data files obtained via ASE). Short-range interactions are treated by evaluating a fitted pair-potential for each hydrogen atom within a radius of r_c around every other hydrogen atom, however, this method has not been verified to work as of the conclusion to this project.

PYTHON FILES

  potential_gradient.py: code containing simulation class, able to return all the forces and energy given a .cell file

  M_Trajectory.py: code containing simulation subclass (of Molecules() in potential_gradient.py), able to return all the forces and energy, and modify data given a .data file

  separate data.py: used to separate a single input.data file containing many structures into separate input.data files containing one structure each. Useful for M_Trajectory simulation

  fit_potentials.py: code containing many different attempts at fitting a pair-potential to the CASTEP data, for a single hydrogen molecule stretched along the x-axis

  tabulate_FE.py: given a directory containing folders containing .castep files, where each folder follows the naming convention "sep_###" (where ### is a floating point number representing the hydrogen covalent bond length), this code creates a txt file containing the atomic separation and associated Final Energy

  new-H2.ipynb: Jupyter notebook used to generate the NNP symmetry functions, developed by Yu Cai. This file has been modified for this project by changing the short-range/long-range interaction border to 1.5 Angstroms, in-line with the calculation cut-off used throughout the project
