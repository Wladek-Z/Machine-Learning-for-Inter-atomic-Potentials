import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

rc = 2.5
rc_i = 2
cutoff_E = -25.068 # eV, data from H2.CASTEP corresponding to a hydrogen molecule with infinite bond length

# Load the data
folder1 = "" # Directory containing a txt file with information about the bond length and associated energy, as given by CASTEP, for a single hydrogen molecule
data = pd.read_csv(f"{folder1}/energies.txt", ",")
seps_all = data["Separation (Angstrom)"].values
energs_all = data["Final Energy (eV)"].values
thresh_min = 0.4 # Lower threshold for filtering separations
thresh_max = 2 # Upper threshold for filtering separations
seps = seps_all[(seps_all >= thresh_min) & (seps_all <= thresh_max)]  # Filter out separations less than some threshold
energs = energs_all[(seps_all >= thresh_min) & (seps_all <= thresh_max)]  # Corresponding energies

def morse_potential(x, D_e, B, r_e, v):
    """Morse potential function."""
    return D_e * (np.exp(-2 * B * (x - r_e)) - 2 * np.exp(-B * (x - r_e))) + v

def extendedLJ_potential(x, De, re, a, b, c, d):
    """Extended Lennard-Jones potential function."""
    return De * (1 - (re / x)**(a + b * (re / x) + c * (re / x)**2))**2 + d

def extendedLJ_potential2(x, A, B, C, K, D, I, G, J, M, F, L, E):
    """Extended Lennard-Jones potential function."""
    return A + B / x**1 + C / x**2 + K / x**3 + D / x**4 + I / x**5 + G / x**6 + J / x**7 + M / x**8 + F / x**9 + L / x**10 + E / x**12

def ELJ_potentialHaji(x, D_e, q, B_0, B_1, B_2, r_e, y_0):
    """Extended Lennard-Jones potential function using Hajigeorgiou's method."""
    r = r_e / x
    z = (r - r_e) / (r + r_e)
    q = int(2 * q)

    J = (r - r_e) / (z**(q) * r + r_e)
    n_r = B_0 + B_1 * J + B_2 * J**2

    return D_e * (1 - r**n_r)**2 + y_0

def Wang_potential(x, ea, sig, R, mu, nu):
    """LJ potential from Wang et al."""
    return ea * ((sig / x)**(2 * round(mu, 0)) - 1) * ((R / x)**(2 * round(mu, 0)) - 1)**(2 * round(nu, 0)) + cutoff_E


def fit_morse(x, y):
    """Fit the Morse Potential to the data"""
    p0 = [1, 1, 3, 0]  # Initial guess for parameters: [D_e, B, r_e, v]
    popt, pcov = curve_fit(morse_potential, x, y, p0=p0, bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
    err = np.sqrt(np.diag(pcov))
    print(f"Fitted parameters: D_e = {popt[0]:.4f} ± {err[0]:.4f}, B = {popt[1]:.4f} ± {err[1]:.4f}, r_e = {popt[2]:.4f} ± {err[2]:.4f}, v = {popt[3]:.4f} ± {err[3]:.4f}")

    # Generate data for the fitted curve
    x_fit = np.linspace(min(x), max(x), 250)
    y_fit = morse_potential(x_fit, *popt)

    return x_fit, y_fit

def fit_extendedLJ(x, y):
    """Fit the Extended Lennard-Jones Potential to the data"""
    numparams = 12
    p0 = np.ones(numparams)  # Initial guess for parameters: [v, D_e, A, a, ...]
    popt, pcov = curve_fit(extendedLJ_potential2, x, y, p0=p0, bounds=([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0], 
        np.inf*np.ones(numparams)))
    err = np.sqrt(np.diag(pcov))

    return popt, err

def plot_morse():
    x_fit, y_fit = fit_morse(seps, energs)

    plt.plot(seps, energs, marker='.', linestyle='none', color='indigo', label='Data')
    plt.plot(x_fit, y_fit, color='orange', label='Morse')
    plt.title("Potential Energy Curve of H2 Molecule")
    plt.xlabel(r"Separation [$\AA$]")
    plt.ylabel("Final Energy [eV]")
    plt.legend()
    plt.show()

def plot_extended_LJ():
    popt, err = fit_extendedLJ(seps, energs)

    print(f"Fitted parameters:")
    for i in range(int(len(popt))):
        print(f"C_{i+1} = {popt[i]:.8e}")

    # Generate data for the fitted curve
    y_fit = extendedLJ_potential2(seps, *popt)

    sort = np.argsort(seps)
    seps_s = seps[sort]
    y_fit = y_fit[sort]
    energs_s = energs[sort]

    plt.plot(seps_s, energs_s, marker='.', linestyle='none', color='indigo', label='Data')
    plt.plot(seps_s, y_fit, color='orange', label='Extended LJ')
    plt.title("Potential Energy Curve of H2 Molecule")
    plt.xlabel(r"Separation [$\AA$]")
    plt.ylabel("Final Energy [eV]")
    plt.legend()
    plt.show()
    
    residuals = energs_s - y_fit

    plt.plot(seps_s, residuals, marker='.', linestyle='none', color='red')
    plt.title("Residuals of Linear Regression for ELJ Potential")
    plt.xlabel(r"Separation [$\AA$]")
    plt.ylabel("Residuals [eV]")
    plt.axhline(0, color='black', linestyle='--')
    plt.show()
    

def ELJ_linear_regression():
    """Perform linear regression on the Extended Lennard-Jones potential."""
    N = len(seps)
    A = np.zeros((N, 10))

    for i in range(N):
        A[i, 0] = 1 
        A[i, 1] = 1 / seps[i]**1
        A[i, 2] = 1 / seps[i]**2
        A[i, 3] = 1 / seps[i]**3
        A[i, 4] = 1 / seps[i]**4
        A[i, 5] = 1 / seps[i]**5
        A[i, 6] = 1 / seps[i]**6
        A[i, 7] = 1 / seps[i]**7
        A[i, 8] = 1 / seps[i]**8
        A[i, 9] = 1 / seps[i]**12

    return lstsq(A, energs)

def plot_ELJ_LR():
    """Plot the results of the linear regression for the Extended Lennard-Jones potential."""
    coeffs = ELJ_linear_regression()[0]
    x_fit = np.linspace(0.3, 2.5, 250)
    y_fit = extendedLJ_potential2(x_fit, *coeffs)

    print("Coefficients from Linear Regression:")
    for i, coeff in enumerate(coeffs):
        print(f"C_{i},{coeff:.8e}")

    plt.plot(seps, energs, marker='.', linestyle='none', color='indigo', label='Data')
    plt.plot(x_fit, y_fit, color='orange', label='ELJ Linear Regression')
    plt.title("Potential Energy Curve of H2 Molecule")
    plt.xlabel(r"Separation [$\AA$]")
    plt.ylabel("Final Energy [eV]")
    plt.legend()
    plt.show()
    
def residuals_LR():
    """Calculate and print the residuals of the linear regression."""
    coeffs = ELJ_linear_regression()[0]
    y_fit = extendedLJ_potential2(seps, *coeffs)
    residuals = energs - y_fit

    plt.plot(seps, residuals, marker='.', linestyle='none', color='red')
    plt.title("Residuals of Linear Regression for ELJ Potential")
    plt.xlabel(r"Separation [$\AA$]")
    plt.ylabel("Residuals [eV]")
    plt.axhline(0, color='black', linestyle='--')
    plt.show()

def chi_squared_LR():
    """Calculate and print the chi-squared value for the linear regression."""
    chi2 = ELJ_linear_regression()[1]
    print(f"Chi-squared value: {chi2:.8e}")

def write_coeffs_ELJLR():
    """Write the coefficients to a file."""
    file = input("Enter the filename to save coefficients: ")
    filename = f"{folder1}/{file}.txt"
    with open(filename, "w") as f:
        coeffs = ELJ_linear_regression()[0]
        f.write("Power of x,Value\n")
        for i, coeff in enumerate(coeffs):
            f.write(f"C_{i},{coeff:.8e}\n")

def ELJ_Haji():
    """Fit the Extended Lennard-Jones potential using Hajigeorgiou's method."""
    popt, pcov = curve_fit(ELJ_potentialHaji, seps, energs, p0=[10, 0, 1, 1, 1, 1, 0], bounds=([0, 0, -np.inf, -np.inf, -np.inf, 0, -np.inf], [np.inf, 12, np.inf, np.inf, np.inf, 2.5, np.inf]))
    err = np.sqrt(np.diag(pcov))
    return popt, err

def plot_ELJ_Haji():
    """Plot the results of the Hajigeorgiou's method for the Extended Lennard-Jones potential."""
    popt, err = ELJ_Haji()

    print(f"Fitted parameters: D_e = {popt[0]:.8e} ± {err[0]:.8e}, q = {popt[1]:.8e} ± {err[1]:.8e}, B_0 = {popt[2]:.8e} ± {err[2]:.8e}, B_1 = {popt[3]:.8e} ± {err[3]:.8e}, B_2 = {popt[4]:.8e} ± {err[4]:.8e}, r_e = {popt[5]:.8e} ± {err[5]:.8e}, y_0 = {popt[6]:.8e} ± {err[6]:.8e}")

    # Generate data for the fitted curve
    x_fit = np.linspace(min(seps), max(seps), 250)
    y_fit = ELJ_potentialHaji(x_fit, *popt)

    plt.plot(seps, energs, marker='.', linestyle='none', color='indigo', label='Data')
    plt.plot(x_fit, y_fit, color='orange', label='ELJ Hajigeorgiou')
    plt.title("Potential Energy Curve of H2 Molecule")
    plt.xlabel(r"Separation [$\AA$]")
    plt.ylabel("Final Energy [eV]")
    plt.legend()
    plt.show()

def residuals_Haji():
    """Calculate and print the residuals of Hajigeorgiou's ELJ potential"""
    popt, err = ELJ_Haji()
    y_fit = ELJ_potentialHaji(seps, *popt)
    residuals = energs - y_fit

    plt.plot(seps, residuals, marker='.', linestyle='none', color='red')
    plt.title("Residuals of Hajigeorgiou's ELJ Potential")
    plt.xlabel(r"Separation [$\AA$]")
    plt.ylabel("Residuals [eV]")
    plt.axhline(0, color='black', linestyle='--')
    plt.show()

def write_coeffs_haji():
    """Write the coefficients from Hajigeorgiou's method to a file."""
    file = input("Enter the filename to save coefficients: ")
    filename = f"{folder1}/{file}.txt"
    with open(filename, "w") as f:
        popt, err = ELJ_Haji()
        f.write(f"D_e,{popt[0]:.8e}\n")
        f.write(f"q,{popt[1]:.8e}\n")
        f.write(f"B_0,{popt[2]:.8e}\n")
        f.write(f"B_1,{popt[3]:.8e}\n")
        f.write(f"B_2,{popt[4]:.8e}\n")
        f.write(f"r_e,{popt[5]:.8e}\n")
        f.write(f"y_0,{popt[6]:.8e}\n")

def find_mu_nu():
    best_chi2 = np.inf
    best_params = None
    energs_cut = n2p2_cutoff(seps)

    for mu in range(1, 13):  # Example integer range for mu
        for nu in range(1, 13):  # Example integer range for nu
            def wang_fixed(x, ea, sig, R):
                return ea * ((sig / x)**(2 * mu) - 1) * ((R / x)**(2 * mu) - 1)**(2 * nu) + cutoff_E
            try:
                popt, pcov = curve_fit(wang_fixed, seps, energs_cut, p0=[1, 1, 1], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
                residuals = energs_cut - wang_fixed(seps, *popt)
                chi2 = np.sum(residuals**2)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = (mu, nu, *popt)
                    errs = np.sqrt(np.diag(pcov))
            except Exception:
                continue

    print(f"Fitted parameters: mu = {best_params[0]}, nu = {best_params[1]}, ea = {best_params[2]:.8e} ± {errs[0]:.8e}, sig = {best_params[3]:.8e} ± {errs[1]:.8e}, R = {best_params[4]:.8e} ± {errs[2]:.8e}")

    # Generate data for the fitted curve
    x_fit = np.linspace(min(seps), max(seps), 250)
    y_fit = Wang_potential(x_fit, best_params[2], best_params[3], best_params[4], best_params[0], best_params[1])

    plt.plot(seps, energs_cut, marker='.', linestyle='none', color='indigo', label='Data')
    plt.plot(x_fit, y_fit, color='orange', label='Wang Potential')
    plt.title("Potential Energy Curve of H2 Molecule")
    plt.xlabel(r"Separation [$\AA$]")
    plt.ylabel("Final Energy [eV]")
    plt.legend()
    plt.show()

def fit_Wang():
    """Fit the Wang potential to the data."""
    # Modify the H2 PES data
    energs_cut = n2p2_cutoff(seps)

    p0 = [1, 1, 2, 8, 9]
    popt, pcov = curve_fit(Wang_potential, seps, energs_cut, p0=p0, bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, 12, 12]))
    err = np.sqrt(np.diag(pcov))

    return popt, err

def plot_Wang():
    """Plot the Wang potential."""
    popt, err = fit_Wang()

    print(f"Fitted parameters: ea = {popt[0]:.8e} ± {err[0]:.8e}, sig = {popt[1]:.8e} ± {err[1]:.8e}, R = {popt[2]:.8e} ± {err[2]:.8e}, mu = {popt[3]:.8e} ± {err[3]:.8e}, nu = {popt[4]:.8e} ± {err[4]:.8e}")

    # Generate data for the fitted curve
    x_fit = np.linspace(min(seps), max(seps), 250)
    y_fit = Wang_potential(x_fit, *popt)

    energs_cut = n2p2_cutoff(seps)

    plt.plot(seps, energs_cut, marker='.', linestyle='none', color='indigo', label='Data')
    plt.plot(x_fit, y_fit, color='orange', label='Wang Potential')
    #plt.plot(seps, energs, linestyle='none', marker = "x", color='magenta', label='Original Energies', alpha=0.25)
    plt.title("Potential Energy Curve of H2 Molecule")
    plt.xlabel(r"Separation [$\AA$]")
    plt.ylabel("Final Energy [eV]")
    plt.legend()
    plt.show()

def n2p2_cutoff(r):
    """Apply a cutoff to the potential energy (n2p2's POLY_2)"""
    x = (r - rc_i) / (rc - rc_i) 
    cutoff_func = ((15 - 6 * x) * x - 10) * x**3 + 1
    return np.where(r < rc_i, energs, np.where(r < rc, energs * cutoff_func + cutoff_E * (1 - cutoff_func), cutoff_E))

if __name__ == "__main__":
    print("Write some code to use the functions")
