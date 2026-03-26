"""
saha.py
-------
Computes the ionization fraction x_e(z) using the Saha equation.

x_e = fraction of hydrogen that is ionized (free electrons / total hydrogen)
  x_e = 1  --> fully ionized plasma (early universe, high temperature)
  x_e = 0  --> fully neutral hydrogen (late universe, low temperature)

We solve this as a function of redshift z, where:
  z = 0    --> today
  z = 1100 --> recombination epoch (what we care about)
  z = 1500 --> still mostly ionized plasma
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Physical constants (SI units) ──────────────────────────────────────────
k_B   = 1.380649e-23    # Boltzmann constant       [J/K]
hbar  = 1.054571817e-34 # Reduced Planck constant  [J·s]
m_e   = 9.10938e-31     # Electron mass            [kg]
eV    = 1.60218e-19     # 1 electron-volt in Joules
E_I   = 13.6 * eV       # Ionization energy of hydrogen [J]
c     = 3.0e8           # Speed of light           [m/s]

# ── Cosmological parameters ────────────────────────────────────────────────
# These describe our universe (Planck 2018 values)
H0       = 67.4 * 1e3 / 3.086e22   # Hubble constant today  [1/s]  (67.4 km/s/Mpc)
Omega_b  = 0.0224 / (0.674**2)     # Baryon density parameter
T0       = 2.725                    # CMB temperature today  [K]
rho_crit = 3 * H0**2 / (8 * np.pi * 6.674e-11)  # Critical density [kg/m^3]

# ── Derived: number density of hydrogen today ──────────────────────────────
# We assume all baryons are hydrogen (good approximation for recombination)
m_H = 1.6726e-27  # Proton mass [kg]
n_H0 = (Omega_b * rho_crit) / m_H   # Hydrogen number density today [1/m^3]


def temperature(z):
    """
    CMB temperature at redshift z.
    The universe cools as it expands: T scales as (1+z).
    """
    return T0 * (1 + z)


def hydrogen_number_density(z):
    """
    Number density of hydrogen nuclei at redshift z.
    As the universe expands, volume grows as (1+z)^3, so density falls.
    """
    return n_H0 * (1 + z)**3


def saha_rhs(z):
    """
    Computes the right-hand side of the Saha equation:

        x_e^2 / (1 - x_e) = RHS

    Once we know RHS, we solve the quadratic for x_e.
    """
    T   = temperature(z)
    n_H = hydrogen_number_density(z)

    # The thermal de Broglie factor: (m_e k_B T / 2 pi hbar^2)^(3/2)
    thermal_factor = (m_e * k_B * T / (2 * np.pi * hbar**2))**1.5

    # Boltzmann suppression: exp(-E_I / k_B T)
    boltzmann      = np.exp(-E_I / (k_B * T))

    return (thermal_factor * boltzmann) / n_H


def solve_xe(z):
    """
    Solves the Saha equation for x_e at a given redshift z.

    Starting from:    x_e^2 / (1 - x_e) = R

    Rearranging:      x_e^2 + R * x_e - R = 0

    Quadratic formula gives:
                      x_e = [-R + sqrt(R^2 + 4R)] / 2

    We take the positive root since x_e must be between 0 and 1.
    """
    R  = saha_rhs(z)
    xe = (-R + np.sqrt(R**2 + 4 * R)) / 2
    return xe


# ── Main computation ───────────────────────────────────────────────────────
# Define a range of redshifts from z=500 (well after recombination)
# to z=1700 (well before recombination, fully ionized)
z_values  = np.linspace(500, 1700, 1000)
xe_values = np.array([solve_xe(z) for z in z_values])


# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(z_values, xe_values, color='royalblue', linewidth=2)

# Mark the recombination redshift (where x_e ≈ 0.5)
ax.axvline(x=1100, color='tomato', linestyle='--', linewidth=1.5,
           label='z ≈ 1100  (recombination)')
ax.axhline(y=0.5,  color='gray',  linestyle=':',  linewidth=1,
           label='$x_e = 0.5$')

ax.set_xlabel('Redshift  z', fontsize=13)
ax.set_ylabel('Ionization fraction  $x_e$', fontsize=13)
ax.set_title('Ionization History of Hydrogen\n(Saha Equation)', fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(500, 1700)
ax.set_ylim(-0.05, 1.05)
ax.invert_xaxis()   # Redshift increases to the left = going back in time

# Annotate the two regimes
ax.text(600,  0.08, 'Neutral\n(post-recombination)',
        fontsize=10, color='steelblue', ha='center')
ax.text(1550, 0.92, 'Ionized plasma\n(pre-recombination)',
        fontsize=10, color='steelblue', ha='center')

ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cmb-decoupling/ionization_history.png', dpi=150)
plt.show()
print("Plot saved!")

# ── Quick sanity check: print x_e at a few key redshifts ──────────────────
print("\nIonization fraction at key redshifts:")
print(f"  z = 1500  →  x_e = {solve_xe(1500):.4f}  (should be ≈ 1, fully ionized)")
print(f"  z = 1100  →  x_e = {solve_xe(1100):.4f}  (transition point)")
print(f"  z =  800  →  x_e = {solve_xe(800):.6f}  (should be ≈ 0, mostly neutral)")
