# Simple Python Crank-Nicolson implementation of the 1e, 1D TDSE
# using sparse linear algebra libraries of scipy
# Code written by:
#  Maria Richter (MBI Berlin)  mrichter@mbi-berlin.de
#  Felipe Morales (MBI Berlin) morales@mbi-berlin.de
#  Misha Ivanov (MBI Berlin)   mivanov@mbi-berlin.de

# Please refer any questions to any of the authors

import scipy as sp
import numpy as np
import numpy.fft as fft
from numpy import sqrt,where,gradient,linspace
from scipy import interpolate, integrate, sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
plt.ion()
import time as timing
from matplotlib import colors as col
from matplotlib.pyplot import figure,title,xlabel,ylabel,plot,semilogy,axis,arrow,axvline
import os
import sys
import math
import os.path
import configparser
import argparse

parser = argparse.ArgumentParser(
    description='1D 1e CN TDSE code: Single calculation.')
parser.add_argument('input', metavar='input_file', type=str,
                    help='ini file with the input parameters')

args = parser.parse_args()
print("Parsing input file ", args.input)
try:
    file = open(args.input, "r")
except IOError:
    print("Error: File does not appear to exist.")
file.close()

Config = configparser.ConfigParser()
Config.read(args.input)
xMIN = Config.getfloat(" TDSE ", "xMIN")
xMAX = Config.getfloat(" TDSE ", "xMAX")
Nx = Config.getint(" TDSE ", "Nx")
E0 = Config.getfloat(" TDSE ", "E0")
omega = Config.getfloat(" TDSE ", "omega")
N_on_off = Config.getfloat(" TDSE ", "N_on_off")
N_flat = Config.getfloat(" TDSE ", "N_flat")
factor_nt = Config.getfloat(" TDSE ", "factor_nt")

def mkfilename(suffix):
    filename = os.path.join("results", os.path.splitext(args.input)[0]+suffix)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    return filename

# Some functions to make our life easier

# Convenience function to calculate the projection onto bound states


def project_into_bstates(R, wf, alleigen, indices):
    amp = np.zeros((indices.size), dtype=np.complex128)
    for i in range(indices.size):
        integrand = normalize_vector(alleigen[:, i], R)
        amp[i] = sp.integrate.simps(integrand * wf, x=R)
    return amp


# Calculate the norm of a given vector on the grid given by R
def get_norm(vector, R):
    return sp.integrate.simps(abs(vector)**2, x=R)
# Integrate vector using samples along the given axis and the composite Simpson's rule.
# R - the points at which vector is sampled.
# Simpson's rule: approximating the integral of a function using quadratic polynomials

# Normalize a given vector on the grid given by R


def normalize_vector(vector, R):
    return vector/np.sqrt(get_norm(vector, R))

# Vector potential as a function of time


def vectorpotential(E0, omega, time):
    if time <= T_on_off:
        env = (np.sin((np.pi*(time/T_on_off))/2.0))**2
    elif time <= (T_on_off + T_flat):
        env = 1.0
    elif time <= (2.0*T_on_off + T_flat):
        env = (np.sin((np.pi*((-time + T_flat)/(T_on_off)))/2))**2
    else:
        env = 0.0
    return (E0/omega) * env * np.sin(omega * time)


def envelope(E0, omega, time):
    if time <= T_on_off:
        env = (np.sin((np.pi*(time/T_on_off))/2.0))**2
    elif time <= (T_on_off + T_flat):
        env = 1.0
    elif time <= (2.0*T_on_off + T_flat):
        env = (np.sin((np.pi*((-time + T_flat)/(T_on_off)))/2))**2
    else:
        env = 0.0
    return (E0) * env

# Definitions and constants


# complex numbers i and -i
ci = complex(0.0, 1.0)
cmi = complex(0.0, -1.0)


# Constant for the soft core potential
a = 1.4142

# Spatial grid definition
x = np.linspace(xMIN, xMAX, Nx)
dx = x[1]-x[0]

print("SPATIAL GRID:: [", xMIN, ":", Nx, ":",
      xMAX, "], with dx :: ", dx, " [a.u.] ")


# Laser and temporal grid definitions

# Total times of turn off/on and flat parts
T_on_off = N_on_off*((2*np.pi)/omega)
T_flat = N_flat*((2*np.pi)/omega)
# Total time of the pulse
T_max = (2*T_on_off) + T_flat

# The integer number here is a factor of how many time points per
# atomic unit of time we want, so we can always keep similar time step
# 8.0 gives approx dt = 0.125 a.u.
Nt = int(T_max * factor_nt)
# This is our temporal grid
t = np.linspace(0.0, T_max, Nt)
dt = t[1]-t[0]

print("TEMPORAL GRID:: [", 0.0, ":", Nt, ":",
      T_max, "], with dt :: ", dt, " [a.u.] ")
print("LASER PARAMETERS :: frequency = ", omega, " field = ", E0, " [a.u.] ")

# Parameters for the Manolopoulos CAP (complex absorbing potential) [D.E. Manolopoulos, JCP 117, 9552 (2002)]
ma_c = 2.62206
ma_a = 1.0-(16.0/ma_c**3)
ma_b = (1.0-(17.0/ma_c**3))/ma_c**2
ma_delta = 0.2
ma_kmin = 0.2
ma_Emin = 0.5 * ma_kmin**2
absorbregionwidth = ma_c / (2.0 * ma_delta * ma_kmin)
rcap = xMAX - absorbregionwidth
# DEFINE CAP (complex absorbing potential) ALONG THE SPATIAL GRID
CAP = np.zeros(Nx, dtype=np.complex128)
almost_one = 1.0-np.finfo(float).eps
print("CAP PARAMETERS :: kmin =", ma_kmin,
      " width = ", absorbregionwidth, "[a.u.]")

for x_point in range(x.size):
    r = x[x_point]
    if abs(r) <= rcap:
        CAP[x_point] = complex(0.0, 0.0)
    elif abs(r) == xMAX:
        r = ma_c * almost_one
        CAP[x_point] = (ci * ma_Emin) * (ma_a * r - ma_b * (r)
                                         ** 3 + 4/((ma_c - r)**2) - 4/((ma_c + r)**2))
    else:
        r = ma_c * ((abs(r)-rcap)/(xMAX-rcap))
        CAP[x_point] = (ci * ma_Emin) * (ma_a * r - ma_b * (r)
                                         ** 3 + 4/((ma_c - r)**2) - 4/((ma_c + r)**2))

# Define CAP matrix for propagation
HCAP = sp.sparse.spdiags([CAP], [0], Nx, Nx)

# Construnction of the Hamiltionian for both the TISE and the TDSE
# Find eigenvalues eigenvectors of our Hamiltonian

# This is the soft potential that gives a ground state energy of 0.5 a.u.
V = -1.0/(np.sqrt((x**2) + (a**2)))
derivV = -x / sqrt((x**2 + 1.99996)**3)

if Config.getboolean("DUMP", "potential", fallback=False):
    data = np.transpose(np.vstack((x,V,derivV)))
    np.savetxt(mkfilename(".potential"), data)

# Kinetic energy constant
Ekin = -1.0/(2.0*(dx**2))

# We use the 5 point stencil to define our KEO
diag = np.zeros(Nx)
diag_1 = np.zeros(Nx)
diag_2 = np.zeros(Nx)

diag = -(5.0/2.0)*Ekin+V
diag_1[0:Nx] = (4.0/3.0)*Ekin
diag_2[0:Nx] = -(1.0/12.0)*Ekin

# Return a sparse matrix from diagonals.diag_1 - first upper and lower diagonal, diag - main diagonal. (NR, NR) - shape of result.
H = sp.sparse.spdiags([diag_2, diag_1, diag, diag_1,
                       diag_2], [-2, -1, 0, 1, 2], Nx, Nx)

# If the eigenvectors and values for this box size are calculated just load them
# if not recalculate
filename_vec = "eigen/eigenvec.%d.%f.%f.npy" % (Nx, xMIN, xMAX)
filename_val = "eigen/eigenva.%d.%f.%f.npy" % (Nx, xMIN, xMAX)

if os.path.isfile(filename_vec) and os.path.isfile(filename_val):
    print("Loading eigenvectors and eigenvalues for the potential")
else:
    print("Calculating eigenvectors and eigenvalues for the potential")
# Find k eigenvalues and eigenvectors of the square matrix H with smallest magnitude (which='SR').
    [eigenva, eigenvec] = sp.sparse.linalg.eigs(H, k=40, which='SR')
    np.save(filename_vec, eigenvec)
    np.save(filename_val, eigenva)

eigenvec = np.load(filename_vec)
eigenva = np.load(filename_val)

# Renormalize them, and flip such that first peak is positive
for i in range(eigenva.size):
    eigenvec[:, i] = normalize_vector(eigenvec[:, i], x)
    v = eigenvec[:, i]
    jj = next(filter(lambda i: abs(v[i])>1e-3, range(len(v))))
    eigenvec[:, i] *= np.sign(np.real(v[jj]))

# Pick our initial wavefunction (0 is the ground state)

initial_state = 0
u = eigenvec[:, initial_state]
initial = u.copy()
print("INITIAL STATE :: which =", initial_state,
      " energy = ", eigenva[initial_state].real, "[a.u.]")

# Setup a few arrays for the propagation
I = sp.sparse.identity(Nx)
diaglaser = np.zeros(Nx)
dipole = np.zeros(Nt)
dipole_acc = np.zeros(Nt)
dipole_acc2 = np.zeros(Nt)

electric_field = np.zeros(Nt)
vector_potential = np.zeros(Nt)

A1 = (I + ci*(dt/2.0)*H - ci*(dt/2.0)*HCAP)
b1 = (I - ci*(dt/2.0)*H + ci*(dt/2.0)*HCAP)

# Lets select the bound states only to project later on
indices_bound_states = where(eigenva < 0.0)[0]
how_many_bound = indices_bound_states.size

# Now we can propagate
print("Propagating")

for time_step in range(t.size):
    time = t[time_step]
    if (time_step == 0) or (time_step == t.size-1):
        field = 0.0
# calculate the electric field from the vector potential, every time step
    else:
        field = (-1.0/dt) * (vectorpotential(E0, omega, time+dt) -
                             vectorpotential(E0, omega, time))
    diaglaser = x * field
    Hlaser = sp.sparse.spdiags([diaglaser], [0], Nx, Nx)
# Recalculate A and b for advancing a time step
    A = (A1 + ci*(dt/2.0)*Hlaser)
    b = (b1 - ci*(dt/2.0)*Hlaser)*u
# Solve the sparse linear system Au=b = > Advance our solution dt
    u = sp.sparse.linalg.spsolve(A, b)

# Calculate the dipole length
    dipole[time_step] = sp.integrate.simps((abs(u)**2)*x, x=x)

# Calculate the dipole acceleration (converges much faster)
    dipole_acc[time_step] = sp.integrate.simps(
        abs(u)**2 * gradient(V, dx), x=x) - field
    dipole_acc2[time_step] = sp.integrate.simps(
        abs(u)**2 * -derivV, x=x) - field

# Keep the electric field and the vector potential just in case
    electric_field[time_step] = field
    vector_potential[time_step] = vectorpotential(E0, omega, time)
# Every 100 time steps plot some information of the propagation
    if (np.mod(time_step, 100) == 0):
        print("Time :: ", time, "[a.u.] , total norm ", get_norm(u, x))

amp_bound_states = np.zeros((indices_bound_states.size), dtype=np.complex128)
amp_bound_states = project_into_bstates(x, u, eigenvec, indices_bound_states)
Pbound = abs(amp_bound_states)**2

if Config.getboolean("DUMP", "bound_population", fallback=False):
    data = np.transpose(np.vstack((np.arange(0, how_many_bound)+1, Pbound**2)))
    np.savetxt(mkfilename(".bound_population"), data)

if Config.getboolean("DUMP", "bound_states", fallback=False):
    data = np.transpose(np.vstack((x, np.transpose(np.real(eigenvec)))))
    step = Config.getint("DUMP", "bound_xstep", fallback=1)
    np.savetxt(mkfilename(".bound_states"), data[0:-1:step])

total_bs_population = sum(abs(amp_bound_states[indices_bound_states])**2)
print("Bound states population :: ", total_bs_population)
print("Continuum population :: ", 1.0 - total_bs_population)

# Now we plot nice figures
figure(1)
title("Electric field")
xlabel("time [a.u.]")
ylabel("Efield [a.u.]")
plot(t, electric_field)
figure(2)
title("Bound state population (excluding g.s.)")
plot(np.arange(1, how_many_bound)+1, Pbound[1:]**2, "o-")
xlabel("Bound state number")
ylabel("Population")
figure(3)
title("Dipole (length form)")
plot(t, dipole)
xlabel("time [a.u.]")
ylabel("Dipole")

figure(4)
title("Dipole (acc form)")
plot(t, dipole_acc)
xlabel("time [a.u.]")
ylabel("Dipole")
# PES

# First we obtain the wavefunction without the contribution of the boundstates
wf_no_bound = u.copy()
for i in range(how_many_bound):
    wf_no_bound -= amp_bound_states[i] * eigenvec[:, i]

p = fft.fftshift(fft.fftfreq(int(Nx), d=dx)*(2.0*np.pi))
Wk = 0.5*p**2
PES = abs(fft.fftshift(fft.fft(wf_no_bound)))**2

if Config.getboolean("DUMP", "PES", fallback=False):
    step = Config.getint("DUMP", "PESstep", fallback=1)
    data = np.transpose(np.vstack((p, Wk, PES)))
    np.savetxt(mkfilename(".pes"), data[0:-1:step])

figure(5)
Up = E0**2/(4.0*omega**2)
print("Ponderomotive energy :: ", Up, " Ha")
# 2 Up - 10 Up limits

title("Spectra (FFT of wavefunction)")
# And we can FFT it to get the spectra
semilogy(p, PES)
xlabel("Momentum [a.u.]")
axis_old = axis()
axis([0, 4.0, axis_old[2], axis_old[3]])

figure(6)
# 2 Up - 10 Up limits

title("Spectra (FFT of wavefunction)")
# And we can FFT it to get the spectra
semilogy(Wk, PES)
xlabel("Energy [a.u.]")
axis_old = axis()
axis([0, 4.0, axis_old[2], axis_old[3]])

# This plot can help us see where the wavefunction (without bound states) is with respect to the box
figure(7)
title("Prob. density of the w.f.: check the borders")
plot(x, abs(wf_no_bound)**2)
xlabel("x coordinate [a.u.]")

# This is simply an energy diagram of our system, with an arrow representing the photon energy
# and the eigenstates shifted by up as a function of time
figure(8)
xforline = np.array([0, 1])
title("Eigenstate energies shifted + photon energy")

subtimearray = linspace(t.min(), t.max(), 100)
E0_here = 0.0
line = np.zeros(subtimearray.size)
for i in range(20):
    for ti in range(subtimearray.size):
        time = subtimearray[ti]
        E0_here = envelope(E0, omega, time)
        Up_here = E0_here**2/(4.0*omega**2)
        line[ti] = Up_here + eigenva[i].real
        # print time,E0_here,Up_here

    plot(subtimearray, line)
arrow(0.5*(t.max() - t.min()),
      eigenva[0].real, 0.0, omega, length_includes_head=True, lw=3, ec='b', fc='b')
xlabel("Time [a.u.]")
ylabel("Energy [a.u.]")

# We can also look at the HHG spectra, by looking at the FFT of the dipole
figure(9)
Up = E0**2/(4.0*omega**2)
Ip = abs(eigenva[0].real)
Cutoff = Ip + 3.17 * Up
title("HHG spectra")
energy = fft.fftshift(fft.fftfreq(int(Nt), d=dt)*(2.0*np.pi))
hhg_spectra = abs(fft.fftshift(fft.fft(dipole_acc)))**2
hhg_spectra_l = abs(fft.fftshift(fft.fft(dipole)))**2
xlabel("Energy (units of omega)")
semilogy(energy/omega, hhg_spectra)
axvline(Ip/omega, color="k", lw=3)
axvline(Cutoff/omega, color="k", lw=3)
axis_old = axis()
axis([0, 60, axis_old[2], axis_old[3]])

if Config.getboolean("DUMP", "HHG", fallback=False):
    data = np.transpose(np.vstack((energy, hhg_spectra, hhg_spectra_l)))
    np.savetxt(mkfilename(".hhg"), data)

figure(10)
title("Pulse spectrum")
pulse_spectra = abs(fft.fftshift(fft.fft(electric_field)))**2
semilogy(energy, pulse_spectra)
axis_old = axis()
axis([omega-2.0*omega, omega+2.0*omega, axis_old[2], axis_old[3]])
