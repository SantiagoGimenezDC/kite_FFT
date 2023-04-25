""" Density of states of pristine graphene

    ##########################################################################
    #                         Copyright 2022, KITE                           #
    #                         Home page: quantum-kite.com                    #
    ##########################################################################

    Units: Energy in eV
    Lattice: Honeycomb
    Configuration: Periodic boundary conditions, double precision,
                    automatic rescaling, size of the system 512x512, with domain decomposition (nx=ny=2)
    Calculation type: Average DOS
    Last updated: 28/07/2022
"""

__all__ = ["main"]

import kite
import numpy as np
import pybinding as pb
from pybinding.repository.graphene import a_cc, a
from pybinding.constants import hbar
from math import sqrt, pi

el_charge = 1.602 * 10 ** -19  #: [C] electron charge
t = -2.506  #: [eV] graphene nearest neighbor hopping #ORIGINAL: -2.506 in 2048x2048
vf = 1.5 * a_cc * np.abs(t) / hbar * 10 ** -9  #: [m/s] Fermi velocity

lambda_R = 0.3*t  #: [eV] Rashba SOC in units of t
lambda_I = 0.0  #: [eV] Intrinsic SOC in units of t
exch = 0.4*t   #: [eV] Exchange in units of t
# allow sublatice dependent intrinsic SOC (spin valley)
lambda_I_A = lambda_I
lambda_I_B = lambda_I

rashba_so = lambda_R * 2.0 * 1.0j / 3.0  #: [eV] constant and geometrical factors that will define Rashba SOC

km_so = lambda_I * 1j / (3 * sqrt(3))  #: [eV] constant and geometrical factors that will define intrinsic SOC

km_so_A = lambda_I_A * 1j / (3 * sqrt(3))  #: [eV] constant and geometrical factors that will define intrinsic SOC
km_so_B = lambda_I_B * 1j / (3 * sqrt(3))  #: [eV] constant and geometrical factors that will define intrinsic SOC

a1 = np.array([+a * sqrt(3) / 2, +a / 2, 0])  #: [nm] unit cell vectors graphene
a2 = np.array([+a * sqrt(3) / 2, -a / 2, 0])

posA = np.array([-a_cc / 2, 0])
posB = np.array([+a_cc / 2, 0])

# delta1 = [acc, 0]
# delta2 = [-0.5 * acc, +sqrt(3) / 2 * acc]
# delta3 = [-0.5 * acc, -sqrt(3) / 2 * acc]

def honeycomb_lattice_with_SO(onsite=(0, 0)):
    """Make a honeycomb lattice with nearest neighbor hopping with SOC

    Parameters
    ----------
    onsite : tuple or list
        Onsite energy at different sublattices.
    """

    # create a lattice with 2 primitive vectors
    lat = pb.Lattice(
        a1=a1, a2=a2
    )

    # Add sublattices
    lat.add_sublattices(
        # name, position, and onsite potential
        ('Aup', posA, onsite[0]+exch),
        ('Bup', posB, onsite[1]+exch),
        ('Adown', posA, onsite[0]-exch),
        ('Bdown', posB, onsite[1]-exch)
    )

    # Add hoppings-
    lat.add_hoppings(
        # ([f - i ], i , f )
        # inside the main cell, between which atoms, and the value
        ([0, 0], 'Adown', 'Bdown', t),
        ([0, 0], 'Aup', 'Bup', t),
        # between neighboring cells, between which atoms, and the value
        ([0, -1], 'Aup', 'Bup', t),
        ([0, -1], 'Adown', 'Bdown', t),

        ([-1, 0], 'Aup', 'Bup', t),
        ([-1, 0], 'Adown', 'Bdown', t)
    )

    if np.abs(lambda_R) > 0:
        lat.add_hoppings(
            # Rashba nearest neighbor, spin flip
            # inside the main cell, between which atoms, and the value
            ([0, 0], 'Aup', 'Bdown', 1j * rashba_so),  # delta1
            ([0, -1], 'Aup', 'Bdown', (+sqrt(3) / 2 - 0.5 * 1j) * rashba_so),  # delta2
            ([-1, 0], 'Aup', 'Bdown', (-sqrt(3) / 2 - 0.5 * 1j) * rashba_so),  # delta3

            ([0, 0], 'Adown', 'Bup', -1j * rashba_so),  # delta1
            ([0, -1], 'Adown', 'Bup', (+sqrt(3) / 2 + 0.5 * 1j) * rashba_so),  # delta2
            ([-1, 0], 'Adown', 'Bup', (-sqrt(3) / 2 + 0.5 * 1j) * rashba_so)  # delta3
        )

    if np.abs(lambda_I) > 0:
        # Kane-Mele SOC, same spin next-nearest
        # between neighboring cells, between which atoms, and the value
        lat.add_hoppings(
            ([0, -1], 'Aup', 'Aup', +km_so_A),
            ([0, -1], 'Adown', 'Adown', -km_so_A),

            ([-1, 0], 'Aup', 'Aup', -km_so_A),
            ([-1, 0], 'Adown', 'Adown', +km_so_A),

            ([1, -1], 'Aup', 'Aup', -km_so_A),
            ([1, -1], 'Adown', 'Adown', +km_so_A),

            ([0, -1], 'Bup', 'Bup', -km_so_B),
            ([0, -1], 'Bdown', 'Bdown', +km_so_B),

            ([-1, 0], 'Bup', 'Bup', +km_so_B),
            ([-1, 0], 'Bdown', 'Bdown', -km_so_B),

            ([1, -1], 'Bup', 'Bup', +km_so_B),
            ([1, -1], 'Bdown', 'Bdown', -km_so_B),
        )

    return lat




def graphene_lattice(onsite=(0, 0)):
    """Return lattice specification for a honeycomb lattice with nearest neighbor hoppings"""

    # parameters
    a = 0.24595  # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = 2.7  # eV

    # define lattice vectors
    a1 = a * np.array([1, 0])
    a2 = a * np.array([1 / 2, 1 / 2 * np.sqrt(3)])

    # create a lattice with 2 primitive vectors
    lat = pb.Lattice(a1=a1, a2=a2)

    # add sublattices
    lat.add_sublattices(
        # name, position, and onsite potential
        ('A', [0, -a_cc/2], onsite[0]),
        ('B', [0,  a_cc/2], onsite[1])
    )

    # Add hoppings
    lat.add_hoppings(
        # inside the main cell, between which atoms, and the value
        ([0, 0], 'A', 'B', -t),
        # between neighboring cells, between which atoms, and the value
        ([1, -1], 'A', 'B', -t),
        ([0, -1], 'A', 'B', -t)
    )

    return lat


def main(onsite=(0, 0)):
    """Prepare the input file for KITEx"""
    # load lattice
    lattice = graphene_lattice(onsite)

    #lattice = honeycomb_lattice_with_SO()

    # number of decomposition parts [nx,ny] in each direction of matrix.
    # This divides the lattice into various sections, each of which is calculated in parallel
    nx = 4
    ny = 8
    # number of unit cells in each direction.
    lx = 160
    ly = 320

    # make config object which caries info about
    # - the number of decomposition parts [nx, ny],
    # - lengths of structure [lx, ly]
    # - boundary conditions [mode,mode, ... ] with modes:
    #   . "periodic"
    #   . "open"
    #   . "twisted" -- this option needs the extra argument angles=[phi_1,..,phi_DIM] where phi_i \in [0, 2*M_PI]
    #   . "random"

    # Boundary Mode
    mode = "periodic"
    # add Disorder

    #disorder = kite.Disorder(lattice)
    #disorder.add_disorder(['Aup', 'Adown'], 'Uniform', +0.0, 0.1)
    #disorder.add_disorder(['Bup', 'Bdown'], 'Uniform', +0.0, 0.1)
    
    
    disorder = kite.Disorder(lattice)
    disorder.add_disorder('A', 'Uniform', +0.0, 0.135)
    disorder.add_disorder('B', 'Uniform', +0.0, 0.135)
    
    # - specify precision of the exported hopping and onsite data, 0 - float, 1 - double, and 2 - long double.
    # - scaling, if None it's automatic, if present select spectrum_range=[e_min, e_max]
    configuration = kite.Configuration(
        divisions=[nx, ny],
        length=[lx, ly],
        boundaries=[mode, mode],
        is_complex=False,
        precision=1,
        spectrum_range=[-9,9]
    )

    # specify calculation type
    calculation = kite.Calculation(configuration)

    calculation.conductivity_dc_FFT(
        num_points=10,
        num_moments=1000,
        num_random=3,
        num_disorder=3,
        direction="xy",
        temperature=0.00,
        num_reps=1,
        kernel=1
    )
    
    modification = kite.Modification(magnetic_field=250)

    # configure the *.h5 file
    output_file = "mag_test_l.h5"
    kite.config_system(lattice, configuration, calculation, modification, filename=output_file, disorder=disorder)

    # for generating the desired output from the generated HDF5-file, run
    # ../build/KITEx graphene_lattice-output.h5
    # ../tools/build/KITE-tools graphene_lattice-output.h5

    # returning the name of the created HDF5-file
    return output_file


if __name__ == "__main__":
    main()
