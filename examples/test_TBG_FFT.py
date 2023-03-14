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
import matplotlib.pyplot as plt
import pybinding as pb
import math

from scipy.spatial import cKDTree
from pybinding.repository import graphene

c0 = 0.335  # [nm] graphene interlayer spacing

def two_graphene_monolayers():
    """Two individual AB stacked layers of monolayer graphene without any interlayer hopping,"""
    from pybinding.repository.graphene.constants import a_cc, a, t

    lat = pb.Lattice(a1=[a/2, a/2 * math.sqrt(3)], a2=[-a/2, a/2 * math.sqrt(3)])
    lat.add_sublattices(('A1', [0,   a_cc,   0]),
                        ('B1', [0,      0,   0]),
                        ('A2', [0,      0, -c0]),
                        ('B2', [0,  -a_cc, -c0]))
    lat.register_hopping_energies({'gamma0': t})
    lat.add_hoppings(
        # layer 1
        ([0,  0], 'A1', 'B1', 'gamma0'),
        ([0,  1], 'A1', 'B1', 'gamma0'),
        ([1,  0], 'A1', 'B1', 'gamma0'),
        # layer 2
        ([0, 0], 'A2', 'B2', 'gamma0'),
        ([0, 1], 'A2', 'B2', 'gamma0'),
        ([1, 0], 'A2', 'B2', 'gamma0'),
        # not interlayer hopping
    )
    lat.min_neighbors = 2
    return lat


def twist_layers(theta):
    """Rotate one layer and then a generate hopping between the rotated layers,
       reference is AB stacked"""
    theta = theta / 180 * math.pi  # from degrees to radians

    @pb.site_position_modifier
    def rotate(x, y, z):
        """Rotate layer 2 by the given angle `theta`"""
        layer2 = (z < 0)
        x0 = x[layer2]
        y0 = y[layer2]
        x[layer2] = x0 * math.cos(theta) - y0 * math.sin(theta)
        y[layer2] = y0 * math.cos(theta) + x0 * math.sin(theta)
        return x, y, z

    @pb.hopping_generator('interlayer', energy=0.1)  # eV
    def interlayer_generator(x, y, z):
        """Generate hoppings for site pairs which have distance `d_min < d < d_max`"""
        positions = np.stack([x, y, z], axis=1)
        layer1 = (z == 0)
        layer2 = (z != 0)

        d_min = c0 * 0.98
        d_max = c0 * 1.1
        kdtree1 = cKDTree(positions[layer1])
        kdtree2 = cKDTree(positions[layer2])
        coo = kdtree1.sparse_distance_matrix(kdtree2, d_max, output_type='coo_matrix')

        idx = coo.data > d_min
        abs_idx1 = np.flatnonzero(layer1)
        abs_idx2 = np.flatnonzero(layer2)
        row, col = abs_idx1[coo.row[idx]], abs_idx2[coo.col[idx]]
        return row, col  # lists of site indices to connect

    @pb.hopping_energy_modifier
    def interlayer_hopping_value(energy, x1, y1, z1, x2, y2, z2, hop_id):
        """Set the value of the newly generated hoppings as a function of distance"""
        d = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        interlayer = (hop_id == 'interlayer')
        energy[interlayer] = 0.4 * c0 / d[interlayer]
        return energy

    return rotate, interlayer_generator, interlayer_hopping_value



def main(onsite=(0, 0)):
    """Prepare the input file for KITEx"""
    # load lattice
    model = pb.Model(
      two_graphene_monolayers(),
      pb.rectangle(10,10),
      twist_layers(theta=21.798)
    )
    
    plt.figure(figsize=(6.5, 6.5))
    model.plot()
    plt.title(r"$\theta$ = 21.798 $\degree$")
    plt.show()


    lattice =  pb.Model(
      two_graphene_monolayers(),
      pb.circle(radius=1.5),
      twist_layers(theta=21.798)
    )

    # number of decomposition parts [nx,ny] in each direction of matrix.
    # This divides the lattice into various sections, each of which is calculated in parallel
    nx = ny = 2
    # number of unit cells in each direction.
    lx = ly = 128

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

    # - specify precision of the exported hopping and onsite data, 0 - float, 1 - double, and 2 - long double.
    # - scaling, if None it's automatic, if present select spectrum_range=[e_min, e_max]
    configuration = kite.Configuration(
        divisions=[nx, ny],
        length=[lx, ly],
        boundaries=[mode, mode],
        is_complex=True,
        precision=1,
        spectrum_range=[-9,9]
    )

    # specify calculation type
    calculation = kite.Calculation(configuration)

    calculation.conductivity_dc_FFT(
        num_points=10,
        num_moments=512,
        num_random=1,
        num_disorder=1,
        direction="xx",
        temperature=0.00,
        num_reps=1,
        kernel=1
    )
    
    #modification = kite.Modification(magnetic_field=1650)

    # configure the *.h5 file
    output_file = "test_TBG_FFT.h5"
    kite.config_system(lattice, configuration, calculation, filename=output_file)

    # for generating the desired output from the generated HDF5-file, run
    # ../build/KITEx graphene_lattice-output.h5
    # ../tools/build/KITE-tools graphene_lattice-output.h5

    # returning the name of the created HDF5-file
    return output_file


if __name__ == "__main__":
    main()
