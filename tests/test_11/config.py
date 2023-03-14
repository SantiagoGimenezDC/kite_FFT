import kite
import numpy as np
import pybinding as pb


def square_lattice(onsite=(0, 0)):
    a1 = np.array([1, 0])
    a2 = np.array([0, 1])
    lat = pb.Lattice( a1=a1, a2=a2)
    lat.add_sublattices( ('A', [0, 0], onsite[0]))
    lat.add_hoppings(
        ([1, 0], 'A', 'A', - 1),
        ([0, 1], 'A', 'A', - 1))

    return lat

lattice = square_lattice()
nx = ny = 2
lx = ly = 128

a,b = -6,6
M = 32

configuration = kite.Configuration(divisions=[nx, ny], length=[lx, ly], boundaries=["periodic", "periodic"], is_complex=False, precision=1, spectrum_range=[a,b])
calculation = kite.Calculation(configuration)
calculation.dos(num_points=1000, num_moments=M, num_random=10, num_disorder=1)
kite.config_system(lattice, configuration, calculation, filename='config.h5')
