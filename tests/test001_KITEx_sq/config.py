import numpy as np
import pybinding as pb
import kite


def square():
    a1 = np.array([1, 0])
    a2 = np.array([0, 1])
    lat = pb.Lattice( a1=a1, a2=a2)
    lat.add_sublattices( ('A', [0, 0], 0))
    lat.add_hoppings(
        ([0, 1], 'A','A', 1),
        ([1, 0], 'A','A', 1)
    )

    return lat


lattice = square()
nx = ny = 2
lx = ly = 64
configuration = kite.Configuration(divisions=[nx, ny], length=[lx, ly], boundaries=["periodic", "periodic"], is_complex=False, precision=1, spectrum_range=[-5, 5])
calculation = kite.Calculation(configuration)
calculation.dos(num_points=1000, num_moments=64, num_random=1, num_disorder=1)
kite.config_system(lattice, configuration, calculation, filename='config.h5')
