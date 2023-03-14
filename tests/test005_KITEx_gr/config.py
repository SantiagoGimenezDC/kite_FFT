import numpy as np
import pybinding as pb
import kite


def graphene(onsite=(0, 0)):
    theta = np.pi / 3
    t = 1  # eV
    a1 = np.array([1 + np.cos(theta), np.sin(theta)])
    a2 = np.array([0, 2 * np.sin(theta)])
    lat = pb.Lattice( a1=a1, a2=a2)

    lat.add_sublattices(
        ('A', [0, 0], onsite[0]),
        ('B', [1, 0], onsite[1])
    )
    lat.add_hoppings(
        ([0, 0], 'A', 'B', - t),
        ([-1, 0], 'A', 'B', - t),
        ([-1, 1], 'A', 'B', - t)
    )

    return lat


lattice = graphene()
nx = ny = 2
lx = ly = 64
configuration = kite.Configuration(divisions=[nx, ny], length=[lx, ly], boundaries=["periodic", "periodic"], is_complex=False, precision=1, spectrum_range=[-3.1, 3.1])
calculation = kite.Calculation(configuration)
calculation.conductivity_optical(num_points=1024, num_disorder=1, num_random=1, num_moments=32, direction='yy')
kite.config_system(lattice, configuration, calculation, filename='config.h5')
