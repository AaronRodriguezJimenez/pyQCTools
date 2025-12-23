""" Convenience functions to construct common small-molecule geometries.

These geometries can go straight into the `atom` argument for `pyscf.gto.M`,
    to construct pyscf molecules.
The format is a list of ordered pairs,
    where the first index is the chemical symbol as a string,
    and the second index is a 3-tuple `(x,y,z)`
    giving the Cartesian coordinates of the atomic nucleus.

Code taken from Kyle Sherbert's PySCFWorkspace.
"""

import numpy

def HChain(n, d):
    """ A uniformly-spaced chain of `n` hydrogen atoms with bond distance `d`. """
    return [('H', (0, 0, i*d)) for i in range(n)]

def HCycle(n, d):
    """ A uniformly-spaced ring of `n` hydrogen atoms with bond distance `d`. """
    π   = numpy.pi
    sin = numpy.sin
    cos = numpy.cos

    θ = 2*π / n             # ANGULAR SEPARATION BETWEEN EACH NUCLEUS
    r = d / (2 * sin(θ/2))  # RADIUS FROM CENTER FOR EACH NUCLEUS
    return [('H', (r*cos(i*θ), r*sin(i*θ), 0)) for i in range(n)]

def H4Hedron(d):
    """ A regular tetrahedron of hydrogen atoms with bond distance `d`. """
    r = d / numpy.sqrt(2)       # EDGE-LENGTH OF CUBE CIRCUMSCRIBING THE TETRAHEDRON
    return [
        ('H', ( 0, 0, 0)),
        ('H', ( 0, r, r)),
        ('H', ( r, 0, r)),
        ('H', ( r, r, 0)),
    ]

def H6Hedron(d):
    """ A regular octehedron of hydrogen atoms with bond distance `d`. """
    r = d / numpy.sqrt(2)       # RADIUS OF SPHERE CIRCUMSCRIBING THE OCTAHEDRON
    return [
        ('H', ( 0, 0, r)),
        ('H', ( r, 0, 0)),
        ('H', ( 0, r, 0)),
        ('H', (-r, 0, 0)),
        ('H', ( 0,-r, 0)),
        ('H', ( 0, 0,-r)),
    ]

def HeH(d):
    """ A proton and an α particle separated by distance `d`. """
    return [
        ('He',(0, 0, 0)),
        ('H', (0, 0, d)),
    ]

##########################################################################################
#   SOME VAGUELY REALISTIC MOLECULES

def H2(d=0.7414):
    """ Diatomic molecular hydrogen with bond distance `d`.

    Defaults give experimental geometry as taken from NIST CCCBDB.

    """
    return [
        ('H', (0, 0, 0)),
        ('H', (0, 0, d)),
    ]

def N2(d=1.098):
    """ Diatomic molecular hydrogen with bond distance `d`.

    Defaults give experimental geometry as taken from NIST CCCBDB.

    """
    return [
        ('N', (0, 0, 0)),
        ('N', (0, 0, d)),
    ]

def Cr2(d=1.50):
    """ Diatomic molecular hydrogen with bond distance `d`.

    Defaults give experimental geometry as taken from NIST CCCBDB.

    """
    return [
        ('Cr', (0, 0, 0)),
        ('Cr', (0, 0, d)),
    ]

def LiH(d=1.595):
    """ Lithium hydride with bond distance `d`.

    Defaults give experimental geometry as taken from NIST CCCBDB.

    """
    return [
        ('Li',(0, 0, 0)),
        ('H', (0, 0, d)),
    ]

def BeH2(d=1.3264):
    """ Beryllium hydride with bond distance `d`.

    Defaults give experimental geometry as taken from NIST CCCBDB.

    """
    return [
        ('H', (0, 0,-d)),
        ('Be',(0, 0, 0)),
        ('H', (0, 0, d)),
    ]

def H2O(d=0.958, θ=1.823):
    """ Water with bond distance `d` and bond angle `θ`.

    Defaults give experimental geometry as taken from NIST CCCBDB.

    """
    sin = numpy.sin
    cos = numpy.cos

    return [
        ('O', (0, 0, 0)),
        ('H', (d*sin(-θ/2), 0, d*cos(-θ/2))),
        ('H', (d*sin( θ/2), 0, d*cos( θ/2))),
    ]

def ethylene():
    """
    Ethylene molecule from NIST
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C74851&Mask=4&Type=ANTOINE&Plot=on
    """
    return [
        ('C', (0.5108 ,   0.8683 ,   0.0000)),
        ('H', (0.0000 ,   1.7366 ,   0.0000)),
        ('H', (0.0000 ,   0.0000 ,   0.0000)),
        ('C', (1.4812 ,   0.8683 ,   0.0000)),
        ('H', (1.9920 ,   1.7366 ,   0.0000)),
        ('H', (1.9920,    0.0000,    0.0000))
    ]