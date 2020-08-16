import numpy as np
cimport numpy as np
from libc.math cimport sqrt
#cython: boundscheck=False, wraparound=False, nonecheck=False

"""
Defines 3-vectors and their operations.
"""

cdef class vector3:
    """
    A 3-vector with real x, y, z components.
    """
    cdef public double complex x
    cdef public double complex y
    cdef public double complex z

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    cpdef double c_dot(self, v):
        return self.x*v.x + self.y*v.y + self.z*v.z

    dpdef double c_norm(self):
        return sqrt(self.c_dot(self))

    dpdef double c_squrenorm(self):
        return self.c_dot(self)
