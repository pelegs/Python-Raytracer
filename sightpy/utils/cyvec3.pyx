from __future__ import print_function
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, abs
cimport cython
#cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3

"""
Defines 3-vectors and their operations.
"""

NUMBERS = [int, float, complex, np.float64]

ctypedef fused Number:
    cython.int
    cython.float
    cython.double
    cython.complex

cdef class vec3:
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

    def __str__(self):
        # Used for debugging.
        # This method is called when you print an instance. 
        return '({},{},{})'.format(self.x, self.y, self.z)

    """
    Addition
    """
    def __add__(self, v):
        if isinstance(v, vec3):
            return self.add(v)
        elif type(v) in NUMBERS:
            return self.num_add(v)
        elif isinstance(v, np.ndarray):
            return self.ndarr_add(v)

    def __radd__(self, v):
        if isinstance(v, vec3):
            return self.add(v)
        elif type(v) in NUMBERS:
            return self.num_add(v)
        elif isinstance(v, np.ndarray):
            return self.ndarr_add(v)

    cpdef vec3 add(self, vec3 v):
        return vec3(self.x+v.x, self.y+v.y, self.z+v.z)

    cpdef vec3 num_add(self, Number v):
        return vec3(self.x+v, self.y+v, self.z+v)

    cpdef vec3 ndarr_add(self, np.ndarray v):
        return vec3(self.x+v, self.y+v, self.z+v)

    """
    Subtraction
    """
    def __sub__(self, v):
        if isinstance(v, vec3):
            return self.sub(v)
        elif type(v) in NUMBERS:
            return self.num_sub(v)
        elif isinstance(v, np.ndarray):
            return self.ndarr_sub(v)
    
    def __rsub__(self, v):
        if isinstance(v, vec3):
            return self.rsub(v)
        elif type(v) in NUMBERS:
            return self.num_rsub(v)
        elif isinstance(v, np.ndarray):
            return self.ndarr_rsub(v)

    cpdef vec3 sub(self, vec3 v):
        return vec3(self.x-v.x, self.y-v.y, self.z-v.z)

    cpdef vec3 num_sub(self, Number v):
        return vec3(self.x-v, self.y-v, self.z-v)
    
    cpdef vec3 ndarr_sub(self, np.ndarray v):
        return vec3(self.x-v, self.y-v, self.z-v)
    
    cpdef vec3 rsub(self, vec3 v):
        return vec3(v.x-self.x, v.y-self.y, v.z-self.z)

    cpdef vec3 num_rsub(self, Number v):
        return vec3(v-self.x, v-self.y, v-self.z)
    
    cpdef vec3 ndarr_rsub(self, np.ndarray v):
        return vec3(v-self.x, v-self.y, v-self.z)

    """
    Multiplication
    """
    def __mul__(self, v):
        if isinstance(v, vec3):
            self.mul(v)
        elif type(v) in NUMBERS:
            self.num_mul(v)
        elif isinstance(v, np.ndarray):
            self.ndarr_mul(v)
    
    def __rmul__(self, v):
        if isinstance(v, vec3):
            self.mul(v)
        elif type(v) in NUMBERS:
            self.num_mul(v)
        elif isinstance(v, np.ndarray):
            self.ndarr_mul(v)

    cpdef vec3 mul(self, vec3 v):
        return vec3(self.x * v.x, self.y * v.y, self.z * v.z)

    cpdef vec3 num_mul(self, Number v):
        return vec3(self.x * v, self.y * v, self.z * v)

    cpdef vec3 ndarr_mul(self, np.ndarray v):
        return vec3(self.x * v, self.y * v, self.z * v)

    """
    Other functions
    """
    def __abs__(self):
        return self.abs()

    cpdef vec3 abs(self):
        return vec3(abs(self.x), abs(self.y), abs(self.z))

    cpdef vec3 real(self, vec3 v):
        return vec3(np.real(v.x), np.real(v.y), np.real(v.z))

    cpdef vec3 imag(self, vec3 v):
        return vec3(np.imag(v.x), np.imag(v.y), np.imag(v.z))

    cpdef vec3 yzx(self):
        return vec3(self.x, self.z, self.x)
    cpdef vec3 zxy(self):
        return vec3(self.z, self.x, self.y)

    cpdef double complex average(self):
        return (self.x + self.y + self.z) / 3.0
