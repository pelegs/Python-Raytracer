from __future__ import print_function
import numpy as np
from numpy import exp as npexp
from numpy import sqrt as npsqrt
cimport numpy as np
from libc.math cimport sqrt, abs, exp
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

    ###################
    # UNARY OPERATORS #
    ###################
   
    def __pos__(self):
        return self

    def __neg__(self):
        return self.neg()

    cpdef vec3 neg(self):
        return vec3(-self.x, -self.y, -self.z)

    ####################
    # BINARY OPERATORS #
    ####################

    """
    Addition
    """
    def __add__(self, v):
        if type(self) in NUMBERS: # HACK
            return v.__add__(self)
        if isinstance(v, vec3):
            return self.add(v)
        elif type(v) in NUMBERS:
            return self.num_add(v)
        elif isinstance(v, np.ndarray):
            return self.ndarr_add(v)

    def __radd__(self, v):
        return self.__add__(v)

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
        if type(self) in NUMBERS: # HACK
            return v.__mul__(self)
        if isinstance(v, vec3):
            return self.mul(v)
        elif type(v) in NUMBERS:
            return self.num_mul(v)
        elif isinstance(v, np.ndarray):
            return self.ndarr_mul(v)

    cpdef vec3 mul(self, vec3 v):
        return vec3(self.x*v.x, self.y*v.y, self.z*v.z)

    cpdef vec3 num_mul(self, Number v):
        return vec3(self.x * v, self.y * v, self.z * v)

    cpdef vec3 ndarr_mul(self, np.ndarray v):
        return vec3(self.x * v, self.y * v, self.z * v)
    
    """
    Division
    """
    def __truediv__(self, v):
        if isinstance(v, vec3):
            return self.div(v)
        elif type(v) in NUMBERS:
            return self.num_div(v)
        elif isinstance(v, np.ndarray):
            return self.ndarr_div(v)
    
    def __rtruediv__(self, v):
        if isinstance(v, vec3):
            return self.div(v)
        elif type(v) in NUMBERS:
            return self.num_div(v)
        elif isinstance(v, np.ndarray):
            return self.ndarr_div(v)

    cpdef vec3 div(self, vec3 v):
        if v.x == 0 or v.y == 0 or v.z == 0:
            return self
        else:
            return vec3(self.x / v.x, self.y / v.y, self.z / v.z)

    cpdef vec3 num_div(self, Number v):
        if v == 0:
            return self
        else:
            return vec3(self.x / v, self.y / v, self.z / v)

    cpdef vec3 ndarr_div(self, np.ndarray v):
        if v[0] == 0 or v[1] == 0 or v[2] == 0:
            return self
        else:
            return vec3(self.x / v, self.y / v, self.z / v)

    ###################
    # OTHER FUNCTIONS #
    ###################
    
    def __abs__(self):
        return self.abs()

    cpdef vec3 abs(self):
        return vec3(abs(self.x), abs(self.y), abs(self.z))

    # ignore the m, it's there because of modulus (i.e. pow(x,y,z))
    def __pow__(self, a, m):
        return self.pow(a)

    cpdef vec3 pow(self, Number a):
        return vec3(self.x**a, self.y**a, self.z**a)

    cpdef vec3 real(self, vec3 v):
        return vec3(np.real(v.x), np.real(v.y), np.real(v.z))

    cpdef vec3 imag(self, vec3 v):
        return vec3(np.imag(v.x), np.imag(v.y), np.imag(v.z))

    cpdef vec3 yzx(self):
        return vec3(self.x, self.z, self.x)
    cpdef vec3 zxy(self):
        return vec3(self.z, self.x, self.y)

    cpdef vec3 exp(vec3 v):
        return vec3(np.exp(v.x), np.exp(v.y), np.exp(v.z))

    cpdef vec3 sqrt(vec3 v):
        return vec3(npsqrt(v.x), npsqrt(v.y), npsqrt(v.z))

    cpdef double complex dot(self, vec3 v):
        return self.x * v.x.conjugate() + self.y * v.y.conjugate() + self.z * v.z.conjugate()

    cpdef vec3 cross(self, vec3 v):
        # NOTE: this is the NON-conjugated cross product
        return vec3(self.y*v.z - self.z*v.y, -self.x*v.z + self.z*v.x,  self.x*v.y - self.y*v.x)

    cpdef double complex length(self):
        return npsqrt(self.dot(self))
    
    cpdef double complex square_length(self):
        return self.dot(self)
    
    cpdef vec3 normalize(self):
        # NOTE: the division operator returns the vector if mag==0
        return self / self.length()

    cpdef double complex average(self):
        return (self.x + self.y + self.z) / 3.0

    def to_array(self):
        return np.array([self.x , self.y , self.z])
    
    def components(self):
        return (self.x, self.y, self.z)
