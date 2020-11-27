#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

from cyvec3 import vec3

v1 = vec3(3.2+2j, -1.5+0.5j, 0)
v2 = vec3(1.5+3j, 2-0.2j, -7.5+1j)
print(v1+v2)
print(v1+2-1j)
print(v1-v2)
print(v1-(2-1j))
print(v1*v2)
print(abs(v1))
print(v1.yzx())

v3 = vec3(-1,2,3)
print(v3.average())
