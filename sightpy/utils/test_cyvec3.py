#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

from cyvec3 import vec3

v1 = vec3(3,0+4j,0)
v2 = vec3(3,2-4j,1-2j)
print(v1.length())
print(v1.square_length())
print(v1.dot(v2))
print(v2.dot(v1))
