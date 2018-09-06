"""
Makes 'from core import ...' work.
"""
from os.path import dirname as dir
from sys import path

path.append(dir(path[0]))
