"""Bounded exact-value cache for immutable world-map polyline interpolation."""
from functools import lru_cache
import numpy as np
import flow_planner.data.data_process.map_process as m
_original=m.interpolate_points
@lru_cache(maxsize=1024)
def _cached(dtype,shape,payload,num_points,interpolation):
    points=np.frombuffer(payload,dtype=np.dtype(dtype)).reshape(shape)
    return _original(points,num_points,interpolation=interpolation)
def interpolate_points(line,num_point,interpolation=None):
    a=np.ascontiguousarray(line)
    return _cached(a.dtype.str,a.shape,a.tobytes(),num_point,interpolation).copy()
m.interpolate_points=interpolate_points
