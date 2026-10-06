from .radiance_field import Strivec
from .octree import DfsOctree as Octree
from .gaussian import Gaussian

def __getattr__(name):
    if name == 'MeshExtractResult':
        from .mesh import MeshExtractResult
        return MeshExtractResult
    raise AttributeError(name)
