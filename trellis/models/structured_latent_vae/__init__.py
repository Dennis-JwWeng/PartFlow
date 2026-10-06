from .encoder import SLatEncoder, ElasticSLatEncoder
from .decoder_gs import SLatGaussianDecoder, ElasticSLatGaussianDecoder
from .decoder_rf import SLatRadianceFieldDecoder, ElasticSLatRadianceFieldDecoder

def __getattr__(name):
    if name in ('SLatMeshDecoder', 'ElasticSLatMeshDecoder'):
        from . import decoder_mesh
        return getattr(decoder_mesh, name)
    raise AttributeError(name)
