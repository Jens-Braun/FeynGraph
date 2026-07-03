from .feyngraph import topology as topology

def __getattr__(name):
    return getattr(topology, name)
