from distutils.core import setup, Extension
import numpy

# define the extension module
traj_tools = Extension('traj_tools', sources=['traj_tools.c'], 
        include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[traj_tools])
