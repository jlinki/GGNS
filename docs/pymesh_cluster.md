# Installing PyMesh on the BwUniCluster 2.0
The following steps are related to the official [PyMesh installation guide](https://pymesh.readthedocs.io/en/latest/installation.html), but do not use any apt-get commands.

Start by cloning the Repo:
```bash
git clone https://github.com/PyMesh/PyMesh.git
cd PyMesh
git submodule update --init
export PYMESH_PATH=`pwd`
``` 
Install the requirements in your conda environment (should usually only install `nose`)
```bash
pip install -r $PYMESH_PATH/python/requirements.txt
``` 
Before you do any cmake building do the following steps:

Load the modules:
```bash
module load compiler/gnu/11.2
module load devel/cmake/3.18
``` 

Install the following packages using conda:
```bash
conda install -c conda-forge nlohmann_json
conda install -c conda-forge cgal
``` 
If you mixed up the order you might also need:
```bash
conda install -c conda-forge mpfr
``` 

Export the following paths:
``` 
export MPFR_LIB=$CONDA_PREFIX/lib/
export MPFR_INC=$CONDA_PREFIX/include
export CGAL_PATH=$CONDA_PREFIX/lib/cmake/CGAL/
```

Delete the cgal package from `third_party/cgal` and
remove everthing regarding cgal from `setup.py` and `build.py`  (also from the ` elif`).

Build PyMesh by executing:
``` 
pip install .
```

Test PyMesh by exectuting:
```
python -c "import pymesh; pymesh.test()"
```
If the `libstdc++` error occurs:
```
rm $CONDA_PREFIX/lib/libstdc++.so
rm $CONDA_PREFIX/lib/libstdc++.so.6
rm $CONDA_PREFIX/lib/libstdc++.so.6.0.28
cp /opt/bwhpc/common/compiler/gnu/11.2.0/lib64/libstdc++* $CONDA_PREFIX/lib
```

