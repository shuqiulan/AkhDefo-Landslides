
# Akhdefo: 
Computer Vision for Slope Stability: Land Deformation Monitoring Using Optical Satellite Imagery
# Background of Akh-Defo:
AKh-Defo is combination of two different words 1) Akh in Kurdish language means land, earth or soil (origion of the word is from Kurdish badini dailect) 2) Defo is short of English word deformation.

# Recommended Citation:
Muhammad M, Williams-Jones G, Stead D, Tortini R, Falorni G and Donati D (2022) Applications of ImageBased Computer Vision for Remote Surveillance of Slope Instability.Front. Earth Sci. 10:909078. doi: 10.3389/feart.2022.909078

# [Documentation Website](https://akhdefo.readthedocs.io/en/latest/README.html#installation-of-akhdefo-software)

# [Tutorials](https://akhdefo.readthedocs.io/en/latest/Tutorials.html)

# [Latest Version of AkhDefo](https://pypi.org/project/akhdefo-functions/)

# Updates:
* Akhdefo version one is deprecated please use Akhdefo version 2.
* Akhdefo now can run on the cloud for real-time processing
* Akhdefo now consist of 21 Modules that performs end to end python-based GIS and Image Processing and Custumized Figure generation

# Installation of Akhdefo Software:

1.[download anaconda environment akhdefov2.yml file](akhdefov2.yml) 

2.[download python package  requirement  pip_req.txt file](pip_req.txt) 

3.Create new python Anaconda environment using the below command

```python
conda env create -f akhdefov2.yml

```

4.Install required python packages using below command

```python
pip install -r pip_req.txt
```

5.Now install Akhdefo using the below command

```python
pip install akhdefo-functions
```

# Akhdefo Functions Summary

##  Preprocessing functions to unzip and sort images

```python
from akhdefo_functions import copyImage_Data, copyUDM2_Mask_Data , unzip
```

##   Import Module to mosaic and crop raster to Area of Interest using shapefile

```python
from akhdefo_functions import Mosaic, Crop_to_AOI
```

##  Import Module to filter rasters

```python
from akhdefo_functions import Filter_PreProcess
```

## Import Module to calculate triplet raster velocities

```python
 from akhdefo_functions import DynamicChangeDetection
 
 ```

##  Import Modules for plotting

```python
from akhdefo_functions import akhdefo_viewer
from akhdefo_functions import plot_stackNetwork
from akhdefo_functions import Akhdefo_resample
from akhdefo_functions import Akhdefo_inversion
```

##  Import Module to process stacked triplet velocities and collect velocity candidate points

```python
from akhdefo_functions import stackprep'
```
##  Python module to coregister satallite images

```python
from akhdefo_functions  import  Coregistration
```

##  python module to process and calculate timestamp linear deformation rates

```python
from akhdefo_functions import  Time_Series
```

##  python module to plot timeseries profile for selected points

```python
from  akhdefo_functions import  akhdefo_ts_plot

```
