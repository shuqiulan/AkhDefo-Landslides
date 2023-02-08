# Filter and raster enhancement

This program performs image color calibration and it has the ability to mask bad pixels.

**import Module**


```python
%matplotlib inline
from akhdefo_functions import Filter_PreProcess

```

**run the module**

Note: you can apply this module at any stage for instance before merging and mosaic or cropping to AOI. Although its recommended to apply before cropping raster images to your desired area of interest.


```python
_filters=Filter_PreProcess(unfiltered_folderPath="data/image_raster_tifs", UDM2_maskfolderPath= "data/UDM2_raster_tifs", 
                          outpath_dir="data/filtred_rasters", Udm_Mask_Option=False, plot_figure=True)
```

    strat working on folder data/image_raster_tifs
     start Processing Image Number:  0 : data/image_raster_tifs\5851965_1062413_2022-08-12_24a4_BGRN_SR_clip.tif
    (3, 1773, 4761)
    


    
![png](output_4_1.png)
    



    
![png](output_4_2.png)
    


    process is completed
     start Processing Image Number:  1 : data/image_raster_tifs\5864669_1062413_2022-08-17_2474_BGRN_SR_clip.tif
    (3, 1773, 4761)
    


    
![png](output_4_4.png)
    



    
![png](output_4_5.png)
    


    process is completed
     start Processing Image Number:  2 : data/image_raster_tifs\5869058_1062413_2022-08-20_2453_BGRN_SR_clip.tif
    (3, 1773, 4761)
    


    
![png](output_4_7.png)
    



    
![png](output_4_8.png)
    


    process is completed
     start Processing Image Number:  3 : data/image_raster_tifs\5869169_1062413_2022-08-20_248c_BGRN_SR_clip.tif
    (3, 1773, 4761)
    


    
![png](output_4_10.png)
    



    
![png](output_4_11.png)
    


    process is completed
     start Processing Image Number:  4 : data/image_raster_tifs\5874216_1062413_2022-08-22_2465_BGRN_SR_clip.tif
    (3, 1773, 4761)
    


    
![png](output_4_13.png)
    



    
![png](output_4_14.png)
    


    process is completed
     start Processing Image Number:  5 : data/image_raster_tifs\5881666_1062413_2022-08-25_241e_BGRN_SR_clip.tif
    (3, 1773, 4761)
    


    
![png](output_4_16.png)
    



    
![png](output_4_17.png)
    


    process is completed
     start Processing Image Number:  6 : data/image_raster_tifs\5881828_1062413_2022-08-25_2414_BGRN_SR_clip.tif
    (3, 1773, 4761)
    


    
![png](output_4_19.png)
    



    
![png](output_4_20.png)
    


    process is completed
     start Processing Image Number:  7 : data/image_raster_tifs\5892964_1062413_2022-08-30_2460_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_22.png)
    



    
![png](output_4_23.png)
    


    process is completed
     start Processing Image Number:  8 : data/image_raster_tifs\5893138_1062413_2022-08-30_2427_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_25.png)
    



    
![png](output_4_26.png)
    


    process is completed
     start Processing Image Number:  9 : data/image_raster_tifs\5895742_1062413_2022-08-31_2459_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_28.png)
    



    
![png](output_4_29.png)
    


    process is completed
     start Processing Image Number:  10 : data/image_raster_tifs\5898249_1062413_2022-09-01_24a4_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_31.png)
    



    
![png](output_4_32.png)
    


    process is completed
     start Processing Image Number:  11 : data/image_raster_tifs\5900307_1062413_2022-09-02_2464_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_34.png)
    



    
![png](output_4_35.png)
    


    process is completed
     start Processing Image Number:  12 : data/image_raster_tifs\5902754_1062413_2022-09-03_2499_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_37.png)
    



    
![png](output_4_38.png)
    


    process is completed
     start Processing Image Number:  13 : data/image_raster_tifs\5903099_1062413_2022-09-03_2426_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_40.png)
    



    
![png](output_4_41.png)
    


    process is completed
     start Processing Image Number:  14 : data/image_raster_tifs\5912289_1062413_2022-09-07_241f_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_43.png)
    



    
![png](output_4_44.png)
    


    process is completed
     start Processing Image Number:  15 : data/image_raster_tifs\5912994_1062413_2022-09-07_2495_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_46.png)
    



    
![png](output_4_47.png)
    


    process is completed
     start Processing Image Number:  16 : data/image_raster_tifs\5917223_1062413_2022-09-09_2478_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_49.png)
    



    
![png](output_4_50.png)
    


    process is completed
     start Processing Image Number:  17 : data/image_raster_tifs\5917611_1062413_2022-09-09_247e_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_52.png)
    



    
![png](output_4_53.png)
    


    process is completed
     start Processing Image Number:  18 : data/image_raster_tifs\5920303_1062413_2022-09-10_2475_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_55.png)
    



    
![png](output_4_56.png)
    


    process is completed
     start Processing Image Number:  19 : data/image_raster_tifs\5926948_1062413_2022-09-13_2482_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_58.png)
    



    
![png](output_4_59.png)
    


    process is completed
     start Processing Image Number:  20 : data/image_raster_tifs\5940456_1062413_2022-09-19_2474_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_61.png)
    



    
![png](output_4_62.png)
    


    process is completed
     start Processing Image Number:  21 : data/image_raster_tifs\5941375_1062413_2022-09-18_2473_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_64.png)
    



    
![png](output_4_65.png)
    


    process is completed
     start Processing Image Number:  22 : data/image_raster_tifs\5943127_1062413_2022-09-20_247c_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_67.png)
    



    
![png](output_4_68.png)
    


    process is completed
     start Processing Image Number:  23 : data/image_raster_tifs\5943207_1062413_2022-09-20_2435_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_70.png)
    



    
![png](output_4_71.png)
    


    process is completed
     start Processing Image Number:  24 : data/image_raster_tifs\5944094_1062413_2022-08-31_2212_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_73.png)
    



    
![png](output_4_74.png)
    


    process is completed
     start Processing Image Number:  25 : data/image_raster_tifs\5945478_1062413_2022-09-21_2276_BGRN_SR_clip.tif
    (3, 1423, 4083)
    


    
![png](output_4_76.png)
    



    
![png](output_4_77.png)
    


    process is completed
     start Processing Image Number:  26 : data/image_raster_tifs\5955030_1062413_2022-09-25_2451_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_79.png)
    



    
![png](output_4_80.png)
    


    process is completed
     start Processing Image Number:  27 : data/image_raster_tifs\5955195_1062413_2022-09-25_2480_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_82.png)
    



    
![png](output_4_83.png)
    


    process is completed
     start Processing Image Number:  28 : data/image_raster_tifs\5960569_1062413_2022-09-27_2442_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_85.png)
    



    
![png](output_4_86.png)
    


    process is completed
     start Processing Image Number:  29 : data/image_raster_tifs\5964662_1062413_2022-09-29_247d_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_88.png)
    



    
![png](output_4_89.png)
    


    process is completed
     start Processing Image Number:  30 : data/image_raster_tifs\5970747_1062413_2022-10-01_2473_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_91.png)
    



    
![png](output_4_92.png)
    


    process is completed
     start Processing Image Number:  31 : data/image_raster_tifs\5974078_1062413_2022-10-02_247f_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_94.png)
    



    
![png](output_4_95.png)
    


    process is completed
     start Processing Image Number:  32 : data/image_raster_tifs\5977188_1062413_2022-10-03_2251_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_97.png)
    



    
![png](output_4_98.png)
    


    process is completed
     start Processing Image Number:  33 : data/image_raster_tifs\5978388_1062413_2022-10-03_2495_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_100.png)
    



    
![png](output_4_101.png)
    


    process is completed
     start Processing Image Number:  34 : data/image_raster_tifs\5985561_1062413_2022-10-06_2262_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_103.png)
    



    
![png](output_4_104.png)
    


    process is completed
     start Processing Image Number:  35 : data/image_raster_tifs\5990651_1062413_2022-10-08_2233_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_106.png)
    



    
![png](output_4_107.png)
    


    process is completed
     start Processing Image Number:  36 : data/image_raster_tifs\6010545_1062413_2022-10-16_2212_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_109.png)
    



    
![png](output_4_110.png)
    


    process is completed
     start Processing Image Number:  37 : data/image_raster_tifs\6016090_1062413_2022-10-18_2455_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_112.png)
    



    
![png](output_4_113.png)
    


    process is completed
     start Processing Image Number:  38 : data/image_raster_tifs\6016170_1062413_2022-10-18_2470_BGRN_SR_clip.tif
    (3, 923, 846)
    


    
![png](output_4_115.png)
    



    
![png](output_4_116.png)
    


    process is completed
    All process is completed
    


```python

```
