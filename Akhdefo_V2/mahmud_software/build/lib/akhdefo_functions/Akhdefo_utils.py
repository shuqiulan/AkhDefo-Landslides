
def Akhdefo_resample(input_raster="", output_raster="" , xres=3.125 , yres=3.125, SavFig=False , convert_units=False):
    """
    This program performs raster resampling for  rasters
   
    Parameters
    ----------

    input_raster: str
        path to input raster

    output_raster: str
        path to output raster


    xres: float
        horizontal resolution

    yres: float 
        vertical resolution

    SavFig: bool
        True to save output plot False to ignore exporting plot

    convert_units: bool 
        if True converts raster value units from m to mm
    
    Returns
    -------
    Raster geotif

    """
    from osgeo import gdal
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn_image as sb
    import rasterio
    import os
   
    ds = gdal.Open(input_raster)

    # resample
    dsRes = gdal.Warp(output_raster, ds, xRes = xres, yRes = yres, 
                    resampleAlg = "bilinear")
    
    dsRes =None 
    ds = None
  
    # # visualize
    src=rasterio.open(output_raster)
    meta= src.meta
    meta.update({'nodata': np.nan})
    array=src.read(1, masked=True)
    # #array = dsRes.GetRasterBand(1).ReadAsArray()
    array[array==-32767.0]=np.nan
    array=array*1000
    plt.figure()
    sb.imghist(array, descibe=True, cmap='Spectral' , orientation='v')
    # plt.colorbar()
    if SavFig==True:
        plt.savefig("resampled.jpg", dpi=300)
    if convert_units==True:
        
        src.close() # close the rasterio dataset
        os.remove(output_raster) # delete the file 
        
        rs=rasterio.open(output_raster, "w+", **meta)
        rs.write(array, indexes=1)
    plt.show()


def Akhdefo_inversion(horizontal_InSAR="", Vertical_InSAR="", EW_Akhdefo="", NS_Akhdefo="", demFile="", output_folder=r""):
    """
    This program calculates 3D displacement velocity (East-West,North-South and vertical) using combined optical and InSAR products
   
    Parameters
    ----------

    horizontal_InSAR: str
        path to East Velocity InSAR product in geotif format

    Vertical_InSAR: str
        path to Vertical Velocity InSAR product in geotif format

    EW_Akhdefo: str 
        path to east-west velocity  akhdefo(optical) product in geotif format

    NS_Akhdefo: str
        path to north-south velocity  akhdefo(optical) product in geotif format

    demFile: str
        path to DEM raster in geotif format

    output_folder : str
        path to save raster products 

    
    Returns
    -------
    Three geotif rasters
        3D-Velocity (D3D in mm/year) raster
        Plunge raster in degrees
        Trend raster in degress


    """
    
    import rasterio as rio
    import numpy as np
    import os
    import cmocean
    from akhdefo_functions.AkhdefoPlot import akhdefo_viewer
    
    
    if not os.path.exists(output_folder ):
        os.makedirs(output_folder)
    #Load images with rasterio
    D_EW_InSAR=rio.open(horizontal_InSAR)
    D_vertical_insar=rio.open(Vertical_InSAR)
    D_EW_akhdefo=rio.open(EW_Akhdefo)
    D_NS_akhdefo=rio.open(NS_Akhdefo)
    #read images with rasterio
    DEW_insar=D_EW_InSAR.read(1, masked=True)
    DEW_insar[DEW_insar==-32767.0]=np.nan
    DEW_insar=DEW_insar*1000

    DEW_akhdefo=D_EW_akhdefo.read(1, masked=True)
    D_vertical=D_vertical_insar.read(1, masked=True)
    D_vertical[D_vertical==-32767.0]=np.nan
    D_vertical=D_vertical*1000

    DNS_akhdefo=D_NS_akhdefo.read(1, masked=True)

    print (DEW_akhdefo.shape)
    print(D_vertical.shape)
    print (DEW_akhdefo.shape)
    print(DEW_insar.shape)
    DH=np.hypot(DEW_akhdefo, DNS_akhdefo)
    D3D=np.hypot(DH, D_vertical )

    meta=D_EW_InSAR.meta

    trend_radians=np.arcsin(DNS_akhdefo/DH)
    trend_degrees=np.degrees(trend_radians)
    print ("Trend in degree raw data: ", trend_degrees.min(), trend_degrees.max())
    trend_degrees=(450 - trend_degrees ) % 360

    plung_radians=np.arcsin(D_vertical/D3D)
    plung_degree=np.degrees(plung_radians)
    #plung_degree=(90-plung_degree)% 90

    print ("DH: ", DH.max(), DH.min())
    print("D3D: ", D3D.max(), D3D.min())

    #Save products
    _3D_vel=output_folder + "/" + "D3D.tif"
    plung=output_folder+ "/" + "plung_degree.tif"
    trend=output_folder+ "/" + "trend_degrees.tif"
    # with rio.open("DH.tif", 'w', **meta) as dst:
    #         dst.write(DH, indexes=1)
    with rio.open(_3D_vel, 'w', **meta) as dst:
            dst.write(D3D, indexes=1)
    with rio.open(trend, 'w', **meta) as dst:
            dst.write(trend_degrees, indexes=1)
    with rio.open(plung, 'w', **meta) as dst:
            dst.write(plung_degree, indexes=1)


    
    p1=akhdefo_viewer(Path_to_DEMFile=demFile, rasterfile=_3D_vel , cbar_label="mm/year", title="3D Velocity", pixel_resolution_meter=3.125, outputfolder=output_folder,
    outputfileName="3D_Disp.jpg", unit=1, cmap=cmocean.cm.speed, alpha=0.8, noDATA_Mask=True, Normalize=True)
    p2=akhdefo_viewer(Path_to_DEMFile=demFile, rasterfile=plung , cbar_label="degrees", title="Plunge of Dispalcement Velocity", pixel_resolution_meter=3.125, outputfolder=output_folder,
    outputfileName="plunge.jpg", unit=1, cmap=cmocean.cm.delta, alpha=0.8, noDATA_Mask=True, Normalize=True)
    p3=akhdefo_viewer(Path_to_DEMFile=demFile, rasterfile=trend , cbar_label="degress", title="Trend of Dispalcement Velocity", pixel_resolution_meter=3.125, outputfolder=output_folder,
    outputfileName="trend.jpg", unit=1, cmap=cmocean.cm.phase, alpha=0.8, noDATA_Mask=True, Normalize=True)


def Auto_Variogram(path_to_shapefile=r"", column_attribute="", latlon=False):  
    '''
    This program automatically selects best variogram model which later 
    can be used to interpolate datapoints.
    
    Parameters
    ----------
    path_to_shapefile: str 
    type path to shapefile to include data (point data)
    the shapefile attribute must have x, y or lat, lon columns
    
    column_attribute: str
        Name of shapefile field attribute include data

    Returns
    -------
    str
        name of best variogram model
        also figure for plotted variogram models
    
    '''
    import numpy as np
    from matplotlib import pyplot as plt
    import gstools as gs
    import geopandas as gpd  
    
    geodata=gpd.read_file(path_to_shapefile)
    
    
    ###############################################################################
    # Estimate the variogram of the field with automatic bins and plot the result.
    from Akhdefo_Tools import utm_to_latlon
    
    if latlon==True:
        geographic=utm_to_latlon(easting=geodata.x, northing=geodata.y, zone_number=10, zone_letter="N")
        x=geographic[0]
        y=geographic[1]
        z=geodata[column_attribute]
    else:
        x=geodata.x 
        y=geodata.y
        z=geodata[column_attribute]
    bin_center, gamma = gs.vario_estimate((x, y), z, latlon=latlon)

    ###############################################################################
    # Define a set of models to test.

    models = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Matern": gs.Matern,
        "Stable": gs.Stable,
        "Rational": gs.Rational,
        "Circular": gs.Circular,
        "Spherical": gs.Spherical,
        "SuperSpherical": gs.SuperSpherical,
        "JBessel": gs.JBessel,
    }
    scores = {}

    ###############################################################################
    # Iterate over all models, fit their variogram and calculate the r2 score.
    fig, (ax1,ax2,ax3)=plt.subplots(ncols=1, nrows=3, figsize=(10,10))
    # plot the estimated variogram
    ax1.scatter(bin_center, gamma, color="k", label="data")
    #ax = plt.gca()

    # fit all models to the estimated variogram
    for idx, model in enumerate(models):
        fit_model = models[model](dim=2, len_scale=4, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)
        para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True)
        fit_model.plot(x_max=max(bin_center), ax=ax1)
        scores[model] = r2
        ax1.legend()
        
                
    ###############################################################################
    # Create a ranking based on the score and determine the best models

    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print("RANKING by Pseudo-r2 score", max(scores, key=scores.get))
    
    for i, (model, score) in enumerate(ranking, 1):
        print(f"{i:>6}. {model:>15}: {score:.5}")
       
    
    max_score=max(scores, key=scores.get)
    for idx, model in enumerate(models):
    
        if max_score==list(models)[idx]:
            fit_model = models[model](dim=2, len_scale=4, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)
            fit_model.plot(x_max=max(bin_center), ax=ax2)
            ax2.set_title(max_score)
            ax2.legend()
            
            import pykrige.kriging_tools as kt
            from pykrige.ok import OrdinaryKriging

            # a GSTools based covariance model
            #model_var = gs.models[model](dim=2, len_scale=4, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)
            if latlon==True: 
                gridx = np.arange((x.min()), x.max(), 0.2)
                gridy = np.arange(y.min(), y.max(), 0.2)
            else: 
                gridx = np.arange(geodata.x.min(),geodata.x.max(), 10)
                gridy = np.arange(geodata.y.min(), geodata.y.max(), 10)
            OK1 = OrdinaryKriging(geodata.x, geodata.y, geodata[column_attribute], fit_model)
            z1, ss1 = OK1.execute("grid", gridx, gridy)
            ax3.set_title((model))
            img=ax3.imshow(z1[z1!=0], origin="lower", cmap='Spectral')
            fig.colorbar(img, ax=ax3)
            
    fig.show()
    
    return (max(scores, key=scores.get),bin_center, gamma )