
######


def stackprep(path_to_flowxnFolder=r"", path_toFlowynFolder=r"", dem=r"", print_list=False, start_date="YYYYMMDD", end_date="YYYYMMDD", output_stackedFolder=r"",
VEL_scale=("month","year") , xres=3.125, yres=3.125, Velocity_shapeFile=False, Resampling=True ):
    
    '''
    This program collects velocity candiate points for time-series analysis.

    Parameters
    ----------
    
    path_to_flowxnFolder: str
        path to folder include east-west velocity files
    
    path_toFlowynFolder: str
        path to folder include north-south velocity files
    
    dem: str
        path to digital elevation model file will be used to geocode the products
    
    print_list: bool
        print list of temporal proceesed dates default is False
    
    start_date: str
        "YYYYMMDD"
    
    end_date: str
        "YYYYMMDD"
    
    output_stackedFolder: str
    
    VEL_scale: str
        "month" or "year") at this stage you can ignore this option; will be removed from future versions
      
    xres: float

    yres: float
     
    Velocity_shapeFile: bool

        set to True if need to generate points for temporal deformation analysis
    
    Resampling : bool

        if True reduce number of measurement points but faster processing

    Returns
    -------
    ESRI Shapefile
        This file include candiate velocity points for timeseries analysis
        
    
    '''
    import geopandas as gpd
    from shapely.geometry import Point
    from functools import partial
    import geopandas as gpd
    from geocube.api.core import make_geocube
    from geocube.rasterize import rasterize_points_griddata
    import earthpy.spatial as es
    import rasterio
    import glob
    import os
    from dateutil import parser
    import numpy as np
    import matplotlib.pyplot as plt
    import rioxarray as rxr
    from rasterio.plot import plotting_extent
    import earthpy.spatial as es

    
    def akdefo_calibration_points(path_to_Raster=r"", output_filePath=r"",outputFile_name="" ):
        if not os.path.exists(output_filePath):
            os.makedirs(output_filePath)
        dataset = rasterio.open(path_to_Raster, "r+")
        dataset.nodata=np.nan
        data_array = dataset.read(1)
        no_data=dataset.nodata
        geometry = [Point(dataset.xy(x,y)[0],dataset.xy(x,y)[1]) for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        vel = [data_array[x,y] for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]  
        # # # #VEL_STDV
        # # dataset = rasterio.open(path_to_STDV_Raster, "r+")
        # # dataset.nodata=np.nan
        # # data_array = dataset.read(1)
        # # no_data=dataset.nodata
        # # geometry = [Point(dataset.xy(x,y)[0],dataset.xy(x,y)[1]) for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # # vel_stdv = [data_array[x,y] for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # # #VEL_Direction
        # dataset = rasterio.open(path_VEL_direction, "r+")
        # dataset.nodata=np.nan
        # data_array = dataset.read(1)
        # no_data=dataset.nodata
        # geometry = [Point(dataset.xy(x,y)[0],dataset.xy(x,y)[1]) for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # vel_dir = [data_array[x,y] for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # ##EW and EW_STDV
        # dataset = rasterio.open(path_EW_Raster, "r+")
        # dataset.nodata=np.nan
        # data_array = dataset.read(1)
        # no_data=dataset.nodata
        # geometry = [Point(dataset.xy(x,y)[0],dataset.xy(x,y)[1]) for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # vel_EW = [data_array[x,y] for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # ######
        # # dataset = rasterio.open(path_EW_Raster_STDV, "r+")
        # # dataset.nodata=np.nan
        # # data_array = dataset.read(1)
        # # no_data=dataset.nodata
        # # geometry = [Point(dataset.xy(x,y)[0],dataset.xy(x,y)[1]) for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # # vel_ew_stdv = [data_array[x,y] for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]

        # ##NS and NS_STDV
        # dataset = rasterio.open(path_NS_Raster, "r+")
        # dataset.nodata=np.nan
        # data_array = dataset.read(1)
        # no_data=dataset.nodata
        # geometry = [Point(dataset.xy(x,y)[0],dataset.xy(x,y)[1]) for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # vel_ns = [data_array[x,y] for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # ######
        # # dataset = rasterio.open(path_NS_Raster_STDV, "r+")
        # # dataset.nodata=np.nan
        # # data_array = dataset.read(1)
        # # no_data=dataset.nodata
        # # geometry = [Point(dataset.xy(x,y)[0],dataset.xy(x,y)[1]) for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]
        # # vel_ns_stdv = [data_array[x,y] for x,y in np.ndindex(data_array.shape) if data_array[x,y] != no_data]



        ######                   
        df = gpd.GeoDataFrame({'geometry':geometry,'VEL':vel})

        # df = gpd.GeoDataFrame({'geometry':geometry, "VEL_Dir":vel_dir, 
        # "VEL_EW":vel_EW, "VEL_EW_STDV":vel_ew_stdv , "VEL_NS":vel_ns, 
        # "VEL_NS_STDV":vel_ns_stdv})

        df.crs = dataset.crs
        df.index.name = 'SiteID'
        df=df.dropna()
        print(df.head(5))
        # def find_outliers(col, thresh=3.5):
        #     from scipy import stats
        #     z=np.abs(stats.zscore(col))
        #     idx_outliers=np.where(z > thresh, True, False)
        #     return pd.Series(idx_outliers,index=col.index)
        # outlier_idx=find_outliers(df["VEL"], thresh=3.5)
        # df=df.loc[outlier_idx==False]
        # df.describe()
        # print(df.head(5))
        Q1 = np.percentile(df["VEL"], 25, interpolation = 'midpoint')
        Q2 = np.percentile(df["VEL"], 50, interpolation = 'midpoint')  
        Q3 = np.percentile(df["VEL"], 75, interpolation = 'midpoint')
        IQR = Q3 - Q1
        q_low= Q1 - 1.5 * IQR
        q_hi=Q3 + 1.5 * IQR
        # q_low = df["VEL"].quantile(0.01)
        # q_hi  = df["VEL"].quantile(0.99)
        df = df[(df["VEL"] < q_hi) & (df["VEL"] > q_low)]

        outlier =[]
        for x in df["VEL"]:
            if ((x> q_hi) or (x<q_low)):
                outlier.append(x)
        print(' outlier in the dataset is', outlier)

        #df.to_file(output_filePath + "/temp.shp", driver='ESRI Shapefile')
        #gdf_NS=gpd.read_file(output_filePath + "/temp.shp")
        # q_low = gdf_NS["VEL"].quantile(0.01)
        # q_hi  = gdf_NS["VEL"].quantile(0.99)

        # 
        df['x'] = df.geometry.apply(lambda p: p.x)
        df['y'] = df.geometry.apply(lambda p: p.y)
       
        # import fiona
        # fiona.supported_drivers['KML'] = 'rw'
        # gdf=df
        # gdf.to_file('velocity.kml', driver='KML')
        df.to_file(output_filePath + "/" + outputFile_name +".shp", driver='ESRI Shapefile')

        # geo_grid_vel = make_geocube(
        #         vector_data=df,
        #         measurements=['VEL_NS'],
        #         resolution=(100, 100),
        #     output_crs="epsg:32610",
        #  rasterize_function=partial(rasterize_points_griddata, method="cubic") )

        # geo_grid_velstd = make_geocube(
        #         vector_data=df,
        #         measurements=['VEL_EW'],
        #         resolution=(100, 100),
        #     output_crs="epsg:32610",
        #  rasterize_function=partial(rasterize_points_griddata, method="cubic") )

        # geo_grid_vel["VEL_NS"].rio.to_raster("stack_data/VEL_NS.tif")
        # geo_grid_velstd["VEL_EW"].rio.to_raster("stack_data/VEL_EW.tif")
        
       


    def is_outlier(points, thresh=1):
        """
        Returns a boolean array with True if points are outliers and False 
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

###############################################################################
###############
    def interpolate_missing_pixels(
                image: np.ndarray,
                mask: np.ndarray,
                method: str = 'nearest',
                fill_value: int = 0):
            """
            :param image: a 2D image
            :param mask: a 2D boolean image, True indicates missing values
            :param method: interpolation method, one of
                'nearest', 'linear', 'cubic'.
            :param fill_value: which value to use for filling up data outside the
                convex hull of known pixel values.
                Default is 0, Has no effect for 'nearest'.
            :return: the image with missing values interpolated
            """
            from scipy import interpolate
            from matplotlib.path import Path

            h, w = image.shape[:2]
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))

            known_x = xx[~mask]
            known_y = yy[~mask]
            known_v = image[~mask]
            missing_x = xx[mask]
            missing_y = yy[mask]


            #######

            # xvalues = np.linspace(int(1), int(1+w), w) 
            # yvalues = np.linspace(int(1), int(1+h), h)

           # missing_x, missing_y = np.meshgrid(xvalues, yvalues)
            



            interp_values = interpolate.griddata(
                (known_x, known_y), known_v, (missing_x, missing_y),
                method=method, fill_value=fill_value
            )

            interp_image = image.copy()
            interp_image[missing_y, missing_x] = interp_values

            #interp_image=interp_values

            return interp_image

    if not os.path.exists(output_stackedFolder):
        os.makedirs(output_stackedFolder)

    glistxn = sorted(glob.glob( path_to_flowxnFolder + "/" +"*.tif"))

#     #convert dates to number of days in the year for start_date
#     YMD= start_date
#     date1 = pd.to_datetime(YMD, format='%Y%m%d')
#    # new_year_day = pd.Timestamp(year=date1.year, month=1, day=1)
#     #StartDate = (date1 - new_year_day).days + 1

#     #convert dates to number of days in the year for start_date
#     YMD= end_date
#     date2 = pd.to_datetime(YMD, format='%Y%m%d')
#     #new_year_day = pd.Timestamp(year=date2.year, month=1, day=1)
#     #EndDate = (date1 - new_year_day).days + 1

#     #No_ofDays= (date2 - date1).days + 1
    # basemap_list=sorted(glob.glob("filtered_rasters/*.tif"))
    # basemap_src=rasterio.open(basemap_list[0])
    # basemap_img=basemap_src.read(1)
    

    def input_dates(start_date="YYYYMMDD", end_date="YYYYMMDD"):
        start_date1=parser.parse(start_date)
        end_date2=parser.parse(end_date)
        date_list_start=[]
        date_list_end=[]
        
        for idx, item in enumerate(glistxn):
            filepath1, img_name = os.path.split(item)
        
            str_date1=img_name[:-22]
            str_date2=img_name[18:-4]
            #input start date
            date_time1 = parser.parse(str_date1)
            date_list_start.append(date_time1)
            #input end date
            date_time2 = parser.parse(str_date2)
            date_list_end.append(date_time2)


        st_date=min(date_list_start, key=lambda d: abs(d - start_date1))
        text_date1=st_date.strftime("%Y%m%d")
        End_date=min(date_list_end, key=lambda d: abs(d - end_date2))
        No_ofDays=(End_date-st_date).days
        
        text_date2=End_date.strftime("%Y%m%d")
        return [text_date1, text_date2, No_ofDays]

     

#20200218_20200220_20200226.tif

    def index(listfilepath):

        for idx, item in enumerate(listfilepath):
            filepath1, img_name = os.path.split(item)
            Select_inputDates=input_dates(start_date, end_date)
            text_date1=Select_inputDates[0]
            text_date2=Select_inputDates[1]
            No_ofDays=Select_inputDates[2]
            
            if img_name[18:-4]==text_date2:
                end_index=idx
                print("end_index: ", end_index, "Date", img_name)
                
            elif img_name[:-22]==text_date1:
                start_index=idx
                print("start_index: ", start_index, "Date", img_name)
            elif print_list==True:
                
                print("Index: ", idx, "Date", img_name)
        return(start_index, end_index, No_ofDays, text_date1, text_date2)

    fn=index(glistxn)
    datefrom=fn[3]
    dateto=fn[4]
    start_index=fn[0]
    end_index=fn[1]
    
    title=("stacked optical velocity from: "+datefrom+" to "+ dateto)
    print( title)
    No_ofDays=fn[2]
    No_ofDays=int(No_ofDays)
    print('No of Days', No_ofDays)

    if VEL_scale=="month":
        VEL_factor=30
    elif VEL_scale=="year":
        VEL_factor=365
    
    
    #Read metadata of first file
    with rasterio.open(glistxn[0]) as src0:
        metaxn = src0.meta

    #Update meta to reflect the number of layers
    
    namesxn = [os.path.basename(x) for x in glistxn[start_index:end_index]]
    print("namesxn", len(namesxn))
    metaxn.update(count=len(namesxn))
    #from rasterio.enums import Resampling
     #Read each layer and write it to stack
   #20220530_20220606_20220608
    with rasterio.open('raster_stackxn.tif', 'w', **metaxn) as dst:
        name_list=[]
        
        for id, layer in enumerate(glistxn[start_index:end_index], start=1):
            filepath1, band_name = os.path.split(layer)
            band_name=band_name[:-4]
            name_list.append(band_name)
            with rasterio.open(layer) as src1:
                
                dst.write_band(id, src1.read(1, masked=True))
                dst.set_band_description(id, band_name)
        #dst.descriptions = tuple(name_list)
     
    #Save Name of files into textfile
    # open file in write mode
    with open(r'Names.txt', 'w') as fp:
        for item in name_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('exporting name list Done')           
    #Read Raster Stack
    with rasterio.open("raster_stackxn.tif") as stack_src:
        stack_dataxn = stack_src.read(masked=True)
       
        stack_metaxn = stack_src.profile
        
        #stack_dataxn = numpy.ma.masked_array(stack_dataxn, mask=(stack_dataxn== 0))

    
    #Check meat data
    stack_metaxn
    
    #print(namesxn[0:110])
    # print ("plot stacked flowxn")
    # ep.plot_bands(stack_dataxn, cmap='gist_rainbow',  scale=False, cbar=True, title=namesxn)  

    print ("Stacking image flowy folder started")
    glistyn = sorted(glob.glob( path_toFlowynFolder + "/" + "*.tif"))
    
    #Read metadata of first file
    with rasterio.open(glistyn[0]) as src0:
        metayn = src0.meta

    #Update meta to reflect the number of layers
    namesyn = [os.path.basename(x) for x in glistyn[start_index:end_index]]
    metayn.update(count = len(namesyn))

     #Read each layer and write it to stack
    with rasterio.open('raster_stackyn.tif', 'w', **metayn) as dst:
        name_list=[]
        for id, layer in enumerate(glistyn[start_index:end_index], start=1):
            filepath1, band_name = os.path.split(layer)
            band_name=band_name[:-4]
            name_list.append(band_name)
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1, masked=True))
                dst.set_band_description(id, band_name)
        #dst.descriptions = tuple(name_list)
            
       

    #Read Raster Stack
    # with rasterio.open("raster_stackyn.tif") as stack_src:
    #     stack_datayn = stack_src.read(masked=True)
        
    #     stack_metayn = stack_src.profile
        
    #     #stack_datayn = numpy.ma.masked_array(stack_datayn, mask=(stack_datayn== 0))

    # #Check meat data
    # stack_metayn

    
    #print(namesxn[0:110])
    # print ("plot stacked flowyn")
    # ep.plot_bands(stack_datayn, cmap='gist_rainbow',  scale=False, cbar=True, title=namesyn) 
# #############################################################################
    # #Read Triplet velocities and write to stack
    # #triplet_vel_list = sorted(glob.glob( path_to_flowxnFolder + "/" +"*.tif"))
    # triplet_vel_list = sorted(glob.glob( "georeferenced_folder/VEL_Triplets/*.tif"))
    # #Read metadata of first file
    # with rasterio.open(triplet_vel_list[0]) as src0:
    #     meta_triplets= src0.meta
    #     meta_1band=src0.meta


    # #working on velocity triplets
    # #######################################################################################################
    # names_triplets = [os.path.basename(x) for x in triplet_vel_list[start_index:end_index]]
    # print("Number of Triplets", len(names_triplets))
    # Reference_date=start_index
    # meta_triplets.update(count=len(namesxn))

    # with rasterio.open('raster_stack_tripletVEL.tif', 'w', **meta_triplets) as dst:
    #     name_list=[]
        
    #     for id, layer in enumerate(triplet_vel_list[start_index:end_index], start=1):
    #         filepath1, band_name = os.path.split(layer)
    #         name_list.append(band_name)
    #         with rasterio.open(layer) as src1:
                
    #             dst.write_band(id, src1.read(1, masked=True))
    #     dst.descriptions = tuple(name_list)
                
    
    # #Read Raster Stack
    # with rasterio.open("raster_stack_tripletVEL.tif") as stack_src:
    #     stack_data_triplets = stack_src.read(masked=True)
       
    #     stack_meta_triplets = stack_src.profile
        
    #     #stack_data_triplets = numpy.ma.masked_array(stack_data_triplets, mask=(stack_data_triplets== 0))
    
    # vel_list=[]
    # for stack_vel in stack_data_triplets:
    #     stack_vel=np.array(stack_vel)
    #     vel_list.append(stack_vel)
    # ep.plot_bands((np.stack(vel_list)), cmap='gist_rainbow', title=names_triplets)

    

    # vel_stacklist=np.stack(vel_list)
    
    # std=np.nanstd(vel_stacklist, axis=0) /No_ofDays * VEL_factor
    
    # std[std > std_mm ]=np.nan
    # mask=np.isnan(std)
    # plt.colorbar(plt.imshow(mask))
    # plt.title(" Velocity STD Map")
    # plt.show

    # velocity=np.nanmean(vel_stacklist, axis=0) /No_ofDays * VEL_factor
    # velocity[mask]=np.nan
 
#     ##################################################################################################
     
    # stackxlist=[]
    # for stackx in stack_dataxn:
        
    #     stackxlist.append(stackx)
    # stackxlistnp=np.stack(stackxlist)
    #Avgx =sum(stackxlist)
    #mean_x =np.average(stackxlist, axis=0) 

    print("No of days: ", No_ofDays)

    #read nstack layers
    from osgeo import gdal
    from osgeo import gdal
    stack_dataxn = gdal.Open('raster_stackxn.tif')
    bands = stack_dataxn.RasterCount

    stackxlist = []  # list to store all the bands
    for band in range(1, bands+1):
        data = stack_dataxn.GetRasterBand(band).ReadAsArray().astype('float')  # (n rows by n cols array)
        stackxlist.append(data)

    stackxlist = np.stack(stackxlist)  # (n bands by n rows by n cols array)
    mean_x = np.nanmean(stackxlist, axis=0)/int(No_ofDays) * VEL_factor  # (n rows by n cols array)

    stack_dataxn=None
    stack_datayn = gdal.Open('raster_stackyn.tif')
    bands = stack_datayn.RasterCount

    stackylistd = []  # list to store all the bands
    for band in range(1, bands+1):
        data = stack_datayn.GetRasterBand(band).ReadAsArray().astype('float')  # (n rows by n cols array)
        stackylistd.append(data)

    stackylistd = np.stack(stackylistd)  # (n bands by n rows by n cols array)
    mean_y = np.nanmean(stackylistd, axis=0)/int(No_ofDays) * VEL_factor  # (n rows by n cols array)

    stack_datayn=None
    # stackylistd=[]
    # for stacky in stack_datayn:
        
    #     stackylistd.append(stacky)

    # stackylistnp=np.stack(stackylistd)
    #Avgy =sum(stackylist)
    # stackylist=np.ma.array(stackylist, mask=(stackylist==0))
    # stackxlist=np.ma.array(stackxlist, mask=(stackxlist==0))

    #mean_x= np.nanmean(stackxlist, axis=0)/int(No_ofDays) * VEL_factor
    #mean_y= np.nanmean(stackylistd, axis=0)/int(No_ofDays) * VEL_factor
    # from scipy import ndimage
    # mean_x=ndimage.median_filter(mean_x, size=20)
    # mean_y=ndimage.median_filter(mean_y, size=20)

    mean_x=mean_x.astype("float32")
    mean_y=mean_y.astype("float32")

    bkgrmask=stackylistd[0]
    bkgrmask[bkgrmask==0]=np.nan
    bkgrmask=np.isnan(bkgrmask)

   
    mean_x[bkgrmask]=np.nan
    mean_y[bkgrmask]=np.nan
    std_x=np.nanstd (stackxlist, axis=0)
    std_y=np.nanstd(stackylistd, axis=0)

    # std_x[std_x > std_mm ]=np.nan
    # std_x[std_x < -std_mm ]=np.nan
   
    # std_y[std_y > std_mm ]=np.nan
    # std_y[std_y < -std_mm ]=np.nan

    mean_x[mean_y ==np.nan ]=np.nan
    mean_y[mean_x ==np.nan]=np.nan


    mask_x=np.isnan(std_x)
    mask_y=np.isnan(std_y)
    std_x[mask_x]=np.nan

    std_y[mask_y]=np.nan
   
    mean_x[mask_x ]=np.nan

    mean_y[mask_y ]=np.nan

    ########

    std_x[mask_y]=np.nan

    std_y[mask_x]=np.nan
   
    mean_x[mask_x ]=np.nan
    mean_y[mask_x ]=np.nan
    mean_x[mask_y ]=np.nan
    mean_y[mask_y ]=np.nan

     

    mean_x=interpolate_missing_pixels(mean_x, mask_x , "cubic")
    mean_y=interpolate_missing_pixels(mean_y, mask_y , "cubic")

    std_mag=std_x
    
    
   
     #######Save georeferenced prodcuts
    # Load the dem
    pathhr=dem
    with rasterio.open(pathhr, 'r') as r1:
        #out_image, out_transform = rasterio.mask.mask(r1, shapes, crop=True)
        meta = r1.meta
        print (meta)
        profile = r1.profile
        profile.update(nodata=np.nan, dtype="float32")
    # meta.update({"driver": "GTiff",
    #              "height": out_image.shape[1],
    #              "width": out_image.shape[2],
    #              "transform": out_transform})

    sv_mag= output_stackedFolder + "/" + datefrom+ "_to_"+ dateto + '_VEL.tif'
    sv_dir= output_stackedFolder + "/" + datefrom+ "_to_"+ dateto + '_dir'+ '.tif'
    sv_E_W= output_stackedFolder + "/" + datefrom+ "_to_"+ dateto + '_E-W'+ '.tif'
    sv_N_S= output_stackedFolder + "/" + datefrom+ "_to_"+ dateto + '_N-S'+ '.tif'
    NS_STDV= output_stackedFolder + "/" + datefrom+ "_to_"+ dateto + '_NS_STDV'+ '.tif'
    EW_STDV= output_stackedFolder + "/" + datefrom+ "_to_"+ dateto + '_EW_STDV'+ '.tif'
    VEL_STDV= output_stackedFolder + "/" + datefrom+ "_to_"+ dateto + '_VEL_STDV'+ '.tif'


    #Direction
    
    
    
    angle_map1=np.arctan2(mean_y,mean_x)
    angle_map1 = np.degrees(angle_map1)
    angle_map1=(450 - angle_map1 ) % 360
    ###
    anglemask=np.isnan(std_x)
    angle_map1[std_y==np.nan]=np.nan
    angle_map1[anglemask]=np.nan
    mean_x[anglemask]=np.nan
    mean_y[anglemask]=np.nan

    velocity=np.hypot(mean_x, mean_y)

    ##Set background Mask

    bkgrmask=stackylistd[0]
    bkgrmask=np.isnan(bkgrmask)

    mean_x[bkgrmask]=np.nan
    mean_y[bkgrmask]=np.nan
    velocity[bkgrmask]=np.nan

    mean_x[bkgrmask]=np.nan
    mean_y[bkgrmask]=np.nan
    velocity[bkgrmask]=np.nan
    mean_x[mean_x==0]=np.nan
    mean_y[mean_y==0]=np.nan
    velocity[velocity==0]=np.nan
    std_mag[std_mag==0]=np.nan
    std_x[std_x==0]=np.nan
    std_y[std_y==0]=np.nan

    mean_x[anglemask]=np.nan
    mean_y[anglemask]=np.nan
    velocity[anglemask]=np.nan

    # velocity = velocity[~is_outlier(velocity)]
    # mean_x = mean_x[~is_outlier(mean_x)]
    # mean_y = mean_y[~is_outlier(mean_y)]

    ################
    
    with rasterio.open(sv_mag, 'w', **profile) as dst:
        dst.write(velocity, indexes=1)
        
        new_meta=dst.meta
        profile = dst.profile
        profile.update(nodata=np.nan, dtype="float32")

    with rasterio.open(VEL_STDV, 'w', **profile) as dst:
        dst.write(std_mag, indexes=1)
        profile = dst.profile
        profile.update(nodata=np.nan, dtype="float32")
    
      
    #E_W=flow_x.astype("float32")
    #N_S=flow_y.astype("float32")
    with rasterio.open(sv_E_W, 'w', **profile) as dst:
        dst.write(mean_x, indexes=1)
        #profile = dst.profile
        profile.update(nodata=np.nan, dtype="float32")

    with rasterio.open(EW_STDV, 'w', **profile) as dst:
        dst.write(std_x, indexes=1)
        profile = dst.profile
        profile.update(nodata=np.nan, dtype="float32")
    
   
    with rasterio.open(sv_N_S, 'w', **profile) as dst:
        dst.write(mean_y, indexes=1)
        profile = dst.profile
        profile.update(nodata=np.nan, dtype="float32")
    
    with rasterio.open(NS_STDV, 'w', **profile) as dst:
        dst.write(std_y, indexes=1)
        profile = dst.profile
        profile.update(nodata=np.nan, dtype="float32")

        
    with rasterio.open(sv_dir, 'w', **profile) as dst:
        dst.write(angle_map1, indexes=1)
        profile = dst.profile
        profile.update(nodata=np.nan, dtype="float32")

    #Names for Interpolated raster
    import earthpy.spatial as es
    with rasterio.open(dem) as src:
        elevation = src.read(1)
        # Set masked values to np.nan
        elevation[elevation < 0] = np.nan
    # Create and plot the hillshade with earthpy
    hillshade = es.hillshade(elevation, azimuth=270, altitude=45)

    dem = rxr.open_rasterio(dem, masked=True)
    dem_plotting_extent = plotting_extent(dem[0], dem.rio.transform())


    import matplotlib.colors as colors
    import earthpy.plot as ep
    # #Plot figures
    fig, ax=plt.subplots(nrows=2, ncols=1, figsize=(10,10))
    ep.plot_bands( hillshade,cbar=False,extent=dem_plotting_extent,ax=ax[0], scale=True)
    ep.plot_bands( hillshade,cbar=False,extent=dem_plotting_extent,ax=ax[1], scale=True)
    a=ax[0].imshow(mean_y, cmap='Spectral' , norm=colors.CenteredNorm(), alpha=0.75)
    ax[0].set_title('NS')
    b=ax[1].imshow(mean_x, cmap='Spectral', norm=colors.CenteredNorm(), alpha=0.75)
    ax[1].set_title('EW')
    
    fig.colorbar(a, ax=ax[0], extend='both')
    fig.colorbar(b,ax=ax[1], extend='both')
    plt.show()

   
    ####Resample Raster outputs to reduce processing time
    if not os.path.exists(output_stackedFolder + "/resampled"):
        os.makedirs(output_stackedFolder + "/resampled")
    sv_mag_resampled= output_stackedFolder + "/resampled" + "/" + datefrom+ "_to_"+ dateto + '_VEL.tif'
    sv_dir_resampled= output_stackedFolder + "/resampled"+ "/" + datefrom+ "_to_"+ dateto + '_dir'+ '.tif'
    sv_E_W__resampled= output_stackedFolder + "/resampled" + "/" + datefrom+ "_to_"+ dateto + '_E-W'+ '.tif'
    sv_N_S__resampled= output_stackedFolder + "/resampled"+ "/" + datefrom+ "_to_"+ dateto + '_N-S'+ '.tif'
    
    def resample(input_raster="", output_raster="" , xres=3.125 , yres=3.125):
        from osgeo import gdal
        import numpy as np
        import matplotlib.pyplot as plt

        ds = gdal.Open(input_raster)

        # resample
        dsRes = gdal.Warp(output_raster, ds, xRes = xres, yRes = yres, 
                        resampleAlg = "bilinear")

        # visualize
        array = dsRes.GetRasterBand(1).ReadAsArray()

        # plt.figure()
        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()

        # close your datasets!
        dsRes =None
        return array

    if Resampling==True:
        
        _1p=resample(sv_mag, sv_mag_resampled, xres=xres, yres=yres )
        _2p=resample(sv_dir, sv_dir_resampled, xres=xres, yres=yres)
        _3p=resample(sv_E_W, sv_E_W__resampled, xres=xres, yres=yres)
        _4p=resample(sv_N_S, sv_N_S__resampled, xres=xres, yres=yres)

   
    
    if Velocity_shapeFile==True:
        print("Velocity Candiate points Collection process started")
        if Resampling==True: 
            _1=akdefo_calibration_points(path_to_Raster=sv_mag_resampled, output_filePath="stack_data", outputFile_name= datefrom + "_to_"+ dateto + "_velocity")
        else:
            _1=akdefo_calibration_points(path_to_Raster=sv_mag, output_filePath="stack_data", outputFile_name= datefrom + "_to_"+ dateto + "_velocity")







#Below functions under development
# def calibrated_ts(velocity_shapefile=r"" ,velocity_raster_stack=r"" , path_tosave_timeseries=r"" ):
#     dataset=rasterio.open("basemap.tif")
#     multi_values_points = pd.Series(dtype='float64')
#         # Read input shapefile with fiona and iterate over each feature
#     with fiona.open(velocity_shapefile) as shp:
#         for feature in shp:
#             siteID = feature['properties']['SiteID']
#             coords = feature['geometry']['coordinates']
            
#             # Read pixel value at the given coordinates using Rasterio
#             # NB: `sample()` returns an iterable of ndarrays.
#             with rasterio.open(velocity_raster_stack) as stack_src:
#                     value = [v for v in stack_src.sample([coords])]

#                     coords_value=[]
                    
#             # Update the pandas serie accordingly
#             multi_values_points.loc[siteID] = value
            
#     multi_values_points
#     #names=[os.path.basename(x) for x in glist]
#     # empty list to read list from a file
#     names = []

#     # open file and read the content in a list
#     with open(r'dates.txt', 'r') as fp:
#         for line in fp:
#             # remove linebreak from a current name
#             # linebreak is the last character of each line
#             x = line[:-1]

#             # add current item to the list
#             names.append(x[18:-4])

#     # display list
#     print(names)

#     df1 = pd.DataFrame(multi_values_points.values.tolist(), index=multi_values_points.index)
#     df1['SiteID'] = df1.index
#     df1
#     gdf=gpd.read_file(velocity_shapefile)
#     df1 = pd.DataFrame(df1[0].values.tolist(), 
#                     columns=[names])


    

#     # df1['geometry']=gdf["geometry"]

#     print ("df1: ", df1)
#     #df1.to_csv(path_tosave_timeseries + "/" + "velocity_ts.csv")
    

#     # df=pd.read_csv(path_tosave_timeseries + "/" + "velocity_ts.csv")
#     # df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

#     # df.interpolate(method='linear', limit_direction='both', axis=0)
#     # print(df)
#     df1['x']=gdf['geometry'].x
#     df1['y']=gdf['geometry'].y
#     df1['SiteID_old']=gdf['SiteID']
#     #df1.index.name = 'SiteID'
#     df1['code']=df1.index
    
#     #df.to_csv(path_tosave_timeseries + "/" + "velocity_ts_new.csv")
    

#     # print (df1.head(5))
# #     from shapely.geometry import Point

# # combine lat and lon column to a shapely Point() object
#     geometry = gpd.points_from_xy(gdf.x, gdf.y, crs=dataset.crs)
#     df = gpd.GeoDataFrame(df1, geometry=geometry)
#     #df1.crs = gdf.crs
#     #new_gdf=pd.read_csv("stack_data/velocity_ts.csv")
#     # from shapely import wkt

#     # new_gdf['geometry'] = new_gdf['geometry'].apply(wkt.loads)
#     # new_gdf.drop('WKT', axis=1, inplace=True) #Drop WKT column
#     # gdf2=gpd.GeoDataFrame(new_gdf, geometry='geometry')
#     # df1.crs=dataset.crs
#     # df1.to_file(path_tosave_timeseries + "/" +"velocity_ts.shp" )
#     geogrid_list=[]
#     for i in range (0,5):
#         temp_measurements=names[i]
#         geo_grid = make_geocube(
#                         vector_data=df,
#                         measurements=[temp_measurements],
#                         resolution=(3.125, 3.125),
#                     output_crs="epsg:32610",
#                 rasterize_function=partial(rasterize_points_griddata, method="cubic"),
#                                     )
#         geo_grid.temp_measurements.where(geo_grid.temp_measurements!=geo_grid.temp_measurements.rio.nodata).plot()




# #########Function to calculate temporal pixel coherence mask

# from genericpath import exists
# from itertools import count
# import rasterio
# from rasterio.plot import show
# from numpy import isnan
# from numpy import isfinite
# import glob
# from functools import reduce
# import os

# from dateutil import parser
# import numpy as np
# import matplotlib.pyplot as plt

# def coh_pixeles(path_to_triplet_velocityFolder=r"", start_date="YYYYMMDD", 
# end_date="YYYYMMDD", output_coherence_maskfile_path=r"", output_calibratedFolder=r""):
#         img_list=sorted(glob.glob(path_to_triplet_velocityFolder + "/" + "*.tif"))

#         if not os.path.exists(output_coherence_maskfile_path):
#             os.makedirs(output_coherence_maskfile_path)
#         output_calibratedFolder="georeferenced_folder" + "/" + output_calibratedFolder
#         if not os.path.exists(output_calibratedFolder):
#             os.makedirs(output_calibratedFolder)

        

#         def input_dates(start_date="YYYYMMDD", end_date="YYYYMMDD"):
#                 start_date1=parser.parse(start_date)
#                 end_date2=parser.parse(end_date)
#                 date_list_start=[]
#                 date_list_end=[]
                
#                 for idx, item in enumerate(img_list):
#                     filepath1, img_name = os.path.split(item)
                
#                     str_date1=img_name[:-4]
#                     str_date2=img_name[:-4]
#                     #input start date
#                     date_time1 = parser.parse(str_date1)
#                     date_list_start.append(date_time1)
#                     #input end date
#                     date_time2 = parser.parse(str_date2)
#                     date_list_end.append(date_time2)


#                 st_date=min(date_list_start, key=lambda d: abs(d - start_date1))
#                 text_date1=st_date.strftime("%Y%m%d")
#                 End_date=min(date_list_end, key=lambda d: abs(d - end_date2))
#                 No_ofDays=(End_date-st_date).days
                
#                 text_date2=End_date.strftime("%Y%m%d")
#                 return [text_date1, text_date2, No_ofDays]

            

#         #20200218_20200220_20200226.tif
                
#         for idx, item in enumerate(img_list):
#             filepath1, img_name = os.path.split(item)
#             Select_inputDates=input_dates(start_date, end_date)
#             text_date1=Select_inputDates[0]
#             text_date2=Select_inputDates[1]
            
#             if img_name[:-4]==text_date2:
#                 end_index=idx
#                 print("end_index: ", end_index, "Date", img_name)
                
#             elif img_name[:-4]==text_date1:
#                 start_index=idx
#                 print("start_index: ", start_index, "Date", img_name)
#                 print("Index: ", idx, "Date", img_name)
#         data_list=[]
#         for i in range (start_index, end_index):
#             src1= rasterio.open(img_list[start_index])
#             src2= rasterio.open(img_list[i])

#             def correlation_coefficient(patch1, patch2):
#                 product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
#                 stds = patch1.std() * patch2.std()
#                 if stds == 0:
#                     return 0
#                 else:
#                     product /= stds
#                     return product

#             meta=src1.meta
#             data1=src1.read(1)
#             data2=src2.read(1)
        
#             sh_row, sh_col=data1.shape
#             d = 3
#             correlation = np.zeros_like(data1)
#             for i in range(d, sh_row - (d + 1)):
#                 for j in range(d, sh_col - (d + 1)):
#                     correlation[i, j] = correlation_coefficient(data1[i - d: i + d + 1,
#                                                         j - d: j + d + 1],
#                                                     data2[i - d: i + d + 1,
#                                                         j - d: j + d + 1]) 
            
#             correlation[correlation<0.6]=np.nan
#             data_list.append(~isnan(correlation))

           
            

            
#             meta.update(count=1)    
#         coh=reduce(lambda x, y: x*y, data_list)
        
        
            
#         outfnamne=output_coherence_maskfile_path+"/"+"coherenceMask.tif"
            

#         with rasterio.open(outfnamne, "w", **meta) as f:
#             f.write(coh, 1)

            
        
#         fig, ax=plt.subplots(1,1 , figsize=(15,15) )
#         show(coh, title="Coherencemask between: "+ str(start_date)+ "-" +str(end_date), cmap="binary", ax=ax)

#         plt.savefig(output_coherence_maskfile_path+"/"+"coherenceMask.jpg", dpi=300)

#         print (" start calibrating images...")
#         ##calibrate Triplets to coh stack pixels
#         for idx, item in enumerate(img_list):
#             filepath1, img_name = os.path.split(item)

#             src= rasterio.open(img_list[idx])
#             profile = src.profile
#             img_data = src1.read(1)

            
#             img_data[coh<0.7]=0
#             #img_data = img_data.filled(fill_value=0)
#             fig, ax=plt.subplots(1,1)
#             show(img_data, title=idx)
#             plt.savefig(output_coherence_maskfile_path+"/" + str(img_name[:-4]) + "_coh.jpg")


#             calib_name= output_calibratedFolder + "/" + img_name

#             with rasterio.open(calib_name, "w", **meta) as f:
#                 f.write(img_data, 1)

#     #show(coh, title="mask")

    


              

        