
def Time_Series(stacked_raster_EW=r"", stacked_raster_NS=r"", velocity_points=r"", dates_name=r"", output_folder="", outputFilename="",
                rasteriz_mean_products=True, std=1, VEL_Scale='year' , velocity_mode="mean", master_reference=False):
    
    '''
    This program uses candiate velocity points from stackprep function and performs linear interpolation in time-domain to calibrate
    stacked velocity. Additionally produces corrected timeseries velocity(daily) in a shapefile.
    
    Parameters
    ----------
    
    stacked_raster_EW: str
    
    stacked_raster_NS: str
    
    velocity_points: str 
        Velcity Candidate points
    
    dates_name: str
        text file include name of each date in format YYYYMMDD
    
    output_folder: str
    
    outputFilename: str
    
    VEL_Scale: str
        'year' , "month" or empty  to calculate velocity within provided dataset date range
    
    velocity_mode: str
        "mean" or "linear"
        
    master_reference: bool
        True if calculate TS to a single reference date, False if calculate TS to subsequent Reference dates
    
    Returns
    -------
    
    Time-series shape file of velocity and direction EW, NS, and 2D(resultant Velocity and direction)
    
    '''
    import rasterio
    import os
    from os.path import isfile, join
    import numpy as np
    import glob
    import geowombat as gw
    import pandas as pd
    import geopandas as gpd
    import scipy.stats as stats
    from geocube.api.core import make_geocube
    from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial
    from functools import partial
    from datetime import datetime
    from dateutil import parser
    
    def Helper_Time_Series(stacked_raster=r"", velocity_points=r"", dates_name=r"", output_folder="", outputFilename="", std=1 , VEL_Scale=VEL_Scale):
        
        '''
        stacked_raster: Path to raster stack .tif file
        
        velocity_points: Path to velocity points in arcmap shapfile format .shp
        
        dates_name: path to text file contains date names of each time series triplets .txt
        
        output_folder: Path to output Folder
        
        outputFilename: name for out time-series velocity shapefile
        '''
        
        import os
        import numpy as np
        import glob
        import geowombat as gw
        import pandas as pd
        import geopandas as gpd
        from sklearn.linear_model import LinearRegression
        
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    
        
        #Open Raster stack, extract pixel info into shape file

        with gw.open(stacked_raster, stack_dim='time') as src:
            print(src)
            df = src.gw.extract(velocity_points)

        #Import names to label timeseries data    
        names = []
        dnames=[]
        with open(dates_name, 'r') as fp:
            for line in fp:
                # remove linebreak from a current name
                # linebreak is the last character of each line
                x = 'D'+ line[:-1]

                # add current item to the list
                names.append(x)
                dnames.append(x[:-18])

        print (len(dnames))
        print(len(df.columns))

        cci=(len(df.columns)- len(dnames))
        df2=df.iloc[:, cci:]
        cc=np.arange(1,cci)
        #Add Timesereises column names
        
        #find outliers using z-score iter 1
        lim = np.abs((df2[cc] - df2[cc].mean(axis=1)) / df2[cc].std(ddof=0, axis=1)) < std
        
        # # # replace outliers with nan
        df2[cc]= df2[cc].where(lim, np.nan)
        
        
        
        df2[cc] = df2[cc].astype(float).apply(lambda x: x.interpolate(method='linear', limit_direction='both'), axis=1).ffill().bfill()
       
        
        df2=df2.T
        
        #find outliers using z-score iter 2
        lim = np.abs((df2 - df2.mean(axis=0)) / df2.std(ddof=0,axis=0)) < std
        #lim=df2.apply(stats.zscore, axis=1) <1
        # # # replace outliers with nan
        df2= df2.where(lim, np.nan)
        
        df2= df2.astype(float).apply(lambda x: x.interpolate(method='linear', limit_direction='both'), axis=0).ffill().bfill()
        
        for col in df2.columns:
            #print (col)
            #df2[col]=pd.to_numeric(df2[col])
            df2[col]= df2[col].interpolate(method='index', axis=0).ffill().bfill()
        
        df2=df2.T
            
           
        #Add Timesereises column names
        df2.columns = dnames
        
        #Calculate Linear Velocity for each data point
        def linear_VEL(df, dnames):
            from datetime import datetime 
            from scipy import stats
            # def best_fit_slope_and_intercept(xs,ys):
            #     from statistics import mean
            #     xs = np.array(xs, dtype=np.float64)
            #     ys = np.array(ys, dtype=np.float64)
            #     m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
            #         ((mean(xs)*mean(xs)) - mean(xs*xs)))
                
            #     b = mean(ys) - m*mean(xs)
                
            #     return m, b
            dd_list=[x.replace("D", "") for x in dnames]
            dates_list=([datetime.strptime(x, '%Y%m%d') for x in dd_list])
            days_num=[( ((x) - (pd.Timestamp(year=x.year, month=1, day=1))).days + 1) for x in dates_list]
            days_num=list(range(0, len(dnames)))
            dslope=[]
            std_slope=[]
            for index, dr in df.iterrows():
                #if index==0:
                rows=df.loc[index, :].values.flatten().tolist()
                row_values=rows
                # dfr = pd.DataFrame(dr).transpose()
                # dfr = dfr.loc[:, ~dfr.columns.str.contains('^Unnamed')]
            
                #slopeVEL=best_fit_slope_and_intercept(days_num, row_values)
                #print("slope", slopeVEL[0])
                slope, intercept, r_value, p_value, std_err = stats.linregress(days_num, row_values)
                dslope.append(slope)
                std_slope.append(std_err)
            return dslope, std_slope
        
        
        
        
            
        
        ###########################################################################
  
        
        dnames_new=[x.replace("D", "") for x in dnames]
        def input_dates(start_date="YYYYMMDD", end_date="YYYYMMDD"):
            start_date1=parser.parse(start_date)
            end_date2=parser.parse(end_date)
            date_list_start=[]
            date_list_end=[]
            for idx, item in enumerate(dnames_new):
                #filepath1, img_name = os.path.split(item) 
                str_date1=item
                str_date2=dnames_new[len(dnames_new)-1]
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

        velocity_scale=(input_dates(start_date=dnames_new[0], end_date=dnames_new[len(dnames_new)-1]))
        
        #################################
        # for idx, row in df2[dnames].iterrows():
        #     lim = np.abs((row[dnames] - df2[dnames]()) / row[dnames].std(ddof=0)) < 1
        #     row[dnames]= row[dnames].where(lim, np.nan)
        #     row[dnames] = row[dnames].astype(float).apply(lambda x: x.interpolate(method='linear', limit_direction='both'), axis=1).ffill().bfill()
            
        
        print (df2.describe())
        temp_df=pd.DataFrame()
        temp_df[dnames[0]]=df2[dnames[0]]
        #Choosing first date as reference for Time Series
        
        if master_reference==True:
            
            df2 = df2.sub(df2[dnames[0]], axis=0)
        else:
            
            df2=df2.diff(axis = 1, periods = 1)
        # count=0
        # for idx, col in enumerate(df2.columns):
        #     df2[col] = df2[col].sub(df2[dnames[count]], axis=0)
        #     count=count+1
            
       
        df2[dnames[0]]=0
            
        linear_velocity=linear_VEL(df2[dnames], dnames)
        out=df2
        if velocity_mode=="mean":
            out['VEL']=out[dnames].mean(axis=1)
            out['VEL_STD']=out[dnames].std(axis=1)
        elif velocity_mode=="linear":
            out['VEL']=linear_velocity[0]
            out['VEL_STD']=linear_velocity[1]
        if VEL_Scale=="month": 
            out['VEL']=out['VEL']/velocity_scale[2] * 30
            out['VEL_STD']=out['VEL_STD']/velocity_scale[2] *30
        elif VEL_Scale=="year":
            out['VEL']=out['VEL']/velocity_scale[2] * 365
            out['VEL_STD']=out['VEL_STD']/velocity_scale[2] * 365
            
        
            
        out['geometry']=df['geometry']
        out['CODE']=df['SiteID']
        #out[dnames[0]]=temp_df[dnames[0]]
        # out['HEIGHT']=0
        # out['H_STDEV']=0
        #out['V_STDEV']=out[dnames].std(axis=1)
        #out['COHERENCE']=0
        #out['H_STDEF']=0
        out['x']=df['x']
        out['y']=df['y']

        col_titles=['CODE','geometry','x', 'y', 'VEL', 'VEL_STD' ]+dnames
        out = out.reindex(columns=col_titles)
        
        

        geo_out=gpd.GeoDataFrame(out, geometry='geometry', crs=df.crs)

        geo_out.to_file(output_folder +"/" + outputFilename)
        (geo_out)

        return geo_out, dnames, linear_VEL
    
    EW=Helper_Time_Series(stacked_raster='raster_stackxn.tif', velocity_points=velocity_points ,
                             dates_name='Names.txt', output_folder='stack_data/TS', outputFilename="TS_EW_"+ os.path.basename(velocity_points), std=std, VEL_Scale=VEL_Scale)
                             
    NS=Helper_Time_Series(stacked_raster='raster_stackyn.tif', velocity_points=velocity_points, 
                             dates_name='Names.txt', output_folder='stack_data/TS', outputFilename="TS_NS_"+ os.path.basename(velocity_points), std=std, VEL_Scale=VEL_Scale)
    
    
    if outputFilename=="":
            outputFilename= "TS_2D_"+ os.path.basename(velocity_points)
            
            
    gdf_ew=EW[0]
    gdf_ns=NS[0]
    dnames=NS[1]
    df_2D_VEL=pd.DataFrame()
    df_2D_VEL['CODE']=gdf_ew['CODE']
    df_2D_VEL['geometry']=gdf_ew['geometry']
    df_2D_VEL['x']=gdf_ew['x']
    df_2D_VEL['y']=gdf_ew['y']
    
   
   #Calculate resultant velocity magnitude
    for col in gdf_ew[dnames].columns:
       
        df_2D_VEL[col]=np.hypot(gdf_ns[col],gdf_ew[col])
       
    df_2D_VEL['VEL_MEAN']=df_2D_VEL[dnames].mean(axis=1)
    df_2D_VEL['V_STDEV']=df_2D_VEL[dnames].std(axis=1)
    #we call linear velocity function from above then reuse it below to replace VEL_2D Mean an STD below for lines
    # linear_2D_Velocity_function=EW[2]
    # linear_2D_Velocity=linear_2D_Velocity_function(df_2D_VEL[dnames], dnames)
    # df_2D_VEL['VEL']=linear_2D_Velocity[0]
    # df_2D_VEL['V_STDEV']=linear_2D_Velocity[1]
    #############################
    col_titles=['CODE','geometry','x', 'y', 'VEL_MEAN' , 'V_STDEV' ]+ dnames 
    df_2D_VEL = df_2D_VEL.reindex(columns=col_titles)
    gdf_2D_VEL=gpd.GeoDataFrame(df_2D_VEL, geometry='geometry', crs=gdf_ew.crs)
    
    
    
    gdf_2D_VEL.to_file(output_folder +"/" + outputFilename)
    
    
    #Calculate resultant velocity direction
    
    dir_df_2D_VEL=pd.DataFrame()
    dir_df_2D_VEL['CODE']=gdf_ew['CODE']
    dir_df_2D_VEL['geometry']=gdf_ew['geometry']
    dir_df_2D_VEL['x']=gdf_ew['x']
    dir_df_2D_VEL['y']=gdf_ew['y']
    
    newcol_dir_list=[]
    for col in gdf_ew[dnames].columns:
        newcol_dir= col
        newcol_dir_list.append(newcol_dir)
        dir_df_2D_VEL[newcol_dir]=np.arctan2(gdf_ns[col],gdf_ew[col])
        dir_df_2D_VEL[newcol_dir]=np.degrees(dir_df_2D_VEL[newcol_dir])
        dir_df_2D_VEL[newcol_dir]=(450-dir_df_2D_VEL[newcol_dir]) % 360
    dir_df_2D_VEL['VELDir_MEAN']=dir_df_2D_VEL[newcol_dir_list].mean(axis=1)
    col_titles=['CODE','geometry','x', 'y', 'VELDir_MEAN'  ]+ newcol_dir_list
    dir_df_2D_VEL = dir_df_2D_VEL.reindex(columns=col_titles)
    dir_gdf_2D_VEL=gpd.GeoDataFrame(dir_df_2D_VEL, geometry='geometry', crs=gdf_ew.crs)
    
    dir_gdf_2D_VEL.to_file(output_folder +"/" + outputFilename[:-4]+"_dir.shp")
    
    #Calcuate Mean Corrected velocity products MEAN X, Y, 2D and Dir
    corrected_mean_products=pd.DataFrame()
    corrected_mean_products['CODE']=gdf_ew['CODE']
    corrected_mean_products['geometry']=gdf_ew['geometry']
    corrected_mean_products['x']=gdf_ew['x']
    corrected_mean_products['y']=gdf_ew['y']
    corrected_mean_products['VEL_E']=gdf_ew['VEL']
    corrected_mean_products['VEL_N']=gdf_ns['VEL']
    #corrected_mean_products['VEL_2D']=df_2D_VEL['VEL_MEAN']
    corrected_mean_products['VEL_2D']=df_2D_VEL['VEL_MEAN']
    corrected_mean_products['2DV_STDEV']=df_2D_VEL['V_STDEV']
    corrected_mean_products['VEL_2DDir']=dir_df_2D_VEL['VELDir_MEAN']
    corrected_mean_products_geo=gpd.GeoDataFrame(corrected_mean_products, geometry='geometry', crs=gdf_ew.crs)
    
    corrected_mean_products_geo.to_file(output_folder +"/" + outputFilename[:-4]+"_mean.shp")
    
    
    if rasteriz_mean_products==True:
        
        corrected_means_list=['VEL_E', 'VEL_N', 'VEL_2D', 'VEL_2DDir', '2DV_STDEV']
       
        for id, col_means in enumerate(corrected_means_list):
                
            geo_grid_vel = make_geocube(
                        vector_data=corrected_mean_products_geo,
                        measurements=[col_means],
                        resolution=(10, 10),
                    output_crs="epsg:32610",
                rasterize_function=partial(rasterize_points_radial, method="nearest") )
            
            geo_grid_vel[col_means].rio.to_raster("stack_data/"+ str(col_means)+'.tif')
            with rasterio.open("stack_data/"+ str(col_means)+'.tif', 'r+')as src:
                meta=src.meta
                src_data=src.read()
            # src.close() # close the rasterio dataset
            # os.remove("stack_data/"+ str(col_means)+'.tif') # delete the file 
            with rasterio.open("stack_data/"+ str(col_means)+'1.tif', 'w+', **meta) as dst:
                dst.write(src_data)
                               