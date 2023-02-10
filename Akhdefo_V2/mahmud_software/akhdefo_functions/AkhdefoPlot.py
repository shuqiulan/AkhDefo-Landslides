
def akhdefo_viewer(Path_to_DEMFile="" , rasterfile="", 
cbar_label="Velocity(mm/year)", title="Akhdefo-Viewer", pixel_resolution_meter=3.125 , outputfolder=r"" , alpha=0.8,
  unit=1, cmap='jet', noDATA_Mask=False, Normalize=True, SetDates_Filename=False, Set_fig_MinMax=False):

    '''
    This program used for plotting raster products.

    Parameters
    ----------

    Path_to_DEMFile : str 
        provide path to digital elevation raster file to be used as shaded base Map

    rasterfile : str
        provide path to raster file to be plotted

    title : str
        provide your desired title for the plot

    pixel_resolution_meter : float
        provide pixel resolution of the digital elevation raster to draw proper figure scalebar

    outputfolder : str
        provide path to ouput folder to save the plot

    outputfileName : str 
        provide name for the output plot including the desired extension such as .jpg, .pmg, .pdf, etc..

    alpha : float
        transparency level for the plotted raster relative to hillshaded basemap

    unit : int
        conversion unit default is 1 (no conversion) if your data is in meter such as velocity in meter/year set unit to 2 to convert to mm/year

    noDATA_MAsk : bool 
        set to True if you do not want to plot zero values of your dataset
    
    Returns
    -------
    Figure

    '''

    import seaborn_image as sb
    from matplotlib_scalebar.scalebar import ScaleBar
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    pixel_resolution_meter=pixel_resolution_meter
    outputfolder=outputfolder
    outputfileName=os.path.basename(rasterfile)
    outputfileName=outputfileName[:-4]+".jpg"
    unit=unit

    #Function to read dem and prepare it to create hillshade for basemap

    def raster2array(geotif_file , unit=1):
        from osgeo import gdal
        metadata = {}
        dataset = gdal.Open(geotif_file)
        metadata['array_rows'] = dataset.RasterYSize
        metadata['array_cols'] = dataset.RasterXSize
        metadata['bands'] = dataset.RasterCount
        metadata['driver'] = dataset.GetDriver().LongName
        metadata['projection'] = dataset.GetProjection()
        metadata['geotransform'] = dataset.GetGeoTransform()
        mapinfo = dataset.GetGeoTransform()
        metadata['pixelWidth'] = mapinfo[1]
        metadata['pixelHeight'] = mapinfo[5]
    #     metadata['xMin'] = mapinfo[0]
    #     metadata['yMax'] = mapinfo[3]
    #     metadata['xMax'] = mapinfo[0] + dataset.RasterXSize/mapinfo[1]
    #     metadata['yMin'] = mapinfo[3] + dataset.RasterYSize/mapinfo[5]  
        metadata['ext_dict'] = {}
        metadata['ext_dict']['xMin'] = mapinfo[0]
        metadata['ext_dict']['xMax'] = mapinfo[0] + dataset.RasterXSize/mapinfo[1]
        metadata['ext_dict']['yMin'] = mapinfo[3] + dataset.RasterYSize/mapinfo[5]
        metadata['ext_dict']['yMax'] = mapinfo[3]
        
        metadata['extent'] = (metadata['ext_dict']['xMin'],metadata['ext_dict']['xMax'],
                            metadata['ext_dict']['yMin'],metadata['ext_dict']['yMax'])
        
        if metadata['bands'] == 1:
            raster = dataset.GetRasterBand(1)
            metadata['noDataValue'] = raster.GetNoDataValue()
            metadata['scaleFactor'] = raster.GetScale()
            print(metadata['scaleFactor'])
            # band statistics
            metadata['bandstats'] = {} #make a nested dictionary to store band stats in same 
            stats = raster.GetStatistics(True,True)
            metadata['bandstats']['min'] = round(stats[0],2)
            metadata['bandstats']['max'] = round(stats[1],2)
            metadata['bandstats']['mean'] = round(stats[2],2)
            metadata['bandstats']['stdev'] = round(stats[3],2)
            metadata['bandstats']['max'] = round(stats[1],2)
            metadata['bandstats']['mean'] = round(stats[2],2)
            array = dataset.GetRasterBand(1).ReadAsArray(0,0,metadata['array_cols'],metadata['array_rows']).astype(np.float32)
            array[array==float(metadata['noDataValue'])]=np.nan
            
            
            if unit==1:
                array
            elif unit==2:
                array=array*1000.0
               

            #array = array/metadata['scaleFactor']
            return array
        
        elif metadata['bands'] > 1:
            print('More than one band ... fix function for case of multiple bands')

    def hillshade(array,azimuth,angle_altitude):
        azimuth = 360.0 - azimuth 
        x, y = np.gradient(array)
        slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth*np.pi/180.
        altituderad = angle_altitude*np.pi/180.
        shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)

        return 255*(shaded + 1)/2

    dtm_array = raster2array(Path_to_DEMFile, unit=1)
    
    
    src= raster2array(rasterfile, unit=unit)

    import skimage.transform as st 
    src = st.resize(src, dtm_array.shape, mode='constant')
   
    # if unit==1:
    #     src
    # elif unit==2:
    #     src=src*1000
    src=src.astype(np.int16)
    
    max=src.max()
    min=src.min()

    src=np.asanyarray(src).astype(np.float32)
   

    # if -1 * min > max:
    #     min= min
    #     max= -1* min
    # else:
    #     max=max
    #     min= -1*max

    print(min, max)
    
    hs_array = hillshade(dtm_array,270,45)
    print(hs_array.shape)
    print(src.shape)

    # srcx_int, src_meta = raster2array(xname_int)
    # srcy_int, src_meta = raster2array(yname_int)
    # srcv_int, src_meta = raster2array(vname_int)
    
    # Use hillshade function on a DTM Geotiff
    import cmocean
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.colors as colors

    if not os.path.exists(outputfolder ):
        os.makedirs(outputfolder)
    import matplotlib as mpl

    if min < 0 and max >0 :
        if Set_fig_MinMax==True:
            min_n=-300
            max_n=300
            min=min_n
            max=max_n
            offset = mcolors.TwoSlopeNorm(vmin=min,
                        vcenter=0., vmax=max)  
        else:
            offset = mcolors.TwoSlopeNorm(vmin=min,
                        vcenter=0., vmax=max)
            
    else  : 
        if Set_fig_MinMax==True:
            min_n=0
            max_n=300
            min=min_n
            max=max_n
            offset=mcolors.Normalize(vmin=min, vmax=max)
        else:
            offset=mcolors.Normalize(vmin=min, vmax=max)
   
    diff = mcolors.LinearSegmentedColormap.from_list("", ["blue","white","red"])
    #scale = {"dx" : pixel_resolution_meter, "units" : "m"}
    fig, ax=plt.subplots(figsize=(10,5), nrows=1, ncols=1)

    axins = inset_axes(
    ax,
    width="5%",  # width: 5% of parent_bbox width
    height="50%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(1.05, 0., 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0)
    if noDATA_Mask==True:
        src[src==0]=np.nan
        src[src==-32768]=np.nan
        src[src==32768]=np.nan
    ax=sb.imshow(hs_array, ax=ax,  cmap="gray", cbar=False, dx=pixel_resolution_meter, units='m', color="k")
    if Normalize==True:

        ax1=ax.imshow(src,  alpha=alpha, norm=offset, cmap=cmap )
    else:
        ax1=ax.imshow(src,  alpha=alpha,  cmap=cmap )
    ax.set_title(str(title))
   
    a=fig.colorbar(ax1, cax=axins, extend="both")
    a.set_label(cbar_label, labelpad=2, y=0.5, rotation=90)
    plt.grid('on') 
    figname=outputfileName
    bn=dtm_array.shape
    
    xs=bn[1]
    ys=bn[0]
    ax.set_xlim(0, xs)
    ax.set_ylim(ys,0)
    if SetDates_Filename==True:
        ax.text(0.5, - 0.07, outputfileName[:-4],horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
             bbox=dict(facecolor='red', alpha=0.5)) 
    plt.savefig(outputfolder + "/" + figname, dpi=300)


def plot_stackNetwork(src_folder=r"", output_folder=r"" , cmap='tab20', date_plot_interval=(5, 30), marker_size=15):
    '''
    This Program plots temporal network of triplets to be stacked for calculating 
    Annual Mean Velocity from stacked optical images.

    Parameters
    ----------
    src_folder : str
        path to georeferenced_folder

    output_folder : str
        path to output folder to save output Figure plot

    cmap : str
        colormap for the plot default is "tab20"

    date_plot_interval : list
        minumum and maximum plot x axis interval dates for the plot
    
    marker_size : float
        size of plotted points default is 15

    Returns
    -------
    Figure
    '''
    import rasterio
    import glob
    import os
    import seaborn as sb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib
    from matplotlib.collections import LineCollection
    
    if not os.path.exists(output_folder ):
        os.makedirs(output_folder)

    #20200218-20200220-20200226.tif

    path_folder=sorted(glob.glob( src_folder + "/" + "flow_xn/*.tif"))
    df=pd.DataFrame(columns= ["Triplet_Dates", "Triplet ID", "Time"])
    label_date_list=[]
    label_time_list=[]
    label_id_list=[]
    delta_dd_list=[]
    for idx, item in enumerate(path_folder):
        
        filepath,filename=os.path.split(path_folder[idx])
        label_date=filename[:-4]
        label_time=filename[:-22]
        label_date_list.append(label_date)
        label_time_list.append(label_time)
        label_id_list.append(idx)

        ###
        #20200218-20200220-20200226.tif
        Date1_YYYY=filename[:-26]
        Date1_MM=filename[4:-24]
        Date1_DD=filename[6:-22]

        Date2_YYYY=filename[9:-17]
        Date2_MM=filename[13:-15]
        Date2_DD=filename[15:-13]

        Date3_YYYY=filename[18:-8]
        Date3_MM=filename[22:-6]
        Date3_DD=filename[24:-4]
        #convert dates to number of days in the year for image1
        YMD= Date1_YYYY+Date1_MM+Date1_DD
        date1 = pd.to_datetime(YMD, format='%Y%m%d')
        new_year_day = pd.Timestamp(year=date1.year, month=1, day=1)
        day_of_the_year_date1 = (date1 - new_year_day).days + 1

        #convert dates to number of days in the year for image2
        YMD= Date2_YYYY+Date2_MM+Date2_DD
        date2 = pd.to_datetime(YMD, format='%Y%m%d')
        new_year_day = pd.Timestamp(year=date2.year, month=1, day=1)
        day_of_the_year_date2 = (date2 - new_year_day).days + 1

            #convert dates to number of days in the year for image3
        YMD= Date3_YYYY+Date3_MM+Date3_DD
        date3 = pd.to_datetime(YMD, format='%Y%m%d')
        new_year_day = pd.Timestamp(year=date3.year, month=1, day=1)
        day_of_the_year_date3 = (date3 - new_year_day).days + 1

        Delta_DD= (date3-date1).days
        Delta_DD=int(Delta_DD)
        if Delta_DD < 0:
            Delta_DD=Delta_DD*-1
        else:
            Delta_DD=Delta_DD*1
        delta_dd_list.append(Delta_DD)   
        #print (delta_dd_list)
    df["Triplet_Dates"]=(label_date_list)
    df["Triplet ID"]=(label_id_list)
    df["Time"]=(label_time_list)
    df['Delta_DD']=delta_dd_list
    df['Time'] = pd.to_datetime(df['Time'].astype(str), format='%Y%m%d')
   
    t=np.array(df["Time"])
    lb=np.array(df["Triplet_Dates"])
    lb_id=np.array(df["Triplet ID"])
    delta_dates=np.array(df['Delta_DD'])
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,5))
    fig.autofmt_xdate()
    a=ax.scatter( t, delta_dates, c=mdates.date2num(t), cmap=cmap, s=marker_size , norm=matplotlib.colors.Normalize())
    ax.plot(t, delta_dates , color='k' , alpha=0.5)
    
    cb=fig.colorbar(a, ax=ax, orientation='horizontal', pad=0.02)
    loc_major = mdates.AutoDateLocator(minticks=date_plot_interval[0], maxticks=date_plot_interval[1])
    #loc_minor = mdates.AutoDateLocator(minticks=1, maxticks =5)
    #cb.ax.xaxis.set_minor_locator(loc_minor)
    cb.ax.xaxis.set_major_locator(loc_major)
    #cb.ax.xaxis.set_minor_formatter(mdates.ConciseDateFormatter(loc_minor))
    cb.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc_major))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=date_plot_interval[1])) 
    #plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=date_plot_interval[0])) 
    plt.xlim(df['Time'][df.index[0]],df['Time'][df.index[-1]] )
    #plt.title("Triplet Stack Network")
    #plt.xlabel("Dates")
    plt.ylabel("Number of Days for Each Triplet")
    # handles, labels = a.legend_elements(prop='colors', alpha=0.2)
    # legend = ax.legend(handles, labels, bbox_to_anchor=(1.08, 1), loc="upper center", title="Number of days" )
    
    cb.ax.tick_params(rotation=90)
    ax.xaxis.tick_top()
      
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel('Dates') 
    plt.xticks(rotation=85) 
    ax.grid(True)

    #print(df.head(5))
    
    plt.savefig( output_folder + "/" + "Stack_Network.jpg", dpi=300, bbox_inches = 'tight')



    

def akhdefo_ts_plot(path_to_shapefile=r"", dem_path=r"", point_size=1.0, opacity=0.75, cmap="turbo",
                    Set_fig_MinMax=True, MinMaxRange=[-50,50] , color_field='VEL', user_data_points="", 
                    path_saveData_points="" , save_plot=False, Fig_outputDir='' , VEL_Scale='year'):
    '''
    This program used for analysis time-series velocity profiles

    Parameters
    ----------

    user_data_points : str
        provide path to csv. file contains x and y coordinate for points of interest
        you can generate this file by providing path to path_saveData_points (POI.csv).
        This is useful to save mouse click positions to repeat the plots for different datasets for example if you plot several TS profiles for
        EW velocity product, you can recreate TS for the same exact position by saving POI.csv with path_saveData_points and then use that as input for the another
        plot such as NS velocity product via setting user_datapoints="POI.csv"
    
    path_to_shapefile : str 
        type path to timeseries shapefile in stack_data/TS folder
    
    dem_path : str
        path to dem raster in geotif fromat

    point_size : float
        size of the sactter plot points

    opacity : float 
        transparency of the scater overlay

    cmap : str
        Matplotlib colormap options example "RdYlBu_r, jet, turbo, hsv, etc..."
                    
    Set_fig_MinMax : bool
        True or False

    MinMaxRange : list
        [-50,50]  Normalize plot colormap range if Set_fig_MinMax=True

    color_field : str 
        'VEL' ,"VEL_2D", 'VEL_N', 'VEL_E', 'VELDir_MEA'

   
    path_saveData_points : str
        optional, provide directory path if you want to save profile data.
        the data will be saved under POI.csv file


    save_plot: bool
        True or False

    Fig_outputDir : str
        if save_plot=True then
        you save your profile plots in interactive html file and jpg image 

     VEL_Scale : str
        'year' or 'month' projects the velocity into provided time-scale

    
    Returns
    -------
    Interactive Figures
    '''
     #####################################################################
    # Import needed packages
    import os
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import rioxarray as rxr
    from rasterio.plot import plotting_extent
    import earthpy as et
    import earthpy.spatial as es
    import earthpy.plot as ep
    import rasterio as rio
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import cmocean
    #################################################
    def ts_plot(df, plot_number, save_plot=False , output_dir="", plot_filename="" , VEL_Scale=VEL_Scale):
    
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.express as px_temp
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import geopandas as gpd 
        import pandas as pd  
        import seaborn as sns  
        import plotly.offline as py_offline
        import os   
        import statsmodels.api as sm
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from datetime import datetime
        import math
        from ipywidgets import interact
        from ipywidgets import widgets
        
        df=pd.read_csv("temp.csv")
        

        df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
        df['dd_str']=df['dd'].astype(str)
        df['dd_str'] = df['dd_str'].astype(str)
        df.rename(columns={ df.columns[1]: "val" }, inplace = True)
        df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y%m%d')
        
        df=df.set_index('dd')
        
        ########################
        df=df.dropna()
        # Make index pd.DatetimeIndex
        df.index = pd.DatetimeIndex(df.index)
        # Make new index
        idx = pd.date_range(df.index.min(), df.index.max())
        # Replace original index with idx
        df = df.reindex(index = idx)
        # Insert row count
        df.insert(df.shape[1],
                'row_count',
                df.index.value_counts().sort_index().cumsum())

        df=df.dropna()
        
        #df=df.set_index(df['row_count'], inplace=True)

        df.sort_index(ascending=True, inplace=True)
    
    
    
    
        
        
        #####start building slider
        widgets.SelectionRangeSlider(
        options=df.index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '1000px'})
        
        
        ############
        def ts_helper(df, VEL_Scale=VEL_Scale, plot_number=plot_number):
            # py_offline.init_notebook_mode()
            # #%matplotlib widget
            # df=pd.read_csv("temp.csv")
            # df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
            # df['dd_str']=df['dd'].astype(str)
            # df['dd_str'] = df['dd_str'].astype(str)
            # df.rename(columns={ df.columns[1]: "val" }, inplace = True)
            # df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y%m%d')
            # df=df.set_index('dd')
            # ########################
            # df=df.dropna()
            # # Make index pd.DatetimeIndex
            # df.index = pd.DatetimeIndex(df.index)
            # # Make new index
            # idx = pd.date_range(df.index.min(), df.index.max())
            # # Replace original index with idx
            # df = df.reindex(index = idx)
            # # Insert row count
            # df.insert(df.shape[1],
            #         'row_count',
            #         df.index.value_counts().sort_index().cumsum())

            # df=df.dropna()
            
            #df=df.set_index(df['row_count'], inplace=True)

            #df=df.sort_index(ascending=True, inplace=True)

            def best_fit_slope_and_intercept(xs,ys):
                from statistics import mean
                xs = np.array(xs, dtype=np.float64)
                ys = np.array(ys, dtype=np.float64)
                m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                    ((mean(xs)*mean(xs)) - mean(xs*xs)))
                
                b = mean(ys) - m*mean(xs)
                
                return m, b

            #convert dattime to number of days per year
            
            
            dates_list=([datetime.strptime(x, '%Y%m%d') for x in df.dd_str])
            #days_num=[( ((x) - (pd.Timestamp(year=x.year, month=1, day=1))).days + 1) for x in dates_list]
            dd_days=dates_list[len(dates_list)-1]- dates_list[0]
            print(dates_list[0], dates_list[len(dates_list)-1] , dd_days)
            dd_days=str(dd_days)
            dd_days=dd_days.removesuffix('days, 0:00:00')
            delta=int(dd_days)
            m, b = best_fit_slope_and_intercept(df.row_count, df.val)
            print("m:", math.ceil(m*100)/100, "b:",math.ceil(b*100)/100)
            regression_model = LinearRegression()
            val_dates_res = regression_model.fit(np.array(df.row_count).reshape(-1,1), np.array(df.val))
            y_predicted = regression_model.predict(np.array(df.row_count).reshape(-1,1))
        
            if VEL_Scale=='year':
                rate_change=regression_model.coef_[0]/ delta * 365.0
            elif VEL_Scale=='month':
                rate_change=regression_model.coef_[0]/delta * 30
            else:
                rate_change=regression_model.coef_[0]/delta
                
            # model evaluation
            mse=mean_squared_error(np.array(df.val),y_predicted)
            rmse = np.sqrt(mean_squared_error(np.array(df.val), y_predicted))
            r2 = r2_score(np.array(df.val), y_predicted)
            
            # printing values
            slope= ('Slope(linear deformation rate):' + str(math.ceil((regression_model.coef_[0]/delta)*100)/100) + " mm/day")
            Intercept=('Intercept:'+ str(math.ceil(b*100)/100))
            #print('MSE:',mse)
            rmse=('Root mean squared error: '+ str(math.ceil(rmse*100)/100))
            r2=('R2 score: '+ str(r2))
            std=("STD: "+ str(math.ceil(np.std(y_predicted)*100)/100)) 
            # Create figure
            #fig = go.Figure()
            
            return y_predicted, rate_change, slope, Intercept, rmse, r2, std, plot_number, print(len(df)), dd_days
        
        
        @interact
        def read_values(
            slider = widgets.SelectionRangeSlider(
            options=df.index,
            index=(0, len(df.index) - 1),
            description='Dates',
            orientation='horizontal',
            layout={'width': '500px'},
            continuous_update=True) ):
            
            #df=pd.read_csv("temp.csv")
            df=pd.read_csv("temp.csv")

            df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
            df['dd_str']=df['dd'].astype(str)
            df['dd_str'] = df['dd_str'].astype(str)
            df.rename(columns={ df.columns[1]: "val" }, inplace = True)
            df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y%m%d')
            
            df=df.set_index('dd')
            
            ########################
            df=df.dropna()
            # Make index pd.DatetimeIndex
            df.index = pd.DatetimeIndex(df.index)
            # Make new index
            idx = pd.date_range(df.index.min(), df.index.max())
            # Replace original index with idx
            df = df.reindex(index = idx)
            # Insert row count
            df.insert(df.shape[1],
                    'row_count',
                    df.index.value_counts().sort_index().cumsum())

            df=df.dropna()
            
            #df=df.set_index(df['row_count'], inplace=True)

            df.sort_index(ascending=True, inplace=True)
            
            
            
            df=df.loc[slider[0]: slider[1]]
            
            
             
            
            helper=ts_helper(df, VEL_Scale=VEL_Scale)
            
            y_predicted=helper[0]
            rate_change=helper[1]
            slope=helper[2]
            Intercept=helper[3]
            rmse=helper[4]
            r2=helper[5]
            std=helper[6]
            plot_number=helper[7]
            
            print(rate_change)
            print(slope)
            print(rmse)
            print(std)
            print(Intercept)
            
        
            # fig = go.Figure()
            
            # plot_number="Plot Number:"+str(plot_number)

            # fig.add_trace(go.Scatter(x=(df.index), y=(df.val)))
            # fig = px.scatter(df, x=list(df.index), y=list(df.val),
            #             color="val", hover_name="val"
            #                 , labels=dict(x="Dates", y="mm/"+VEL_Scale , color="mm/"+VEL_Scale))
            
            
            
            # fig.add_trace(go.Scatter(x=(df.index), y=(df.val),mode = 'lines',
            #                         name = 'draw lines', line = dict(shape = 'linear', color = 'rgb(0, 0, 0)', dash = 'dash'), connectgaps = True))
            
            # fig.add_trace(
            #     go.Scatter(x=(df.index), y=(y_predicted), mode = "lines",name="trendline", marker_color = "black", line_color='red'))
            
            
            
            # Add range slider
            # fig.update_layout(
            #     xaxis=dict(
            #         rangeselector=dict(
            #             buttons=list([  
            #                 dict(count=1,
            #                     label="1y",
            #                     step="year",
            #                     stepmode="backward"),
            #                 dict(step="all")
            #             ])
            #         ),
            #         rangeslider=dict(
            #             visible=True
            #         ),
            #         type="date"
            #     ) 
            # )
            # fig.update_xaxes(rangeslider_thickness = 0.05)
            # #fig.update_layout(showlegend=True)

            # #fig.data[0].update(line_color='black')
            # tt= "Defo-Rate:"+str(round(rate_change,2))+":"+ "Defo-Rate-STD:"+str(round(np.std(y_predicted), 2))+ ":" +plot_number
            
            # # make space for explanation / annotation
            # fig.update_layout(margin=dict(l=20, r=20, t=20, b=60),paper_bgcolor="LightSteelBlue")

            
            # fig.update_layout(
                
            # title_text=tt, title_font_family="Sitka Small",
            # title_font_color="red", title_x=0.5 , legend_title="Legend",
            # font=dict(
            #     family="Courier New, monospace",
            #     size=15,
            #     color="RebeccaPurple" ))
            
            # fig.update_layout(legend=dict(
            # yanchor="top",
            # y=-0,
            # xanchor="left",
            # x=1.01))

           

            
            # fig.update_xaxes(showspikes=True, spikemode='toaxis' , spikesnap='cursor', spikedash='dot', spikecolor='blue', scaleanchor='y', title_font_family="Arial", 
            #                 title_font=dict(size=15))
            # fig.update_yaxes(showspikes=True, spikemode='toaxis' , spikesnap='cursor', spikedash='dot', spikecolor='blue', scaleanchor='x', title_font_family="Arial",
            #                 title_font=dict(size=15))

            fig = go.Figure()
            fig.update_xaxes(range=[slider[0], slider[1]])
            trace1 = go.Scatter(x=(df.index), y=(y_predicted), mode='lines', name='Trendline')
            fig.add_trace(trace1)
            trace2 = go.Scatter(x=(df.index), y=(df.val), mode='markers', name='Data-Points')
            fig.add_trace(trace2)
            trace3 = go.Scatter(x=(df.index), y=(df.val), mode='lines', name='Draw-line', visible='legendonly')
            fig.add_trace(trace3)
            
            fig.update_layout(xaxis_title="Date", yaxis_title="millimeter")
            
            unit=helper[9]+ "days"
            if VEL_Scale=="year":
                unit="year"
            elif VEL_Scale=="month":
                unit="month"
            else:
                unit=unit
            
            tt= "Defo-Rate:"+str(round(rate_change,2))+"mm/"+unit+":"+ "Defo-Rate-STD:"+str(round(np.std(y_predicted), 2))+ ":Plot ID-" + str(plot_number)
            
            fig.update_layout(
                
            title_text=tt, title_font_family="Sitka Small",
            title_font_color="red", title_x=0.5 , legend_title="Legend",
            font=dict(
                family="Courier New, monospace",
                size=15,
                color="RebeccaPurple" ))
            
            fig.update_layout(font_family="Sitka Small")
            
            # fig.update_layout(legend=dict(
            # yanchor="top",
            # y=-0,
            # xanchor="left",
            # x=1.01))
            fig.update_xaxes(tickformat='%Y.%m.%d')
            fig.update_layout(xaxis = go.layout.XAxis( tickangle = 45))
            
            
            fig.update_layout(hovermode="x unified")
            
           
            go.FigureWidget(fig.to_dict()).show()
           
            
            if save_plot==True:
            
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                fig.write_html(output_dir + "/" + plot_filename + ".html" )
                fig.write_image(output_dir + "/" + plot_filename + ".jpeg", scale=1, width=1080, height=300 )
            
            
            
     
    
   #######################################################################3
   
    dnames=[]
    with open('Names.txt', 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x =  'D' + line[:-1]

            # add current item to the list
            dnames.append(x[:-18])

    # import datetime
    # import ipywidgets as widgets
    # start_date = widgets.DatePicker(
    #     description='Start Date',
    #     disabled=False
    # )
    # end_date = widgets.DatePicker(
    #     description='End Date',
    #     disabled=False)
    # widgets.HBox([start_date, end_date])
    
    # Import shapfilepath
    shapefile_path = os.path.join(path_to_shapefile)
    basename = os.path.basename(shapefile_path[:-4])
    # Open shapefile data with geopandas
    gdf = gpd.read_file(shapefile_path)
    gdf.crs
    # Define path to dem data
    #dem_path = 'dem.tif'

    with rio.open(dem_path) as src:
        elevation = src.read(1)
        # Set masked values to np.nan
        elevation[elevation < 0] = np.nan
    # Create and plot the hillshade with earthpy
    hillshade = es.hillshade(elevation, azimuth=270, altitude=45)

    dem = rxr.open_rasterio(dem_path, masked=True)
    dem_plotting_extent = plotting_extent(dem[0], dem.rio.transform())

    # Getting the crs of the raster data
    dem_crs = es.crs_check(dem_path)
    

    # Transforming the shapefile to the dem data crs
    gdf = gdf.to_crs(dem_crs)
    
    
    min=gdf[color_field].min()
    max=gdf[color_field].max()
    import matplotlib.colors as mcolors
    from matplotlib_scalebar.scalebar import ScaleBar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    
    if min < 0 and max >0 :
        if Set_fig_MinMax==True:
            min_n=MinMaxRange[0]
            max_n=MinMaxRange[1]
            min=min_n
            max=max_n
            offset = mcolors.TwoSlopeNorm(vmin=min,
                        vcenter=0., vmax=max)  
        else:
            offset = mcolors.TwoSlopeNorm(vmin=min,
                        vcenter=0., vmax=max)
            
    else  : 
        if Set_fig_MinMax==True:
            min_n=0
            max_n=100
            min=MinMaxRange[0]
            max=MinMaxRange[1]
            offset=mcolors.Normalize(vmin=min, vmax=max)
        else:
            offset=mcolors.Normalize(vmin=min, vmax=max)


    if user_data_points!="":
        print("Click Anywhere on the Figure to plot Your Time-Series Profile and plot Point Locations")


    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(111)
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    
    #fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10,5))
    ep.plot_bands( hillshade,cbar=False,title=basename,extent=dem_plotting_extent,ax=ax1, scale=False)
    img_main=ax1.scatter(gdf.x, gdf.y, c=gdf[color_field], alpha=opacity, s=point_size, picker=1, cmap=cmap, norm=offset)
    scalebar = ScaleBar(1, "m", length_fraction=0.25, scale_loc="right",border_pad=1,pad=0.5, box_color='white', box_alpha=0.5, location='lower right')
    ax1.add_artist(scalebar)
    plt.grid(True)
    #ax.scatter(gdf.x, gdf.y, s= 0.5, c=gdf.VEL_MEAN ,picker=1)
    cb=fig.colorbar(img_main, ax=ax1, cax=cax, extend='both', orientation='horizontal')
    cb.set_label('mm/year', labelpad=2, x=0.5, rotation=0)
    
    global count
    count=0
    
    x_list=[]
    y_list=[]
    label_ID_list=[]
    df_filt1_list=[]
    
    
    def onclick(event):
        global count
        count+=1
        
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print('button=%d, Figure Coordinates: x=%d, y=%d, : Geographic Coordinates: xdata=%f, ydata=%f' % 
            (event.button, event.x, event.y, event.xdata, event.ydata))
        #plt.plot(event.xdata, event.ydata, ',')
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
         
        def filter_rowwise(gdf,find_nearest, ix, iy ):  
            s1=find_nearest(np.array(gdf.x), ix)
            s2=find_nearest(np.array(gdf.y), iy)
            cols = ['x', 'y']
            vals = [s1, s2]
            geo_df=gdf[np.logical_and.reduce([gdf[c] == v for c, v in zip(cols, vals)])]
            df_filt1=geo_df[dnames]
            
            df_filt1.columns = df_filt1.columns.str.replace(r"D", "")
            df_filt1_list.append(df_filt1)
            
            df=df_filt1.T
            
            df.to_csv('temp.csv')
            
            return s1, s2
        
        
        # ##################
        # def pass_flag(gdf):
        #     Flag="yes"
        #     for idx, row in gdf.iterrows():
        #         #print("Index:", idx, ": ",  row['x'], row['y']) 
        #         s1, s2=filter_rowwise(gdf,find_nearest, row['x'], row['y'] ) 
        #         if s1!=row['x'] and s2!=row['y']:
        #             print("No Data Available at this Location, Please click on the data points")
        #             Flag="no"
        #     return Flag
        # ########################3
        # flag=pass_flag(gdf)
        # if flag=='no':
        #     print("No Data Available at this Location, Please click on the data points")
          
        if user_data_points!="":
            df_poi=pd.read_csv(user_data_points)
            for idx, row in df_poi.iterrows():
                #print("Index:", idx, ": ",  row['x'], row['y'])
                
                s1, s2=filter_rowwise(gdf,find_nearest, row['x'], row['y'] )
                
                
                
                    
                df=pd.read_csv('temp.csv')
                ps=ts_plot(df, idx+1, save_plot=save_plot, output_dir=Fig_outputDir, plot_filename=basename+"_"+str(idx+1), VEL_Scale=VEL_Scale)
                ax1.scatter(s1, s2,  marker=idx+1,  label=idx+1, s=100)
                ax1.text( s1, s2, idx+1, fontdict=dict(color="black"),
                bbox=dict(facecolor="white",alpha=0.75))
                
        else:
            s1, s2=filter_rowwise(gdf,find_nearest, ix, iy )
            
        
            df=pd.read_csv('temp.csv')
            
            ps=ts_plot(df, count, save_plot=save_plot, output_dir=Fig_outputDir, plot_filename=basename+"_"+str(count), VEL_Scale=VEL_Scale)
            
            print("count: " , count)

    
        ########################33
            
            #os.unlink("temp.csv")
            
            
            x_list.append(s1)
            y_list.append(s2)
            label_ID_list.append(count)
            
            
            ax1.scatter(event.xdata, event.ydata,  marker=count,  label=count, s=100)
            ax1.text( event.xdata, event.ydata, count, fontdict=dict(color="black"),
                bbox=dict(facecolor="white",alpha=0.75))
        
        
        
        if path_saveData_points!="": 
            if not os.path.exists(path_saveData_points):
                os.mkdir(path_saveData_points)
            df_filt1=pd.concat(df_filt1_list)
            df_filt1['x']=x_list
            df_filt1['y']=y_list
            df_filt1['ID']=label_ID_list
            
            
            df_filt1 = df_filt1.loc[:, ~df_filt1.columns.str.contains('Unnamed')]
            cols = list(df_filt1.columns)
            cols = [cols[-1]] + cols[:-1]
            cols = [cols[-1]] + cols[:-1]
            cols = [cols[-1]] + cols[:-1]
            df_filt1 = df_filt1[cols]
            
            df_filt1.to_csv(path_saveData_points + "/" + "POI.csv")
            
        ax1.legend(loc='upper left') 
        if save_plot==True:
    
            if not os.path.exists(Fig_outputDir):
                os.mkdir(Fig_outputDir)
            
            plt.savefig(Fig_outputDir + "/" + basename + ".png" )
                
        #os.unlink("temp.csv")
          
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    
    
   

    
    
    
    plt.show() 
   

def MeanProducts_plot_ts(path_to_shapefile="", dem_path="" , out_folder="Figs_analysis", color_field="", Set_fig_MinMax=False, MinMaxRange=[-100,100],
                   opacity=0.5, cmap="jet" , point_size=1, cbar_label="mm/year" , batch_plot=False, dates_list="" ):
    
    """
    This program used to plot shapefile data

    Parameters
    ----------

    path_to_shapefile : str

    dem_path : str 

    out_folder : str

    color_field : str
        geopandas column name

    Set_fig_MinMax : bool

    MinMaxRange : list
            
    opacity : float

    cmap : str

    point_size : str 

    cbar_label : str
        "mm/year" or "degrees", etc.. based on unit of the data column name in the color_field

    Returns
    -------
    Figure
    """
    import os
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import rioxarray as rxr
    from rasterio.plot import plotting_extent
    import earthpy as et
    import earthpy.spatial as es
    import earthpy.plot as ep
    import rasterio as rio
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Import shapfilepath
    shapefile_path = os.path.join(path_to_shapefile)
    basename = os.path.basename(shapefile_path[:-4])
    # Open shapefile data with geopandas
    gdf = gpd.read_file(shapefile_path)
    gdf.crs
    # Define path to dem data
    #dem_path = 'dem.tif'

    with rio.open(dem_path) as src:
        elevation = src.read(1)
        # Set masked values to np.nan
        elevation[elevation < 0] = np.nan
    # Create and plot the hillshade with earthpy
    hillshade = es.hillshade(elevation, azimuth=275, altitude=30)

    dem = rxr.open_rasterio(dem_path, masked=True)
    dem_plotting_extent = plotting_extent(dem[0], dem.rio.transform())

    # Getting the crs of the raster data
    dem_crs = es.crs_check(dem_path)
    

    # Transforming the shapefile to the dem data crs
    gdf = gdf.to_crs(dem_crs)
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    if batch_plot==False:
        min=gdf[color_field].min()
        max=gdf[color_field].max()
        import matplotlib.colors as mcolors
        from matplotlib_scalebar.scalebar import ScaleBar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        
        if min < 0 and max >0 :
            if Set_fig_MinMax==True:
                min_n=MinMaxRange[0]
                max_n=MinMaxRange[1]
                min=min_n
                max=max_n
                offset = mcolors.TwoSlopeNorm(vmin=min,
                            vcenter=0., vmax=max)  
            else:
                offset = mcolors.TwoSlopeNorm(vmin=min,
                            vcenter=0., vmax=max)
                
        else  : 
            if Set_fig_MinMax==True:
                min_n=0
                max_n=100
                min=MinMaxRange[0]
                max=MinMaxRange[1]
                offset=mcolors.Normalize(vmin=min, vmax=max)
            else:
                offset=mcolors.Normalize(vmin=min, vmax=max)
                
        fig = plt.figure(figsize=(7,7))
        ax1 = fig.add_subplot(111)
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        
        #fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10,5))
        ep.plot_bands( hillshade,cbar=False,title=color_field,extent=dem_plotting_extent,ax=ax1, scale=False)
        img_main=ax1.scatter(gdf.x, gdf.y, c=gdf[color_field], alpha=opacity, s=point_size, picker=1, cmap=cmap, norm=offset)
        scalebar = ScaleBar(1, "m", length_fraction=0.25, scale_loc="right",border_pad=1,pad=0.5, box_color='white', box_alpha=0.5, location='lower right')
        ax1.add_artist(scalebar)
        plt.grid(True)
        #ax.scatter(gdf.x, gdf.y, s= 0.5, c=gdf.VEL_MEAN ,picker=1)
        cb=fig.colorbar(img_main, ax=ax1, cax=cax, extend='both', orientation='horizontal')
        cb.set_label(cbar_label, labelpad=2, x=0.5, rotation=0)
        
        
        plt.savefig(out_folder+"/"+color_field+".png")
        
        plt.show()
    
    if batch_plot==True:
        if dates_list=="":
            print("provide list of dates in txt file(Names.txt file)")
        else:
            dnames=[]
            with open(dates_list, 'r') as fp:
                for line in fp:
                    # remove linebreak from a current name
                    # linebreak is the last character of each line
                    x = "D" + line[:-1]
                    # add current item to the list
                    dnames.append(x[:-18])
            
            for nd in gdf[dnames]:
                min=gdf[nd].min()
                max=gdf[nd].max()
                import matplotlib.colors as mcolors
                from matplotlib_scalebar.scalebar import ScaleBar
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                
                
                if min < 0 and max >0 :
                    if Set_fig_MinMax==True:
                        min_n=MinMaxRange[0]
                        max_n=MinMaxRange[1]
                        min=min_n
                        max=max_n
                        offset = mcolors.TwoSlopeNorm(vmin=min,
                                    vcenter=0., vmax=max)  
                    else:
                        offset = mcolors.TwoSlopeNorm(vmin=min,
                                    vcenter=0., vmax=max)
                        
                else  : 
                    if Set_fig_MinMax==True:
                        min_n=0
                        max_n=100
                        min=MinMaxRange[0]
                        max=MinMaxRange[1]
                        offset=mcolors.Normalize(vmin=min, vmax=max)
                    else:
                        offset=mcolors.Normalize(vmin=min, vmax=max)
                fig = plt.figure(figsize=(7,7))
                ax1 = fig.add_subplot(111)
                
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('bottom', size='5%', pad=0.05)
                
                #fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10,5))
                ep.plot_bands( hillshade,cbar=False,title=nd,extent=dem_plotting_extent,ax=ax1, scale=False)
                img_main=ax1.scatter(gdf.x, gdf.y, c=gdf[nd], alpha=opacity, s=point_size, picker=1, cmap=cmap, norm=offset)
                scalebar = ScaleBar(1, "m", length_fraction=0.25, scale_loc="right",border_pad=1,pad=0.5, box_color='white', box_alpha=0.5, location='lower right')
                ax1.add_artist(scalebar)
                plt.grid(True)
                #ax.scatter(gdf.x, gdf.y, s= 0.5, c=gdf.VEL_MEAN ,picker=1)
                cb=fig.colorbar(img_main, ax=ax1, cax=cax, extend='both', orientation='horizontal')
                cb.set_label(cbar_label, labelpad=2, x=0.5, rotation=0)
                
                
                plt.savefig(out_folder+"/"+nd+".png")
                
                if len(dnames)>10:
                    print("akhdefo is plotting more than 10 figures to avoid crushing python kernel we skip displaying figures. \n Please see figures inside provided out_folder path")
                else:
                    
                    plt.show()
                
            
            