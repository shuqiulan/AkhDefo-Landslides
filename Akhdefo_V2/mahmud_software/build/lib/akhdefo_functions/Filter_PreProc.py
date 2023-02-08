

def Filter_PreProcess(unfiltered_folderPath=r"", UDM2_maskfolderPath=r"", outpath_dir=r"" , Udm_Mask_Option=False, plot_figure=False):


    """
    This program prepare and uses filters to balanace raster image brightness

    Parameters
    ----------
    unfiltered_folderPath:  str

    UDM2_maskfolderPath:    str

    outpath_dir:    str

    Udm_Mask_Option:    bool
        False if True the program uses planetlabs imagery unusable pixel mask to ignore and mask bad image pixels 

    plot_figure:    bool
        True if you want to display output figure directly inside python
    
    
    Returns
    -------
    geotif rasters
        Filtered geotif rasters
    Figures
        plotted filtered rasters and mask for bad pixels

    """
    
    import glob
    from itertools import count
    import os
    import rasterio
    from rasterio.plot import show
    from skimage import exposure
    import numpy as np
    import cv2
    from rasterio.plot import show_hist
    import matplotlib.pyplot as plt

    import seaborn_image as sea_img
    
    
    mypath = unfiltered_folderPath
    print("strat working on folder", mypath)

    #Setup Folder directories
    
    if not os.path.exists(outpath_dir):
        os.makedirs(outpath_dir)

    figs_dir=r"Figs_analysis/filter_figs"
    fig_masks=r"Figs_analysis/mask_figs"

    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    if not os.path.exists(fig_masks):
        os.makedirs(fig_masks)

    img_list=sorted(glob.glob(unfiltered_folderPath+ "/" +"*.tif"))
    udm_list=sorted(glob.glob(UDM2_maskfolderPath + "/" + "*.tif"))
    

    ###############

    for idx, item in enumerate(img_list):

        print(" start Processing Image Number: ", idx, ":", item)
        #img_item=img_list[idx]
        #msk_item=udm_list[idx]
        filepath1, filename = os.path.split(img_list[idx])
        filename=filename.replace("-", "")
        #rgb=rgb_normalize(img_list[idx])
        img_src=rasterio.open(img_list[idx])
        
        meta=img_src.meta
        # meta.update({'nodata':0})
        # meta.update({'count': 3})
        # meta.update({'dtype': np.uint8})

        img_data = img_src.read([3, 2, 1], masked=True)/10000
        img_data_copy=img_data.copy()

        if UDM2_maskfolderPath != "":
            udm_src=rasterio.open(udm_list[idx])


            clean_mask = udm_src.read(1).astype(bool)
            snow_mask = udm_src.read(2).astype(bool)
            shadow_mask = udm_src.read(3).astype(bool)
            light_haze_mask = udm_src.read(4).astype(bool)
            heavy_haze_mask = udm_src.read(5).astype(bool)
            cloud_mask = udm_src.read(6).astype(bool)
            bad_mask=snow_mask+shadow_mask+cloud_mask+light_haze_mask+heavy_haze_mask
            minimum_mask=shadow_mask+cloud_mask+snow_mask
            bad_mask=~clean_mask + snow_mask
            img_data.mask = bad_mask 
            img_data = img_data.filled(fill_value=0)
        
        def normalize(array):
            array_min, array_max = array.min(), array.max()
            (array - array_min) / (array_max - array_min)
            array=exposure.rescale_intensity(array, out_range=(0, 255)).astype(np.uint8)
            return array
        def pct_clip(array,pct=[2,98]):
            array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
            clip = (array - array_min) / (array_max - array_min)
            clip[clip>1]=1
            clip[clip<0]=0
            return clip

        # Convert to numpy arrays
        r_o = img_src.read(3)
        b_o = img_src.read(2)
        g_o = img_src.read(1)

        r_o=exposure.rescale_intensity(r_o, out_range=(0, 255)).astype(np.uint8)

        b_o=exposure.rescale_intensity(b_o, out_range=(0, 255)).astype(np.uint8)

        g_o=exposure.rescale_intensity(g_o, out_range=(0, 255)).astype(np.uint8)

        rgb_img_data = np.dstack((r_o, b_o, g_o))
        rgb_img_data=np.transpose(rgb_img_data, (2,0,1))

        if UDM2_maskfolderPath!="":
            r = exposure.equalize_hist(r_o, mask=clean_mask+ ~snow_mask)
            b = exposure.equalize_hist(b_o, mask=clean_mask+ ~snow_mask)
            g = exposure.equalize_hist(g_o, mask=clean_mask+ ~snow_mask)
        else:
            r = exposure.equalize_hist(r_o)
            b = exposure.equalize_hist(b_o)
            g = exposure.equalize_hist(g_o)
            
        r = exposure.equalize_adapthist(r, kernel_size=128, clip_limit=0.01, nbins=256)
        b = exposure.equalize_adapthist(b, kernel_size=128, clip_limit=0.01, nbins=256)
        g = exposure.equalize_adapthist(g, kernel_size=128, clip_limit=0.01, nbins=256)

        r=r.astype("float64")
        b=b.astype("float64")
        g=g.astype("float64")

        r[r_o==0]=0
        b[b_o==0]=0
        g[g_o==0]=0

        r=pct_clip(r)
        b=pct_clip(b)
        g=pct_clip(b)

        rgb = np.dstack((r, b, g))
        rgb=np.transpose(rgb, (2,0,1))
        #rgb[rgb==0]=255
        if Udm_Mask_Option==True:
            rgb[img_data==0]=0
            
        else:
            rgb[img_data_copy==0]=0
            


        print (rgb.shape)
        
        rgb=exposure.rescale_intensity(rgb, out_range=(0, 255)).astype(np.uint8)
        
        with rasterio.open( outpath_dir + "/" + str(filename), "w", driver='GTiff', width=img_src.shape[1],
            height=img_src.shape[0], count=3, dtype='uint8', crs=img_src.crs, transform=img_src.transform, nodata=0) as dst:
            dst.write(rgb)
        
        ##Set plot extent
        # import rioxarray as rxr
        # from rasterio.plot import plotting_extent
        # rs = rxr.open_rasterio(udm_list[0], masked=True)
        #rs_plotting_extent = plotting_extent(rs[0], rs.rio.transform())
        fig1, ((ax12, ax22), (ax32, ax42) ) = plt.subplots(2, 2, figsize=(30,20))
        show(rgb_img_data, ax=ax12)
        show_hist(rgb_img_data, ax=ax22, bins=10, lw=0.0, 
        stacked=False, alpha=0.3, histtype='stepfilled', density=True, title="Initial: "+str(filename[:-4]))
        ax22.get_legend().remove()
        show(rgb, ax=ax32 )
        show_hist(rgb, ax=ax42, bins=10, lw=0.0, 
        stacked=False, alpha=0.3, histtype='stepfilled', density=True, title="CLAHE: "+str(filename[:-4]))
        ax42.get_legend().remove()
        bn=img_data.shape
        xs=bn[2]
        ys=bn[1]
        ax12.set_xlim(0, xs)
        ax12.set_ylim(ys,0)
        ax32.set_xlim(0, xs)
        ax32.set_ylim(ys,0)
        
        fig1.savefig(figs_dir+ "/" +  str(filename[:-4])+ ".jpg", dpi=150)
        
        if plot_figure==True:
            plt.show()
        else:
            plt.close(fig1)
        
            

        if UDM2_maskfolderPath!="":
            fig2, (ax12 , ax22) = plt.subplots(2,1, figsize=(15,10))
            show(rgb, ax=ax12)
            show(~clean_mask, cmap="binary", ax=ax22)
            #plt.show()
            bn=img_data.shape
            xs=bn[2]
            ys=bn[1]
            ax12.set_xlim(0, xs)
            ax12.set_ylim(ys,0)
            ax22.set_xlim(0, xs)
            ax22.set_ylim(ys,0)
            fig2.savefig(fig_masks + "/" +  str(filename[:-4])+ ".jpg", dpi=150)
        
            if plot_figure==True:
                plt.show()
            else:
                plt.close(fig2)
                
        if idx == len(img_list):
            break
        print("process is completed")
       
    print("All process is completed")
        
        
        
        
        
        
        
        
        
        
        
        
        # #show(bad_mask, title="bad_mask: "+str(udm_list[idx]), cmap="binary")

        # if Color==True:
        #         RGB=[3,2,1]
        #         img_data = img_src.read([RGB[0], RGB[1], RGB[2]])/1000  # apply RGB ordering and scale down
        #         img_copy=img_data.copy()
        #         meta.update({'driver':'GTiff',
        #          'width':img_src.shape[1],
        #          'height':img_src.shape[0],
        #          'count':3,
        #          'dtype':'float32',
        #          'crs':img_src.crs, 
        #          'transform':img_src.transform,
        #          'nodata':0})
        #         if Udm_Mask_Option==True:
        #             img_data = img_src.read([RGB[0], RGB[1], RGB[2]], masked=True)/1000
        #             img_data.mask = bad_mask
        #             img_data = img_data.filled(fill_value=0)
        #         else:
        #             img_data=img_data

        # elif Color==False:
        #     clean_mask = udm_src.read(1).astype(bool)
        #     gray=2
        #     img_data = img_src.read(gray)  # apply RGB ordering and scale down
        #     img_data=img_data.astype("float")
        #     img_data[clean_mask==False]=np.nan
        #     img_copy=img_data.copy()
        #     meta.update({'driver':'GTiff',
        #          'width':img_src.shape[1],
        #          'height':img_src.shape[0],
        #          'count':1,
        #          'dtype':'float32',
        #          'crs':img_src.crs, 
        #          'transform':img_src.transform,
        #          'nodata':0})


        # print ("img_data shape: ", img_data.shape)

        # img_data=exposure.rescale_intensity(img_data, out_range=(0, 255)).astype(np.uint8)
        
        # #Global Histogram Equalization
        # img_histo_equ = exposure.equalize_hist(img_data, mask=clean_mask)

        
        # img_histo_equ[img_copy==0]=0

        # img_histo_equ=exposure.rescale_intensity(img_histo_equ, out_range=(0, 255)).astype(np.uint8)

        # #Apply CLAHE Filter 1 and 2
        # filteredimage = exposure.equalize_adapthist(img_histo_equ, clip_limit=0.01, nbins=256)

        # filteredimage[img_copy==0]=0

        # if Udm_Mask_Option==True:
        #     filteredimage=filteredimage.astype("float")
        #     filteredimage[clean_mask==False]=np.nan

        # filteredimage=exposure.rescale_intensity(filteredimage, out_range=(0, 255)).astype(np.uint8)

        # filteredimage = exposure.equalize_adapthist(filteredimage, clip_limit=0.01, nbins=256)

        # filteredimage[img_copy==0]=0

        # if Udm_Mask_Option==True:
        #     filteredimage=filteredimage.astype("float")
        #     filteredimage[clean_mask==False]=np.nan

        
        # filteredimage=exposure.rescale_intensity(filteredimage, out_range=(0, 255)).astype(np.uint8)
       
       
        
    
        # filteredimage[img_copy==0]=0
        # if Udm_Mask_Option==True:
        #     filteredimage=filteredimage.astype("float")
        #     filteredimage[clean_mask==False]=np.nan

        # #filteredimage=np.transpose(filteredimage, (1,0))
        # filteredimage=np.transpose(filteredimage, (0,1, 2))
        # print (filteredimage.shape)
        # with rasterio.open(outpath_dir+ "/" + str(filename),  'w+', **meta) as dst:
        #     dst.write(filteredimage)


        