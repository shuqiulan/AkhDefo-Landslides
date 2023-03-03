
def unzip(zipdir, dst_dir ):

    """
    This program unzips all the zip products into one folder

    Parameters
    ----------
   

    zipdir : str
        path to directory contains all the zipfiles

    dst_dir : str
        path to destination folder to copy all unzipped products.

    Returns
    -------
    unzip folder

    """
    import os 
    import shutil
    from zipfile import ZipFile
    import glob
    import os
    from os import listdir
    from os.path import isfile, join
    if not os.path.exists( dst_dir):
        os.makedirs( dst_dir)
    dst_dir=dst_dir
    zipdir=zipdir
    zip_list = [f for f in sorted(os.listdir(zipdir)) if isfile(join(zipdir, f))]

    for n in range (0, len(zip_list)):
        with ZipFile(join(zipdir,zip_list[n]), 'r') as zipObject:
            listOfFileNames = zipObject.namelist()
            for fileName in listOfFileNames:
                if fileName.endswith('.tif'):
                    # Extract a single file from zip
                    zipObject.extract(fileName, dst_dir)
                    print('All the tif files are extracted')


def copyImage_Data(path_to_unzipped_folders=r"", Path_to_raster_tifs=""):
    """
    This program copy all the raster images.

    Parameters
    ----------

    path_to_unzipped_folders : str

    
    Path_to_raster_tifs : str

    Returns
    -------
    rasters

    """
    import os 
    import shutil
    from zipfile import ZipFile
    import glob
    import os
    from os import listdir
    from os.path import isfile, join

    if not os.path.exists( Path_to_raster_tifs):
        os.makedirs( Path_to_raster_tifs)

    S2_path = os.path.join (path_to_unzipped_folders)

    for subdir, dirs, files in os.walk(S2_path):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith("BGRN_SR_clip.tif"):
                print (filepath)
                shutil.copy2(filepath, Path_to_raster_tifs)


def copyUDM2_Mask_Data(path_to_unzipped_folders=r"", Path_to_UDM2raster_tifs=r""):

    """
    This program copy all  raster masks.

    Parameters
    ----------

    path_to_unzipped_folders : str
        file extension must end with udm2_clip.tif

    Path_to_UDM2raster_tifs : str

    Returns
    -------
    rasters

    """
    
    import os 
    import shutil
    from zipfile import ZipFile
    import glob
    import os
    from os import listdir
    from os.path import isfile, join

    if not os.path.exists( Path_to_UDM2raster_tifs):
        os.makedirs( Path_to_UDM2raster_tifs)

    S2_path = os.path.join (path_to_unzipped_folders)

    for subdir, dirs, files in os.walk(S2_path):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith("udm2_clip.tif"):
                print (filepath)
                shutil.copy2(filepath, Path_to_UDM2raster_tifs)

def read_data_prep(zip_dir="", image_dir="image_dir", udm_mask_dir="udm_mask_dir"):
    """
    This program reads planetlabs orthoimagery in zipfolder format
    create different directory for the raster images and corresponding unusable data masks(udm)
    the udm mask include snow, haze, etc... see planetslabs udm mask types for further details

    Parameters
    ----------
   

    zip_dir: str
        path to directory contains all the zipfiles

    image_dir: str
        path to folder contains only raster data images
    
    udm_mask_dir: str
        path to folder contains only udm mask ratsers
    
    Returns
    -------
    unzip folder
    image_dir
    udm_mask_dir
    
    

    """
    
    import os 
    import shutil
    from zipfile import ZipFile
    import glob
    import os
    from os import listdir
    from os.path import isfile, join
    import os 
    import shutil
    from zipfile import ZipFile
    import glob
    import os
    from os import listdir
    from os.path import isfile, join
    from osgeo import gdal
    
    # if not os.path.exists( unzip_dir):
    #     os.makedirs( unzip_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(udm_mask_dir):
        os.makedirs(udm_mask_dir)
    #check if the folder is empty if not delete all files
    import os
    import shutil

    for root, dirs, files in os.walk(image_dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    for root, dirs, files in os.walk(udm_mask_dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    #dst_dir=unzip_dir
    zipdir=zip_dir
    zip_list = [f for f in sorted(os.listdir(zipdir)) if isfile(join(zipdir, f))]

    for n in range (0, len(zip_list)):
        with ZipFile(join(zipdir,zip_list[n]), 'r') as zipObject:
            listOfFileNames = zipObject.namelist()
            for fileName in listOfFileNames:
                if fileName.endswith('BGRN_SR_clip.tif'):
                    # Extract a single file from zip
                    zipObject.extract(fileName, image_dir)
                    print('All the tif files are extracted')
             
                elif fileName.endswith('udm2_clip.tif'):
                    # Extract a single file from zip
                    zipObject.extract(fileName, udm_mask_dir)
                    print('All the tif files are extracted')

            
    S2_path = os.path.join (image_dir)
    #####Rename rasters based on metadata
    def safe_copy(file_path, out_dir, dst = None):
        """Safely copy a file to the specified directory. If a file with the same name already 
        exists, the copied file name is altered to preserve both.

        :param str file_path: Path to the file to copy.
        :param str out_dir: Directory to copy the file into.
        :param str dst: New name for the copied file. If None, use the name of the original
            file.
        """
        name = dst or os.path.basename(file_path)
        if not os.path.exists(os.path.join(out_dir, name)):
            shutil.copy(file_path, os.path.join(out_dir, name))
        else:
            base, extension = os.path.splitext(name)
            i = 1
            while os.path.exists(os.path.join(out_dir, '{}_{}{}'.format(base, i, extension))):
                i += 1
            shutil.copy(file_path, os.path.join(out_dir, '{}_{}{}'.format(base, i, extension)))
    
    
    for subdir, dirs, files in os.walk(S2_path):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith("BGRN_SR_clip.tif"):
                print (filepath)
                filename=gdal.Open(filepath)
                metadata=filename.GetMetadata()
                meta_name=(metadata["TIFFTAG_DATETIME"])
                meta_name1=meta_name.replace(":","-")
                meta_name2=meta_name1.replace(" ","_")
                meta_name2
                # close dataset
                filename = None 
                #print (os.path.join(filepath))
                # Absolute path of a file
                old_name = filepath
                file_basename=os.path.basename(filepath)
                dir_name=os.path.dirname(filepath)
                new_name = dir_name+ "/" + meta_name2 + ".tif"  
                # Renaming the file
                #os.rename(old_name, new_name)
                try:
                    os.rename(old_name, new_name)
                except WindowsError:
                    os.remove(new_name)
                    os.rename(old_name, new_name)
                print(dir_name,"\n"+file_basename)
                #new_name
                #filepath_new = subdir + os.sep + meta_name2
                #shutil.copy2(new_name, image_dir)
                #safe_copy(new_name, image_dir)      
    
    # S3_path = os.path.join (udm_mask_dir)     
    # for subdir, dirs, files in os.walk(S3_path):
    #     for file in files:
    #         filepath = subdir + os.sep + file
    #         if filepath.endswith("udm2_clip.tif"):
    #             print (filepath)
    #             filename=gdal.Open(filepath)
    #             metadata=filename.GetMetadata()
    #             meta_name=(metadata["TIFFTAG_DATETIME"])
    #             meta_name1=meta_name.replace(":","-")
    #             meta_name2=meta_name1.replace(" ","_")
    #             meta_name2
    #             # close dataset
    #             filename = None 
    #             #print (os.path.join(filepath))
    #             # Absolute path of a file
    #             old_name = filepath
    #             file_basename=os.path.basename(filepath)
    #             dir_name=os.path.dirname(filepath)
    #             new_name = dir_name+ "/" + meta_name2 + "_udm2.tif"  
    #             # Renaming the file
    #             #os.rename(old_name, new_name)
    #             try:
    #                 os.rename(old_name, new_name)
    #             except WindowsError:
    #                 os.remove(new_name)
    #                 os.rename(old_name, new_name)
    #             print(dir_name,"\n"+file_basename)
    #             #new_name
    #             #filepath_new = subdir + os.sep + meta_name2
    #             #shutil.copy2(new_name, image_dir)
    #             #safe_copy(new_name, udm_mask_dir)
            
            