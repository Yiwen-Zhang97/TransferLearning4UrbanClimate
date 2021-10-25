import ee

def export_to_drive(img: ee.Image, region: ee.FeatureCollection, radius: int, units: str, scale: float, folder: str, fname: str) -> ee.batch.Task:
    collection = get_output_samples(img, region, radius, units, scale)
    task = ee.batch.Export.table.toDrive(
            collection=collection,
            description=fname,
            folder=folder,
            fileNamePrefix=fname,
            fileFormat='TFRecord')
    task.start()
    
    return task

def sample_image(feature: ee.Feature, img: ee.Image,
                 scale: float) -> ee.Feature:
    
    image_samples = img.sample(
        region=feature.geometry(),
        scale=scale,
        dropNulls=False)
    return image_samples.first().copyProperties(feature)

def get_output_samples(img: ee.Image, region: ee.FeatureCollection, radius: int, units: str, scale: float) -> ee.FeatureCollection:
    kern = ee.Kernel.square(radius=radius, units=units)
    img_patches = img.neighborhoodToArray(kern)
    samples = region.map(lambda feature: sample_image(feature, img_patches, scale))
    return samples


def filter_collection(collection: str, start_date: str, end_date: str, geometry: ee.FeatureCollection = None) -> ee.ImageCollection:
    filtered_collection = ee.ImageCollection(collection) \
                    .filterDate(start_date, end_date)
    if geometry:
        filtered_collection = ee.ImageCollection(collection) \
                                .filterDate(start_date, end_date)\
                                .filterBounds(geometry)
    return filtered_collection

def rescale_rename_landsat(img: ee.Image) -> ee.Image:
    
    '''
    Rescale and rename some of the bands in Landsat 7.
    
    All bands in Landsat 7 with their scaling factors and offset values are listed here.
    For details about band information and bimasks, see https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2 . 

    Name               Scale         Offset    Description
    
    SR_B1              2.75e-05      -0.2      Band 1 (blue) surface reflectance
    
    SR_B2              2.75e-05      -0.2      Band 2 (green) surface reflectance
     
    SR_B3              2.75e-05      -0.2      Band 3 (red) surface reflectance
    
    SR_B4              2.75e-05      -0.2      Band 4 (near infrared) surface reflectance
     
    SR_B5              2.75e-05      -0.2      Band 5 (shortwave infrared 1) surface reflectance
    
    SR_B7              2.75e-05      -0.2      Band 7 (shortwave infrared 2) surface reflectance
    
    ST_B6              0.00341802    149       Band 6 surface temperature
    
    SR_ATMOS_OPACITY   0.001                   A general interpretation of atmospheric opacity
    
    SR_CLOUD_QA                                Cloud Quality Assessment
    
    ST_ATRAN           0.0001                  Atmospheric Transmittance
    
    ST_CDIST           0.01                    Pixel distance to cloud
    
    ST_DRAD            0.001                   Downwelled Radiance
    
    ST_EMIS            0.0001                  Emissivity estimated from ASTER GED
    
    ST_EMSD            0.0001                  Emissivity standard deviation
    
    ST_QA              0.01                    Uncertainty of the Surface Temperature band
    
    ST_TRAD            0.001                   Thermal band converted to radiance
    
    ST_URAD            0.001                   Upwelled Radiance
    
    QA_PIXEL
    
    QA_RADSAT
    
    '''
    
    SR = img.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7']).rename(['Blue','Green','Red','NIR','SWIR1','SWIR2'])

    SR = SR.multiply(2.75e-05).add(-0.2)
    ST = img.select(['ST_B6']).rename(['ST']).multiply(0.00341802).add(149)
    SR_ATMOS_OPACITY = img.select(['SR_ATMOS_OPACITY']).multiply(0.001)
    ST_ATRAN = img.select(['ST_ATRAN']).multiply(0.0001)
    ST_CDIST = img.select(['ST_CDIST']).multiply(0.01)
    ST_DRAD = img.select(['ST_DRAD']).multiply(0.001)
    ST_EMIS = img.select(['ST_EMIS']).multiply(0.0001)
    ST_EMSD = img.select(['ST_EMSD']).multiply(0.0001)
    ST_QA = img.select(['ST_QA']).multiply(0.01)
    ST_TRAD = img.select(['ST_TRAD']).multiply(0.001)
    ST_URAD = img.select(['ST_URAD']).multiply(0.001)
    
    No_scale=img.select(['SR_CLOUD_QA','QA_PIXEL','QA_RADSAT'])
    
    

    landsat_scaled = ee.Image.cat([SR, ST, SR_ATMOS_OPACITY, ST_ATRAN, ST_CDIST, ST_DRAD, ST_EMIS, ST_EMSD, ST_QA, ST_TRAD, ST_URAD, No_scale])
    landsat_scaled = landsat_scaled.copyProperties(img).set('system:time_start', img.get('system:time_start'))

    return landsat_scaled

def update_qamask(img: ee.Image) -> ee.Image:
    '''
    Return cloud and RGB saturation masked of an image, indicated from its QA_PIXEL and QA_RADSAT band.
    '''
    qa = img.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(8).eq(0) 
    cloud_mask = cloud.updateMask(cloud).rename(['maks_cloud'])
    
    rs = img.select('QA_RADSAT')
    blue_sat = (rs.bitwiseAnd(1).eq(0)).And(img.select('Blue').lt(65455))
    blue_sat_mask = blue_sat.updateMask(blue_sat).rename(['blue_sat'])    
    green_sat = (rs.bitwiseAnd(2).eq(0)).And(img.select('Green').lt(65455))
    green_sat_mask = green_sat.updateMask(green_sat).rename(['green_sat']) 
    red_sat = (rs.bitwiseAnd(4).eq(0)).And(img.select('Red').lt(65455))
    red_sat_mask = red_sat.updateMask(red_sat).rename(['red_sat'])
    
    return img.updateMask(cloud_mask).updateMask(blue_sat_mask).updateMask(green_sat_mask).updateMask(red_sat_mask)

# Apply the USGS L7 Phase-2 Gap filling protocol, using a single kernel size. 
def GapFill(src, fill, kernelSize, upscale, MIN_SCALE = 1/3, MAX_SCALE = 3, MIN_NEIGHBORS = 144):
    kernel = ee.Kernel.square(kernelSize * 30, "meters", False)

  # Find the pixels common to both scenes.
    common = src.mask().And(fill.mask())
    fc = fill.updateMask(common)
    sc = src.updateMask(common)

  # Find the primary scaling factors with a regression.
  # Interleave the bands for the regression.  This assumes the bands have the same names.

  #regress contains only common pixels of two scenes
    regress = fc.addBands(sc)

    regress = regress.select(regress.bandNames().sort())

    ratio = 20

#reproject the regree image to contain ratio*ratio regress pixels in one output pixel
#reproject also ensures that the two images are geo-registered
#linearFit reducer expects x input followed by y input, so it maps fill pixel values to source pixel values
    if upscale :
        fit = regress \
          .reduceResolution(ee.Reducer.median(), False, 500) \
          .reproject(regress.select(0).projection().scale(ratio, ratio)) \
          .reduceNeighborhood(ee.Reducer.linearFit().forEach(src.bandNames()), kernel, 'kernel' , False) \
          .unmask() \
          .reproject(regress.select(0).projection().scale(ratio, ratio))
    else :
        fit = regress \
          .reduceNeighborhood(ee.Reducer.linearFit().forEach(src.bandNames()), kernel,'kernel', False)

    offset = fit.select(".*_offset")
    scale = fit.select(".*_scale")

    # Find the secondary scaling factors using just means and stddev
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(),'', True)

    if upscale :
#get mean and std of both source and fill image seperately
        src_stats = src \
          .reduceResolution(ee.Reducer.median(), False, 500) \
          .reproject(regress.select(0).projection().scale(ratio, ratio)) \
          .reduceNeighborhood(reducer, kernel, 'kernel',False) \
          .reproject(regress.select(0).projection().scale(ratio, ratio))

        fill_stats = fill \
          .reduceResolution(ee.Reducer.median(), False, 500) \
          .reproject(regress.select(0).projection().scale(ratio, ratio)) \
          .reduceNeighborhood(reducer, kernel, 'kernel',False) \
          .reproject(regress.select(0).projection().scale(ratio, ratio))
    else :
        src_stats = src \
          .reduceNeighborhood(reducer, kernel,'kernel', False)

        fill_stats = fill \
          .reduceNeighborhood(reducer, kernel,'kernel', False)

    scale2 = src_stats.select(".*stdDev").divide(fill_stats.select(".*stdDev"))
    offset2 = src_stats.select(".*mean").subtract(fill_stats.select(".*mean").multiply(scale2))

#check the gain to make sure that outliers do not have a strong effect on the result
    invalid = scale.lt(MIN_SCALE).Or(scale.gt(MAX_SCALE))
    scale = scale.where(invalid, scale2)
    offset = offset.where(invalid, offset2)

    # When all else fails, just use the difference of means as an offset.
    invalid2 = scale.lt(MIN_SCALE).Or(scale.gt(MAX_SCALE))
    scale = scale.where(invalid2, 1)
    offset = offset.where(invalid2, src_stats.select(".*mean").subtract(fill_stats.select(".*mean")))

    # Apply the scaling and mask off pixels that didn't have enough neighbors.
    count = common.reduceNeighborhood(ee.Reducer.count(), kernel, 'kernel', True, "boxcar")
    scaled = fill.multiply(scale).add(offset) \
      .updateMask(count.gte(MIN_NEIGHBORS))

    return src.unmask(scaled, True)

def fill_per_path(imgCol,path):
    paths = imgCol.filter(ee.Filter.eq('WRS_PATH', path))
    distinctRows = paths.distinct(['WRS_ROW']).aggregate_array('WRS_ROW')
    def fill_per_row(row):
        images = paths.filter(ee.Filter.eq('WRS_ROW', row)).sort('CLOUD_COVER')
        images_masked=images.map(update_qamask)
        images_list = images.toList(images.size())
        images_masked_list=images_masked.toList(images_masked.size())
        
        images_first=ee.Image(images_list.get(0))
        
        result = GapFill(ee.Image(images_masked_list.get(0)), ee.Image(images_masked_list.get(1)), 10, True)
        
        mask_B1=images_first.select(0).rename(['source_mask']).unmask()
        qa_pixel_0=images_first.select(['QA_PIXEL']).rename(['qa_pixel_0'])
        qa_radsat_0=images_first.select(['QA_RADSAT']).rename(['qa_radsat_0'])
        
        qa_pixel_1=ee.Image(images_list.get(1)).select(['QA_PIXEL']).rename(['qa_pixel_1'])
        qa_radsat_1=ee.Image(images_list.get(1)).select(['QA_RADSAT']).rename(['qa_radsat_1'])
        
        result = ee.List([result.addBands([mask_B1.gt(0.0),qa_pixel_0,qa_radsat_0,qa_pixel_1,qa_radsat_1])])
        
        def GapFill_iteration(element,med_result):
            fill_bands=['Blue','Green','Red','NIR','SWIR1','SWIR2','ST','SR_ATMOS_OPACITY','SR_CLOUD_QA',
                        'ST_ATRAN','ST_CDIST','ST_DRAD','ST_EMIS','ST_EMSD','ST_QA','ST_TRAD','ST_URAD']
            image=ee.Image(element)
            index=images_masked_list.indexOf(element)
            image_nomask=ee.Image(images_list.get(index))
            qa_pixel_name=ee.String.cat('qa_pixel_',index.format())
            qa_radsat_name=ee.String.cat('qa_radsat_',index.format())
            iter_result = ee.Image(GapFill(ee.Image(ee.List(med_result).get(-1)).select(fill_bands), image.select(fill_bands), 10, True))
            new_bands=ee.Image(ee.List(med_result).get(-1)).slice(17)
            qa_pixel=image_nomask.select(['QA_PIXEL']).rename(qa_pixel_name)
            qa_radsat=image_nomask.select(['QA_RADSAT']).rename(qa_radsat_name)
            final_result=ee.Image.cat([iter_result, new_bands, qa_pixel,qa_radsat])
            return ee.List(med_result).add(final_result)
        
        result = images_masked_list.slice(2).iterate(GapFill_iteration, result)

        return ee.Image(ee.List(result).get(-1))

    imagePerRow = distinctRows.map(fill_per_row)
    return imagePerRow

def get_gap_filled_image(imgCol):
    distinctPaths = ee.List(imgCol.distinct(['WRS_PATH']).aggregate_array('WRS_PATH'))
    imagePerPath = distinctPaths.map(lambda path: fill_per_path(imgCol,path))
    gap_filled_imgCol = ee.ImageCollection.fromImages(imagePerPath.flatten())
    return gap_filled_imgCol

def get_band_size(element):
    image=ee.Image(element)
    band_size=image.bandNames().size()
    return band_size

def get_pretreated_composite(img_list):
#    img_list = get_gap_filled_image(imgCol)
    band_size_list=img_list.map(get_band_size)
    print(band_size_list)
    max_index=ee.Array(band_size_list).argmax().get(0)
    update_img_list=ee.List([])
    list_size=img_list.size().getInfo()
    print(list_size)
    for i in range(list_size):
        image=ee.Image(img_list.get(i))
        band_size=image.bandNames().size()
        distance=ee.Number(band_size_list.get(max_index)).subtract(ee.Number(image.bandNames().size())).getInfo()
        for n in reversed(range(distance)):
            none_mask=ee.Image().eq(0)
            new_band=ee.Image().updateMask(none_mask).toDouble()\
                    .rename(ee.Image(img_list.get(max_index))\
                    .select(ee.Number(band_size_list.get(max_index)).subtract(n+1)).bandNames())
            print(new_band)
            image=ee.Image.cat([image,new_band])
        update_img_list=update_img_list.add(image)
    update_imgCol=ee.ImageCollection.fromImages(update_img_list)
    update_img=update_imgCol.median().reproject(imgCol.first().projection()).unmask(-9999)
    
    return update_img


def rescale_modis(image):
    return image.multiply(0.02).copyProperties(image, image.propertyNames())

def clip_collection(image, geometry):
    return image.clip(geometry)

def reproject_collection(image):
    return image.reproject('EPSG:32612',scale=1000)

def preprocess_modis(imgCol, start_date: str, end_date: str, geometry: ee.FeatureCollection = None):
    filtered_imgCol = filter_collection(imgCol,start_date=start_date, end_date=end_date,geometry=geometry)
    return filtered_imgCol.select('LST_Day_1km')\
            .map(rescale_modis)\
            .map(reproject_collection)\
            .map(lambda image: clip_collection(image, geometry))