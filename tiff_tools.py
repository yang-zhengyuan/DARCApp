import numpy as np
from osgeo import gdal, osr
from matplotlib import pyplot as plt


def read_array_from_tif(fin, band=1):
    tiff = gdal.Open(fin)
    return np.array(tiff.GetRasterBand(band).ReadAsArray)


def get_output_cmap(data, cmap, mincolor=None, maxcolor=None):
    maxval = data.max()
    if not maxcolor:
        maxcolor = maxval
    if maxval <= 1:
        return None
    ncolors = maxcolor + 1
    colordiff = maxcolor
    if mincolor:
        colordiff -= mincolor
        ncolors -= mincolor
    dcolor = 255.0 / (ncolors - 1)
    color_table = gdal.ColorTable()
    for i in xrange(colordiff):
        color = int(i * dcolor)
        if maxcolor and (i > maxcolor or (mincolor and i + mincolor > maxcolor)):
            color = int(i * maxcolor)
        color = tuple(map(lambda x: int(x * 255),
                          list(cmap(color)).append(i + 1.0 / ncolors)))
        if mincolor:
            color_table.SetColorEntry(int(i + mincolor), color)
        else:
            color_table.SetColorEntry(int(i), color)
    return color_table


def write_array_to_tiff(data, fout, params, dtype=np.uint16, cmap=plt.cm.jet(), nodata=-1 * 2**8 + 1, epsg='4326', maxcolor=None, mincolor=None):
    if dtype == np.uint16:
        outtype = gdal.GDT_UInt16
    elif dtype == np.uint32:
        outtype = gdal.GDT_UInt32
    elif dtype == np.float32:
        outtype = gdal.GDT_Float32
    elif dtype == np.uint8:
        outtype = gdal.GDT_Byte
    else:
        raise Exception('unsupported datatype', dtype)
    data = np.choose(data > 0, (0, data)).astype(dtype)
    color_table = None
    if dtype in (np.uint8, np.uint16):
        color_table = get_output_cmap(data, cmap, mincolor, maxcolor)
    minx, maxy, maxx, miny = params
    rows, cols = np.shape(data)
    xres = (maxx - minx) / float(cols)
    yres = (maxy - miny) / float(rows)
    geo_transform = (minx, xres, 0, maxy, 0, -yres)
    options = ['COMPRESS=DEFLATE', 'TILED=YES']
    out = gdal.GetDriverByName('GTiff').Create(
        fout, cols, rows, 1, outtype, options=options)
    out.SetGeoTransform(geo_transform)
    band = out.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    if color_table:
        band.SetRasterColorTable(color_table)
    band.WriteArray(data)
    out.FlushCache()


# The following method translates given latitude/longitude pairs into pixel locations on a given GEOTIF
# INPUTS: geotifAddr - The file location of the GEOTIF
#      latLonPairs - The decimal lat/lon pairings to be translated in the form [[lat1,lon1],[lat2,lon2]]
# OUTPUT: The pixel translation of the lat/lon pairings in the form [[x1,y1],[x2,y2]]
# NOTE:   This method does not take into account pixel size and assumes a high enough
#         image resolution for pixel size to be insignificant
# credit goes to a now unknown stack overflow post, but as seen here https://github.com/madhusudhanaReddy/WeatherPrediction/blob/master/weatherprediction.py
def latLonToImagePixel(geotifAddr, latLonPairs):
    # Load the image dataset
    ds = gdal.Open(geotifAddr)
    # Get a geo-transform of the dataset
    gt = ds.GetGeoTransform()
    # Create a spatial reference object for the dataset
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    # Set up the coordinate transformation object
    srsLatLong = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srsLatLong, srs)
    # Go through all the point pairs and translate them to latitude/longitude
    # pairings
    pixelPairs = []
    for point in latLonPairs:
        # Change the point locations into the GeoTransform space
        point = list(point)
        try:
            (point[1], point[0], holder) = ct.TransformPoint(
                point[1], point[0])
        except Exception:
            import traceback
            traceback.print_exc()
            raise
        # Translate the x and y coordinates into pixel values
        x = (point[1] - gt[0]) / gt[1]
        y = (point[0] - gt[3]) / gt[5]
        # Add the point to our return array
        pixelPairs.append([int(y), int(x)])
    return pixelPairs


def pixelToLatLon(geotifAddr, pixelPairs):
    # Load the image dataset
    ds = gdal.Open(geotifAddr)
    # Get a geo-transform of the dataset
    gt = ds.GetGeoTransform()
    # Create a spatial reference object for the dataset
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    # Set up the coordinate transformation object
    srsLatLong = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs, srsLatLong)
    # Go through all the point pairs and translate them to pixel pairings
    latLonPairs = []
    for point in pixelPairs:
        # Translate the pixel pairs into untranslated points
        ulon = point[0] * gt[1] + gt[0]
        ulat = point[1] * gt[5] + gt[3]
        # Transform the points to the space
        try:
            (lon, lat, holder) = ct.TransformPoint(ulon, ulat)
        except Exception:
            lat, lon = ulat, ulon
        # Add the point to our return array
        latLonPairs.append([lat, lon])
    return latLonPairs
