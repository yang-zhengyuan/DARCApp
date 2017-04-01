import numpy as np
from osgeo import gdal
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
    out = gdal.GetDriverByName('GTiff').Create(fout, cols, rows, 1, outtype, options=options)
    out.SetGeoTransform(geo_transform)
    band = out.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    if color_table:
        band.SetRasterColorTable(color_table)
    band.WriteArray(data)
    out.FlushCache()



