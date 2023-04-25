from osgeo import gdal, osr
import pandas as pd
import os
import numpy as np


def process_bar(percent, start_str='', end_str='100%', total_length=0, udata=''):
    bar = '#'.join(["\033[31m%s\033[0m" % ''] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent * 100) + end_str + ' ' + udata
    print(bar, end='', flush=True)


def get_geo_info(filename):
    data = gdal.Open(filename)
    geoinfo = data.GetGeoTransform()
    #print(geoinfo)
    x_size = data.RasterXSize
    y_size = data.RasterYSize
    x1 = geoinfo[0]
    y1 = geoinfo[3]
    x2 = x1 + geoinfo[1] * x_size
    y2 = y1 + geoinfo[5] * y_size
    return x1, y1, x2, y2, x_size, y_size


def lat_lon_to_pixel(raster_dataset, location):
    ds = raster_dataset
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srs_lat_lon = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs_lat_lon, srs)
    new_location = [None, None]
    (new_location[0], new_location[1], holder) = ct.TransformPoint(location[0], location[1])
    x = (new_location[0] - gt[0]) / gt[1]  # lon
    y = (new_location[1] - gt[3]) / gt[5]  # lat
    return int(x), int(y)


def getcsv(path, savename='save_geo_info.csv'):
    info = []
    for patch in os.listdir(path):
        for f in os.listdir(os.path.join(path, patch)):
            if '.tif' in f and '.xml' not in f:
                x1, y1, x2, y2, xsize, ysize = get_geo_info(os.path.join(path, patch, f))
                info.append([os.path.join(path, patch, f), x1, y1, x2, y2, xsize, ysize])
    info = np.array(info)
    df = pd.DataFrame(columns=['name', 'x1', 'y1', 'x2', 'y2', 'xsize', 'ysize'], data=info)
    df.to_csv(savename, index=None)


def createdata(geoinfopath, pointpath):
    imgs = np.array(pd.read_csv(geoinfopath))
    points = np.array(pd.read_csv(pointpath))
    # print(points.shape)
    res = []
    for i, point in enumerate(points):
        process_bar(i / points.shape[0])
        for img in imgs:
            path = img[0]
            # loc = point[1:]
            # print(point['lon'])
            loc = [point[-2], point[-1]]
            # print(loc)
            if loc[0] >= img[1] and loc[0] <= img[3] and loc[1] <= img[2] and loc[1] >= img[4]:
                image = gdal.Open(path)
                a, b = lat_lon_to_pixel(image, loc)

                data = image.ReadAsArray(a, b, 1, 1)[:, 0, 0]
                if len(data)==70:
                    # print(data)
                    data = data.reshape([7,10],order='C')
                    data = data[:,[0,1,2,3,4,6,8,9]]
                    data = data.flatten()
                    # print(data)
                res.append(data)
                continue

    res = np.array(res)
    return res

def npy2csv(d,name,value):
    data = np.load(d)
    print(data.shape)
    title = ["OBJECT_ID", "class_id", ]
    title = [str(i + 1) for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=title)
    df.insert(0,"class_id",value)
    df.insert(0, "OBJECT_ID", 99)
    # print(df.head(5))
    df.to_csv(name,index=None)

if __name__ == '__main__':

    #name = "save_geo_info_huabei2020.csv"
    name = "save_geo_info_dongbei2020.csv"
    # 1. 生成geo文件
    # getcsv("/DATA_HUABEI/2020", name)
    # getcsv("/data12t/maize/dongbei_2019", name)

    # 2. 读取
    #maizes = createdata(name, "maize.csv")
    #np.save('maize.npy', maizes)
    #npy2csv("maize.npy", "maize_extract.csv", 0)
    #maizes = createdata(name, "nomaize.csv")
    #np.save('nomaize.npy', maizes)
    #npy2csv("nomaize.npy", "nomaize_extract.csv", 1)
    maizes = createdata(name, "other_reduced.csv")
    np.save('other_reduced.npy', maizes)
    npy2csv("other_reduced.npy", "other_extract.csv", 2) 
    
    
    
    
    # nomaizes = createdata(name, "nonmaize.csv")
    #
    # print(maizes.shape, nomaizes.shape)
    # data = np.concatenate((maizes, nomaizes), axis=0)
    # print(maizes.shape, nomaizes.shape, data.shape)
    # np.save('data2.npy', data)

    # 3. npy to csv
    # npy2csv("data2.npy","data2.csv")


    # maizes = createdata('maizes.csv')
    # nomaizes = createdata('nomaizes.csv')
    # print(maizes.shape, nomaizes.shape)
    # choiceindex = np.random.choice(nomaizes.shape[0], maizes.shape[0], replace=False)
    # nomaizes = nomaizes[choiceindex, :]
    #
    # data = np.concatenate((maizes, nomaizes), axis=0)
    # print(maizes.shape, nomaizes.shape, data.shape)
    # np.save('data2.npy', data)

    # getcsv('/root/maize/hebei/2020')


