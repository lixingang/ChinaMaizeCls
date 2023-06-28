//--------------------
// 1. IMPORT DATA
//-------------------
var hb = ee.Image("users/lixg95/2019/HB2019_lzw");
var db = ee.Image("users/lixg95/2019/PRO2019");
var xb = ee.ImageCollection([
  ee.Image("users/lixg95/2019/2019_PRO_XJ_NEW"),
  ee.Image("users/lixg95/2019/2019_PRO_GSNX_NEW"),
  ]).qualityMosaic('b1');
var xn = ee.ImageCollection([
  ee.Image("users/lixg95/2019/2019_NANFANG"),
  ]).qualityMosaic('b1');

var dem = ee.Image("USGS/SRTMGL1_003").select('elevation'); 
var glc2020 = ee.Image("users/lixg95/GLC30/GLC30_CHINA_2020").uint8();
var glc2015 = ee.Image("users/lixg95/GLC30/GLC30_CHINA_2015").uint8();

var shapefile = ee.FeatureCollection("users/lixg95/FishnetAndProvinces/CHINA_PROVINCE");   
var label_xb = ee.ImageCollection([
  ee.Image("users/lixg95/XB/label_gansu2019"),
  ee.Image("users/lixg95/XB/label_xinjiang2019")]).qualityMosaic('b1');
var ccp = ee.Image("users/lixg95/ChinaCropPhen1km/maize_ma_2019");
var db = db.multiply(db.eq(2)).divide(db);
// Map.addLayer(db2,{min:0,max:1},"db2")
// db = db.multiply(db.eq(2));
xb = xb.multiply(xb.eq(1));
hb = hb.multiply(hb.eq(1));
xn = xn.multiply(xn.eq(1));
var CHINA = ee.ImageCollection([db,xb,hb,xn]).qualityMosaic('b1');

var slope = ee.Terrain.slope(dem).clip(shapefile).uint8()
slope = slope.multiply(slope.gt(10)).divide(slope);
var prov_list = 
  [
    "110000","120000","130000","140000","150000",
    "210000","220000","230000",
    "310000","320000","330000","340000","350000","360000","370000",
    "410000","420000","430000","440000","450000",
    "500000","510000","520000","530000",
    "610000","620000","630000","640000","650000",
  ]
  
//---------------
// 2. DATA PROCESS
//---------------
//拆解到省
var im_dict = [];  
for(var i in prov_list){
  var t_shp = shapefile.filter(ee.Filter.eq('PROVCODE', prov_list[i]));
  im_dict[prov_list[i]] = CHINA.clip(t_shp).uint8()
}
//处理 
for(var i in prov_list){
  var kernel = ee.Kernel.circle({radius: 1});
  var t_shp = shapefile.filter(ee.Filter.eq('PROVCODE', prov_list[i]))
  var ccp_mask = ccp.gt(0).clip(t_shp)
  ccp_mask = ccp_mask.focal_max({kernel: kernel, iterations: 1}).uint8();
  im_dict[prov_list[i]] = im_dict[prov_list[i]].multiply(ccp_mask);
  Map.addLayer(im_dict[prov_list[i]] )
  }
  
  Export.image.toDrive({
    image:im_dict[prov_list[i]],
    description:prov_list[i]+'_2019',
    fileNamePrefix:prov_list[i]+'_2019',
    region:shapefile.filter(ee.Filter.eq('PROVCODE', prov_list[i])),
    scale:10,
    folder:prov_list[i]+'_2019',
    crs:'EPSG:4326',
    //fileDimensions:,
    maxPixels:1e13,
    fileFormat:'GeoTIFF'
  })
  
  
}
//拼接 
var mosaic_images = [];
for (var n in im_dict) {
  mosaic_images.push(im_dict[n].uint8());
}

var CHINA = ee.ImageCollection.fromImages(mosaic_images).qualityMosaic('b1');

var year = 2019;
var output_name= "TEST_CHINA"+year.toString();


var input_image = CHINA;
// Map.addLayer(image)
//---------------
// 3. DATA REPROJECTION
//---------------


var feature_list = [] 
var wkt = 'PROJCS["Asia_North_Albers_Equal_Area_Conic",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["central_meridian",105],PARAMETER["Standard_Parallel_1",25],PARAMETER["Standard_Parallel_2",47],PARAMETER["latitude_of_origin",0],UNIT["Meter",100]]'
var proj_mollweide = ee.Projection(wkt);
for(var i=0;i<prov_list.length;i++)
{
  var target_shp = shapefile
  .filter(ee.Filter.eq('PROVCODE', prov_list[i]));
  var img = input_image.clip(target_shp)
  img = img.reproject(proj_mollweide)
  var area = img.reduceRegions({
      collection: target_shp,
      reducer: ee.Reducer.sum(),
      scale: 100,
    });
  feature_list.push(area)
}

//---------------
// 4. EXPORT
//---------------
var yearbook =  [35645.99,178861.26,3417100.00,1742220.00,3823900.00,2699308.33,4287242.00,5480666.67,1236.60,509760.00,
  63282.46,1234756.43,33053.93,47600.00,3871090.00,3818010.00,751991.20,384380.00,123140.00,596973.00,440929.20,1839360.00,
  501460.00,1802457.28,1179440.00,1000810.46,21370.00,322730.00,1051050.00,]
var predict = ee.FeatureCollection(feature_list)

predict = predict.flatten().aggregate_array("sum");
predict.evaluate(function(predict_local) {
  var result = linearRegression(predict_local,yearbook)
  print(result)
  var chart = ui.Chart.array.values(predict_local, 0, yearbook)
  print(chart)
  }
)


var Feature_Collections = ee.FeatureCollection(feature_list).flatten()
print(Feature_Collections)
// print(Feature_Collections)
Export.table.toDrive({
  collection: Feature_Collections,
  folder: '_ZonalStat',
  description: output_name,
  selectors:["PROVCODE","PROVCODE_D","PROVNAME","sum"],
  fileFormat: 'CSV'
});

// ------------------
// *.Functions
// ------------------

function linearRegression(y,x){

  var lr = {};
  var n = y.length;
  var sum_x = 0;
  var sum_y = 0;
  var sum_xy = 0;
  var sum_xx = 0;
  var sum_yy = 0;

  for (var i = 0; i < y.length; i++) {

      sum_x += x[i];
      
      sum_y += y[i];
      sum_xy += (x[i]*y[i]);
      sum_xx += (x[i]*x[i]);
      sum_yy += (y[i]*y[i]);
  } 
  // print(n,sum_yy,sum_x,)
  lr['slope'] = (n * sum_xy - sum_x * sum_y) / (n*sum_xx - sum_x * sum_x);
  lr['intercept'] = (sum_y - lr.slope * sum_x)/n;
  lr['r2'] = Math.pow((n*sum_xy - sum_x*sum_y)/Math.sqrt((n*sum_xx-sum_x*sum_x)*(n*sum_yy-sum_y*sum_y)),2);

  return lr;
}
