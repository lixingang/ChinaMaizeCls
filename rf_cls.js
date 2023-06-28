var s2Tools = require("users/nanshany1993/common:sentinel2");
var oeel=require('users/OEEL/lib:loadAll')
// var xs=locs.HUABEIX
// var ys=locs.HUABEIY
var xs = [
101,102,98,99,100,101,102,97,98,99,100,101,102,103,98,99,100,101,102,103,98,99,100,101,102,103,104,105,98,99,100,101,102,103,
104,105,106,107,98,99,100,101,102,103,104,105,106,107,108,98,99,100,101,102,103,104,105,106,107,98,99,100,101,102,103,104,105,
106,107,108,109,110,100,101,102,103,104,105,106,107,108,109,110,111,99,100,101,102,103,104,105,106,107,108,109,110,111,112,99,100,
101,102,103,104,105,106,107,108,109,110,111,112,99,100,101,102,103,104,105,106,107,108,109,110,111,112,98,99,100,101,102,103,104,105,106,107,108,109,110,111,98,99,100,101,102,103,104,105,106,107,108,109,110,111,99,100,101,102,103,104,105,106,107,108,109,99,100,101,103,104,105,106,107,108,109,105,106,
107,108,101,102,87,88,98,99,100,101,102,103,84,85,86,87,88,98,99,100,101,102,103,82,83,84,85,86,87,88,89,90,93,94,95,96,97,98,99,
100,101,102,103,79,80,81,82,83,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,101,102,79,80,81,82,83,84,85,86,87,88,89,90,91,92,
93,94,95,96,97,98,99,100,101,102,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,81,82,83,84,85,86,
87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,
103,104,105,106,107,108,109,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,
109,110,111,112,113,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,103,104,105,106,107,108,109,110,112,113,114,115,116,117,118,119,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,105,106,107,108,109,110,111,113,114,115,87,88,89,90,91,92,93,94,95,96,97,98,107,108,109,110,87,88,89,90,91,92,93,94,95,96,97,98,88,89,90,91,92,93,94,95,96,97,98,99,100,89,90,91,92,93,94,95,96,97,98,99,100,92,93,94,95,96,97,98,99,100,94,95,96,97,98,99,100,96,97,98,99,99
]

var ys = [
51,51,52,52,52,52,52,53,53,53,53,53,53,53,54,54,54,54,54,54,55,55,55,55,55,55,55,55,56,56,56,56,56,56,56,56,56,56,57,57,57,57,57,57,57,
57,57,57,57,58,58,58,58,58,58,58,58,58,58,59,59,59,59,59,59,59,59,59,59,59,59,59,60,60,60,60,60,60,60,60,60,60,60,60,61,61,61,61,61,61,61,
61,61,61,61,61,61,61,62,62,62,62,62,62,62,62,62,62,62,62,62,62,63,63,63,63,63,63,63,63,63,63,63,63,63,63,64,64,64,64,64,64,64,64,64,64,64,
64,64,64,65,65,65,65,65,65,65,65,65,65,65,65,65,65,66,66,66,66,66,66,66,66,66,66,66,67,67,67,67,67,67,67,67,67,67,68,68,68,68,39,39,40,40,
40,40,40,40,40,40,41,41,41,41,41,41,41,41,41,41,41,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,43,43,43,43,43,43,43,43,43,
43,43,43,43,43,43,43,43,43,43,43,43,43,43,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,45,45,45,45,45,45,45,45,
45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,47,47,47,47,
47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,
48,48,48,48,48,48,48,48,48,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,50,50,50,50,50,
50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,
51,51,51,51,51,51,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,58,58,58,58,58,58,58,58,58,58,58,58,59,59,59,59,59,59,59,59,59,59,59,59,59,60,60,60,60,60,60,60,60,60,60,60,60,61,61,61,61,61,61,61,61,61,62,62,62,62,62,62,62,63,63,63,63,64



]

var Tyear=2021

//读取数据
var Sample = ee.FeatureCollection('users/lixg95/TrainLabelsNew/HB_NEW')
var fishnet = ee.FeatureCollection('users/lixg95/ChinaFishnet100150_sel')
var bandnames_ex = ['blue','green', 'red','red1','red2','nir','swir1','swir2','ndvi','evi']


//提取训练数据范围的哨兵数据
var image=getdata(2020,geo1)
Map.addLayer(image,{},'image')

//定义随机森林
var classifier  = ee.Classifier.smileRandomForest({
numberOfTrees: 200})

var split=1000


//切分出训练集并进行训练
var split=1000
var len=5320

// var input=Sample.filter(ee.Filter.rangeContains('ID',split*i,split*(i+1)))
//提取数据
var withRandom = Sample.randomColumn('random');

var split = 0.7; 
var trainingPartition = withRandom.filter(ee.Filter.lt('random', split));
var testingPartition = withRandom.filter(ee.Filter.gte('random', split));
var trainingPartition = trainingPartition.randomColumn('Index');
var testingPartition = testingPartition.randomColumn('Index');

for(var i=0;i<10;i=i+0.1)
{
  var samples = trainingPartition.filter(
    ee.Filter.and(ee.Filter.gte('Index', i), ee.Filter.lt('Index', i+0.1)) 
  );
  var training = image.sampleRegions({     
    collection: samples,
     properties: ['class_int'],
     scale: 10,
     tileScale:8
  })
  //训练并赋值于自身
  classifier=classifier.train({
   features:training,
   classProperty:'class_int',
   //inputProperties:bands
  })
}
var trainAccuracy = classifier.confusionMatrix();
print('[train] Resubstitution error matrix: ', trainAccuracy);
print('[train] Training overall accuracy: ', trainAccuracy.accuracy());

var testing =  image.sampleRegions({     
    collection: testingPartition,
     properties: ['class_int'],
     scale: 10,
     tileScale:6
  })
var test = testing.classify(classifier);
print(test)
// 制作混淆矩阵，并打印出来
var testAccuracy = test.errorMatrix('class_int', 'classification');
print('[test] Resubstitution error matrix: ', testAccuracy);
print('[test] Testing overall accuracy: ', testAccuracy.accuracy());
print('[test] consumersAccuracy',testAccuracy.consumersAccuracy());
print('[test] producersAccuracy',testAccuracy.producersAccuracy());
print('[test] kappa',testAccuracy.kappa());
//在这里添加批处理
// var mid = 250
// xs = [21,22,23,31,31,28,29,18,19]
// ys = [70,70,70,73,74,68,68,65,65]

print("length:",xs.length);
for(var i=0;i<xs.length;i++)
{
  //97 49
    download(xs[i],ys[i])
    // download(37,73)
    break;
}

function download(X,Y){
  var name=X.toString()+'_'+Y.toString()
  //获取要分类的区域
  var region=fishnet
       .filter(ee.Filter.and(ee.Filter.eq('idx',X),ee.Filter.eq('idy',Y)))
       .first()
       .geometry();
  //获取这一区域的哨兵数据  
  var input=getdata(Tyear,region).clip(region)
  //用训练得到的随机森林分类
  var classified=input.classify(classifier )
  //1:maize;2:nomaize,3:other
  //Map.addLayer(classified,{min:1,max:3,palette:['red','green','blue']},name,false)
  //save
  Export.image.toDrive({
    image:classified.toUint8(),
    description:Tyear+'_PRO_'+name,
    fileNamePrefix:Tyear+'_PRO_'+name,
    region:region,
    scale:10,
    folder:Tyear+'_HUABEI',
    crs:'EPSG:4326',
    //fileDimensions:,
    maxPixels:1e13,
    fileFormat:'GeoTIFF'
  })
  
   
}

//function of create data
function getdata(year,geo){

  var startDay = ee.Date.fromYMD(year,1,1)
  var endDay = ee.Date.fromYMD(year+1,1,1)
  var s2 = ee.ImageCollection("COPERNICUS/S2")
    .filterBounds(geo)
    // .filterBounds(aoi)
    .filterDate(startDay,endDay)
    .map(sentinel2toa)
    // .map(addVariables) 
    // .map(addTimeBands)
    .map(cloudMask)
    
  var s2filtered = s2.select(bandnames_ex)
  
  // 10-day time series via three steps
  // 1) 10-day composite
  var startDoy = startDay.getRelative('day','year')
  var endDoy = endDay.advance(-1,'day').getRelative('day','year')
  var starts = ee.List.sequence(startDoy, endDoy, 10)
  var composites = ee.ImageCollection(starts.map(function(start) {
    var doy = start
    var filtered = s2filtered.filter(ee.Filter.dayOfYear(start, ee.Number(start).add(10))).median().clip(geo)
    var bandLength = filtered.bandNames().length()
    var mask = ee.Algorithms.If({                   // mask must be done for time band
      condition : ee.Number(bandLength).gt(0),
      trueCase : filtered.select(0).mask(),
      falseCase : ee.Image(0).clip(geo)    
    })
    return filtered
                  .addBands(ee.Image.constant(doy).rename('doy').float())
                  .updateMask(mask)
                  .set('system:time_start',ee.Date.fromYMD(year,1,1).advance(doy,'day').millis())
                  .set('doy',doy)
                  .set('length',bandLength)   
    }));
    
  
    
  // 2) Linear interpolation --------------------------------------------
  var size = composites.size()
  var LIC = composites.toList(size)
  // print('composites',LIC)
  var interpolated = ee.ImageCollection(ee.List.sequence(9,30,1).map(function(i){
    i = ee.Number(i)
    var before = ee.ImageCollection.fromImages(LIC.slice(i.subtract(9),i))
      .filter(ee.Filter.gt('length',0)).mosaic()
    var after = ee.ImageCollection.fromImages(LIC.slice(i.add(1),i.add(10)).reverse())
      .filter(ee.Filter.gt('length',0)).mosaic()
    var boforeY = before.select(bandnames_ex)
    var beforedoy = before.select('doy')
    var afterY = after.select(bandnames_ex)
    var afterdoy = after.select('doy')
    var targetImg = ee.Image(LIC.get(i))
    var currentdoy = ee.Image.constant(targetImg.get('doy')).float();
    var Y = afterY.subtract(boforeY).divide(afterdoy.subtract(beforedoy))
        .multiply(currentdoy.subtract(beforedoy)).add(boforeY)
    var filledImage = ee.Image(ee.Algorithms.If({
      condition : ee.Number(targetImg.get('length')).gt(0), 
      trueCase : targetImg.select(bandnames_ex).unmask(Y),
      falseCase : Y
    }));
    return filledImage.unmask(0).clip(geo)
      .set('system:time_start',targetImg.get('system:time_start'),'doy',targetImg.get('doy')) // can not simply copy all properties of composites
  }))  
  // print('interpolated',interpolated)
  
  var i_size = interpolated.size()
  var i_LIC = interpolated.toList(i_size)
  interpolated =  ee.ImageCollection(ee.List.sequence(1,21,3).map(function(i){
    i = ee.Number(i)
    return i_LIC.get(i);
  }))
  // print(interpolated.mosaic())
  
  var data = interpolated.map(toa_2AtoInt).toBands()
  
  // print(data.bandNames())
  return data.select(data.bandNames(), 
    [ 'blue','green','red','red1','red2','nir','swir1','swir2','ndvi','evi',
    'blue_1','green_1','red_1','red1_1','red2_1','nir_1','swir1_1','swir2_1','ndvi_1','evi_1',
    'blue_2','green_2','red_2','red1_2','red2_2','nir_2','swir1_2','swir2_2','ndvi_2','evi_2',
    'blue_3','green_3','red_3','red1_3','red2_3','nir_3','swir1_3','swir2_3','ndvi_3','evi_3',
    'blue_4','green_4','red_4','red1_4','red2_4','nir_4','swir1_4','swir2_4','ndvi_4','evi_4',
    'blue_5','green_5','red_5','red1_5','red2_5','nir_5','swir1_5','swir2_5','ndvi_5','evi_5',
    'blue_6','green_6','red_6','red1_6','red2_6','nir_6','swir1_6','swir2_6','ndvi_6','evi_6', ]
    )
}




function sentinel2toa (img){
  return img.select(
                      ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10', 'B11','B12','QA60']
                      ,['aerosol', 'blue', 'green', 'red','red1','red2','red3','nir','red4','h2o', 'cirrus','swir1', 'swir2','QA60']
                    )
                    .addBands(img.normalizedDifference(['B8', 'B4']).toDouble().rename('ndvi'))
                    .addBands(img.expression('2.5*((B8-B4)/(B8+6*B4-7.5*B2+1))', {
                        'B8':img.select('B8'),
                        'B4':img.select('B4'),
                        'B2':img.select('B2')
                      }).toDouble().rename('evi'))
                    .divide(10000).toDouble()
                    .set('solar_azimuth',img.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
                    .set('solar_zenith',img.get('MEAN_SOLAR_ZENITH_ANGLE') )
                    .set('system:time_start',img.get('system:time_start'));
}

// function to get cloud score
// simply the cloud_and_shadow_mask
function cloudMask(toa) {
  // authors: Matt Hancher, Chris Hewig and Ian Housman
  
  function rescale(img, thresholds) {
    return img.subtract(thresholds[0]).divide(thresholds[1] - thresholds[0]);
  }
  
  // Compute several indicators of cloudyness and take the minimum of them.
  var score = ee.Image(1);
  
  //Clouds are reasonably bright
  score = score.min(rescale(toa.select(['blue']), [0.1, 0.5]));
  score = score.min(rescale(toa.select(['aerosol']), [0.1, 0.3]));
  score = score.min(rescale(toa.select(['aerosol']).add(toa.select(['cirrus'])), [0.15, 0.2]));
  score = score.min(rescale(toa.select(['red']).add(toa.select(['green'])).add(toa.select('blue')), [0.2, 0.8]));

  //Clouds are moist
  var ndmi = toa.normalizedDifference(['red4','swir1']);
  score=score.min(rescale(ndmi, [-0.1, 0.1]));
  
  // However, clouds are not snow.
  var ndsi = toa.normalizedDifference(['green', 'swir1']);
  score=score.min(rescale(ndsi, [0.8, 0.6]));
  
  // a (somewhat arbitrary) threshold 
  var cloudScoreThreshold = 0.2;
  var cloud = score.gt(cloudScoreThreshold);
  
  var mask = cloud.eq(0);
  return toa.updateMask(mask);
} 

function toa_2AtoInt(img){
  return img.select( bandnames_ex
                    )
                    .multiply(10000)
}



