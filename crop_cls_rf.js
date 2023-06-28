// Crop map generation
// Take Sanjiang Plain (SJ) in 2019 as an example
var region = 
    ee.Geometry.Polygon(
        [[[110.723211970525, 53.83574455590475],
          [110.723211970525, 38.25035379987694],
          [135.596258845525, 38.25035379987694],
          [135.596258845525, 53.83574455590475]]], null, false);
var s2Tools = require("users/nanshany1993/common:sentinel2");

var id = "210100"
// var ids = 
// [210102,210103,210104,210105,210106,210111,210112,210113,210114,210122,210123,210124,
// 210181,210202,210203,210204,210211,210212,210213,210224,210281,210282,210283,210302,
// 210303,210304,210311,210321,210323,210381,210402,210403,210404,210411,210421,210422,
// 210423,210502,210503,210504,210505,210521,210522,210602,210603,210604,210624,210681,
// 210682,210702,210703,210711,210726,210727,210781,210782,210802,210803,210804,210811,
// 210881,210882,210902,210903,210904,210905,210911,210921,210922,211002,211003,211004,
// 211005,211011,211021,211081,211102,211103,211121,211122,211202,211204,211221,211223,
// 211224,211281,211282,211302,211303,211321,211322,211324,211381,211382,211402,211403,
// 211404,211421,211422,211481,220102,220103,220104,220105,220106,220112,220122,220181,
// 220182,220183,220202,220203,220204,220211,220221,220281,220282,220283,220284,220302,
// 220303,220322,220323,220381,220382,220402,220403,220421,220422,220502,220503,220521,
// 220523,220524,220581,220582,220602,220605,220621,220622,220623,220681,220702,220721,
// 220722,220723,220724,220802,220821,220822,220881,220882,222401,222402,222403,222404,
// 222405,222406,222424,222426,230102,230103,230104,230108,230109,230110,230111,230112,
// 230123,230124,230125,230126,230127,230128,230129,230182,230183,230184,230202,230203,
// 230204,230205,230206,230207,230208,230221,230223,230224,230225,230227,230229,230230,
// 230231,230281,230302,230303,230304,230305,230306,230307,230321,230381,230382,230402,
// 230403,230404,230405,230406,230407,230421,230422,230502,230503,230505,230506,230521,
// 230522,230523,230524,230602,230603,230604,230605,230606,230621,230622,230623,230624,
// 230702,230703,230704,230705,230706,230707,230708,230709,230710,230711,230712,230713,
// 230714,230715,230716,230722,230781,230801,230822,230826,230828,230833,230881,230882,
// 230902,230903,230904,230921,231002,231003,231004,231005,231024,231025,231081,231083,
// 231084,231085,231102,231121,231123,231124,231181,231182,231202,231221,231222,231223,
// 231224,231225,231226,231281,231282,231283,232721,232722,232723,150425,150524,150581,
// 150702,150724,150725,150726,150727,150781,150782,150784,150785,152202,152502,152522,
// 152523,152525,152526,152527,152529,152530,152531,150402,150403,150404,150421,150422,
// 150423,150424,150426,150428,150429,150430,150502,150521,150522,150523,150525,150526,
// 150721,150722,150723,150783,152201,152221,152222,152223,152224]

var ids = ["210100","210200","210300","210400","210500","210600","210700","210800","210900","211000","211100","211200",
"211300","211400","220100","220200","220300","220400","220500","220600","220700","220800","222400","230100","230200",
"230300","230400","230500","230600","230700","230800","230900","231000","231100","231200","232700","150400","150500",
"150700","152200","152500","150400","150500","150700","152200"]

var fenqu = ee.FeatureCollection("users/lixg95/FishnetAndProvinces/ResearchRegion")

var bands = ee.List(['red2','swir1','swir2','NDVI','EVI','LSWI','NDSVI','NDTI','RENDVI','REP'])  // 10 index
// var bands = ee.List(['blue', 'green', 'red','red1','red2','red3','nir','red4','swir1', 'swir2'])
var aoi = fenqu.filter(ee.Filter.eq('CITYCODE',id))
Map.addLayer(aoi)
// Step 1: Construct feature candicates from Sentinel-2 images
var year = 2017
var startDay = ee.Date.fromYMD(year,1,1)
var endDay = ee.Date.fromYMD(year+1,1,1)
var s2 = ee.ImageCollection("COPERNICUS/S2")
  .filterBounds(region)
  // .filterBounds(aoi)
  .filterDate(startDay,endDay)
  .map(sentinel2toa)
  .map(addVariables) 
  // .map(addTimeBands)
  .map(cloudMask)
var s2filtered = s2.select(bands)

// 10-day time series via three steps
// 1) 10-day composite
var startDoy = startDay.getRelative('day','year')
var endDoy = endDay.advance(-1,'day').getRelative('day','year')
var starts = ee.List.sequence(startDoy, endDoy, 10)
var composites = ee.ImageCollection(starts.map(function(start) {
  var doy = start
  var filtered = s2filtered.filter(ee.Filter.dayOfYear(start, ee.Number(start).add(10))).median().clip(region)
  var bandLength = filtered.bandNames().length()
  var mask = ee.Algorithms.If({                   // mask must be done for time band
    condition : ee.Number(bandLength).gt(0),
    trueCase : filtered.select(0).mask(),
    falseCase : ee.Image(0).clip(region)    
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
print('composites',LIC)
var interpolated = ee.ImageCollection(ee.List.sequence(9,30,1).map(function(i){
  i = ee.Number(i)
  var before = ee.ImageCollection.fromImages(LIC.slice(i.subtract(9),i))
    .filter(ee.Filter.gt('length',0)).mosaic()
  var after = ee.ImageCollection.fromImages(LIC.slice(i.add(1),i.add(10)).reverse())
    .filter(ee.Filter.gt('length',0)).mosaic()
  var boforeY = before.select(bands)
  var beforedoy = before.select('doy')
  var afterY = after.select(bands)
  var afterdoy = after.select('doy')
  var targetImg = ee.Image(LIC.get(i))
  var currentdoy = ee.Image.constant(targetImg.get('doy')).float();
  var Y = afterY.subtract(boforeY).divide(afterdoy.subtract(beforedoy))
      .multiply(currentdoy.subtract(beforedoy)).add(boforeY)
  var filledImage = ee.Image(ee.Algorithms.If({
    condition : ee.Number(targetImg.get('length')).gt(0), 
    trueCase : targetImg.select(bands).unmask(Y),
    falseCase : Y
  }));
  return filledImage.unmask(0).clip(region)
    .set('system:time_start',targetImg.get('system:time_start'),'doy',targetImg.get('doy')) // can not simply copy all properties of composites
}))  
print('interpolated',interpolated)

var i_size = interpolated.size()
var i_LIC = interpolated.toList(i_size)
interpolated =  ee.ImageCollection(ee.List.sequence(1,21,3).map(function(i){
  i = ee.Number(i)
  return i_LIC.get(i);
}))
// print(interpolated.mosaic())

var data = interpolated.toBands()
// i_LIC = interpolated.toList(interpolated.size())
// var data = ee.Image(ee.List.sequence(0,interpolated.size(),1).map(function(i){
//   i = ee.Number(i)
//   var targetImg = ee.Image(i_LIC.get(i))
//   return targetImg.addBands(targetImg.select(targetImg.bandNames()))
// }))
// print(data)
// 3) SG smoothing --------------------------------------------

// NDVI/LSWI maximum composite

// var window_size = 7
// var order = 3
// var sgs = s2Tools.sgsmooth(interpolated,bands, order, window_size)
// var NDVIs = sgs.qualityMosaic('NDVI')
// var LSWIs = sgs.qualityMosaic('LSWI')
// var day10s = ee.Image(sgs.iterate(mergeBands, ee.Image([])))
// .addBands(NDVIs).addBands(LSWIs)

// // harmonic regression
// // y = a + b1*cos(3pi*t) + b2*sin(3pi*t) + b3*cons(6pi*t) +b4*sin(6pi*t) 
// var dependent = ee.List(['NDVI','EVI','LSWI',])
// var harmonicIndependents = ee.List(['constant', 'cos3', 'sin3', 'cos6' , 'sin6']);
// // The output of the regression reduction is a [X,Y] array image.
// var harmonic = s2
//   .select(harmonicIndependents.cat(dependent))
//   .reduce(ee.Reducer.linearRegression(harmonicIndependents.length(), dependent.length()));
// var coefficients = harmonic.select('coefficients').matrixTranspose()
//   .arrayFlatten([dependent,harmonicIndependents]).clip(region);  

// merge all
// 10-day time seris  10*22=220
// NDVI/LSWI composite 10*2=20
// harmonic coefficients 3*5=15
// in sum 255
// var finalImage = day10s.addBands(coefficients).updateMask(cropland.clip(region)).clip(region)

// var finalImage = day10s.addBands(coefficients)
// print('finalImage',finalImage,10*24+15) // 255

// var finalImage = day10s
// print('finalImage',finalImage,10*24+15) // 255

// Step 2: Train the classifier using the training samples and selected features
// trainTables is the training dataset, which contains the crop type and Sentinel-2 features of each crop training sample
var cropFeatures = ['swir1_6', 'LSWI_4', 'swir2_5', 'LSWI_6', 'REP_11', 'REP_12', 'REP_23', 'REP_13', 'REP_14', 'REP_16', 'RENDVI_22', 'swir2_22', 'LSWI_10', 'NDTI_4', 'RENDVI_23', 'REP_9', 'LSWI_12', 'swir1_22', 'LSWI_9']

var classNames =  ee.FeatureCollection("users/lixg95/TrueLabels/CDLsamples")
Map.addLayer(classNames)
// print(data)
var training = data.sampleRegions({
  collection: classNames,
  properties: ['class'],
  scale: 10,
  tileScale:2
});
print(training)
//精度评价 
var withRandom = training.randomColumn('random');//样本点随机的排列

// 保留一些数据进行测试，以避免模型过度拟合。
var split = 0.7; 
var trainingPartition = withRandom.filter(ee.Filter.lt('random', split));//筛选70%的样本作为训练样本
var testingPartition = withRandom.filter(ee.Filter.gte('random', split));//筛选30%的样本作为测试样本
//分类方法选择smileCart() randomForest() minimumDistance libsvm
// var classifier = ee.Classifier.libsvm().train({
//   features: trainingPartition,
//   classProperty: 'crop',
//   inputProperties: data.bandNames()
// });
print(trainingPartition, testingPartition)
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 400,
  minLeafPopulation: 2, 
  seed: 999})
// print("训练用到的特征：",data.bandNames())
var classifier = rf.train({
  features:trainingPartition,
  classProperty:'class',
  inputProperties:cropFeatures,
})

var test = testingPartition.classify(classifier);
  //计算混淆矩阵
  // var confusionMatrix = test.errorMatrix('crop', 'classification');
  // print('confusionMatrix',confusionMatrix);//面板上显示混淆矩阵
  // print('overall accuracy', confusionMatrix.accuracy());//面板上显示总体精度
  // print('comsumers accuracy', confusionMatrix.consumersAccuracy());//面板上显示总体精
  // print('producers accuracy', confusionMatrix.producersAccuracy());//面板上显示总体精
  // print('kappa accuracy', confusionMatrix.kappa());//面板上显示kappa值


// Step 3: classify the image with trained classifier and output
var classified = data.classify(classifier).uint8()
var label = 'crop_' + id.toString()
Export.image.toAsset({
  image: classified,
  description: label,
  assetId: 'users/lixg95/GeeOutput/'+label,
  scale: 10,
  region: region,  
  maxPixels : 1e13
});

//////////////// Functions //////////////////////////////////////////
// rename Sentinel-2 bands
// set necessary properties
function sentinel2toa (img){
  return img.select(
                      ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10', 'B11','B12','QA60']
                      ,['aerosol', 'blue', 'green', 'red','red1','red2','red3','nir','red4','h2o', 'cirrus','swir1', 'swir2','QA60']
                    )
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

// Use this function to add several indices to Sentinel-2 imagery.
function addVariables(image) {
  var DOY = image.date().getRelative('day', 'year')
  var year = image.date().get('year')
  
  return image
    // Add a NDVI band.
    .addBands(image.normalizedDifference(['nir', 'red']).toDouble().rename('NDVI'))
    // Add a EVI band.
    .addBands(image.expression('2.5*((nir-red)/(nir+6*red-7.5*blue+1))', {
      'nir':image.select('nir'),
      'red':image.select('red'),
      'blue':image.select('blue')
    }).toDouble().rename('EVI'))
    // Add a GCVI: Green Chlorophyll Vegetation Index (Guan Kaiyu, Wang Sherrie)
    .addBands(image.expression('nir/green-1',{
      'nir': image.select('nir'),
      'green': image.select('green'),
    }).toDouble().rename('GCVI'))
    // Add a MSAVI2: Modified Soil-adjusted Vegetation Index (Qi et al. (1994b))
    .addBands(image.expression('1/2 * (2*nir + 1 - ((2*nir+1)**2 - 8*(nir-red))**(1/2))',{
      'nir': image.select('nir'),
      'red': image.select('red'),
    }).toDouble().rename('MSAVI2'))  
    
    // Add a LSWI band.
    .addBands(image.normalizedDifference(['nir','swir1']).toDouble().rename('LSWI'))
    // Add a NDWI band.
    .addBands(image.normalizedDifference(['green','nir']).toDouble().rename('NDWI'))
    // Add a NDSI band.
    .addBands(image.normalizedDifference(['green','swir1']).toDouble().rename('NDSI'))
    
    // Add NDSVI: normalized differential senescent vegetation index (Zhong,2014)
    .addBands(image.normalizedDifference(['swir1','red']).toDouble().rename('NDSVI'))
    // Add NDTI: normalized differential tillage index, relates to residue cover (Zhong,2014)
    .addBands(image.normalizedDifference(['swir1','swir2']).toDouble().rename('NDTI'))
    
    // Add S2 red-edge indices (Sen2-Agri)
    // RENDVI = (nir-red2)/(nir+red2)
    // REP = {705+35*(0.5*(red3+red)-red1)/(red2-red1)}/1000
    // PSRI = (red-blue)/red1
    // CRE = red1/nir
    .addBands(image.normalizedDifference(['nir','red2']).toDouble().rename('RENDVI'))
    
    .addBands(image.expression('(705+35*(0.5*(red3+red)-red1)/(red2-red1))/1000',{
      'red3' : image.select('red3'),
      'red2' : image.select('red2'),
      'red1' : image.select('red1'),
      'red' : image.select('red'),
    }).toDouble().rename('REP'))
    
    .addBands(image.expression('(red-blue)/red1',{
      'red': image.select('red'),
      'red1': image.select('red1'), 
      'blue': image.select('blue'), 
    }).toDouble().rename('PSRI'))
    
    .addBands(image.expression('red1/nir',{
      'red1': image.select('red1'),
      'nir': image.select('nir'),
    }).toDouble().rename('CRE'))

    // add a doy band.
    .addBands(ee.Image(DOY).rename('DOY').toDouble())
    // add a year band.
    .addBands(ee.Image(year).rename('Year').toDouble())
    
    .set('DOY',DOY)
}

function mergeBands(image, previous) {
  return ee.Image(previous).addBands(image);
}
