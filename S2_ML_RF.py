'''
<PBML Final Project>
Sentinel-1 Image Segmentation

@author: hoyeong
'''

#%% Import Library --------------------------------------------
import glob
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from pyproj import Proj
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import geopandas as gpd
import pandas as pd
from skimage import exposure
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#%% Read Sentinel-2 Image --------------------------------------
fp = glob.glob(r'/Users/hoyeong/Documents/PYTHON/PBML/Image/TIFF/*.tif')
fp.sort()

s2_list = []
for img in fp:
    with rasterio.open(img) as src:
            raster_data = {
                'band_data': {'B1': src.read(3),
                              'B2': src.read(4),
                              'B3': src.read(5),
                              'B4': src.read(6),
                              'B5': src.read(7),
                              'B6': src.read(8),
                              'B7': src.read(9),
                              'B8': src.read(10),
                              'B9': src.read(11),
                              'B11': src.read(12),
                              'B12': src.read(13),
                              'NDVI': src.read(14),
                              'NDWI': src.read(15),
                              'REI': src.read(16),
                              'EVI2': src.read(17)},
                'meta': src.meta,
                'bounds': src.bounds,
                'name': src.name.split('/')[-1].split('.')[0]
            }
            s2_list.append(raster_data)

#%% Subset Sentinel-2 Image ------------------------------------

def longlat2window(lon,lat,dataset):
    p = Proj(dataset.crs)
    t = dataset.transform
    xmin, ymin = p(lon[0], lat[0])
    xmax, ymax = p(lon[1], lat[1])
    col_min, row_min = ~t*(xmin, ymin)
    col_max, row_max = ~t*(xmax, ymax)
    return Window.from_slices(rows=(np.floor(row_max), np.ceil(row_min)),
                              cols=(np.floor(col_min), np.ceil(col_max)))


def subset_raster(fp, lon, lat):
    im_list_clip = []
    band_indices = {
        'B1': 3, 'B2': 4, 'B3': 5, 'B4': 6, 'B5': 7, 'B6': 8, 'B7': 9,
        'B8': 10, 'B9': 11, 'B11': 12, 'B12': 13, 'NDVI': 14, 'NDWI': 15, 
        'REI': 16, 'EVI2': 17
    }
    for file in fp:
        with rasterio.open(file) as src:
            window = longlat2window(lon,lat,src)
            new_transform = src.window_transform(window)
            band_data = {band_name: src.read(index, window=window) for band_name, index in band_indices.items()}
            height, width = band_data['B1'].shape
            raster = {
                'band_data': band_data,
                'meta': {
                    'driver': src.meta['driver'],
                    'dtype': src.meta['dtype'],
                    'nodata': {band: np.sum(np.isnan(data)) for band, data in band_data.items()},
                    'width': width,
                    'height': height,
                    'count': 15,
                    'crs': src.meta['crs'],
                    'transform': new_transform,
                },
                'bounds': src.bounds,
                'name': src.name.split('PBML/')[-1].split('.')[0]
            }
            im_list_clip.append(raster)
    return im_list_clip

def match(im_list):
    '''Match shape of arrays in im_list[i]['band_data']['vv/vh_band]'''
    for band in im_list[0]['band_data'].keys():
        min_rows, min_cols = im_list[0]['band_data'][band].shape[0], im_list[0]['band_data'][band].shape[1]

        for j in range(len(im_list)):
            if im_list[j]['band_data'][band].shape[0] < min_rows:
                min_rows = im_list[j]['band_data'][band].shape[0]

            if im_list[j]['band_data'][band].shape[1] < min_cols:
                min_cols = im_list[j]['band_data'][band].shape[1]

        for j in range(len(im_list)):
            im_list[j]['band_data'][band] = im_list[j]['band_data'][band][:min_rows, :min_cols]
            im_list[j]['meta']['width'] = min_cols
            im_list[j]['meta']['height'] = min_rows
    return im_list

# Set AOI_SUB https://geojson.io/#map=2/0/20
aoi_sub = {
        "coordinates": [
          [
            [
              127.23055798548023,
              36.504410803508435
            ],
            [
              127.23055798548023,
              36.46909880041416
            ],
            [
              127.29927687315057,
              36.46909880041416
            ],
            [
              127.29927687315057,
              36.504410803508435
            ],
            [
              127.23055798548023,
              36.504410803508435
            ]
          ]
        ],
        "type": "Polygon"
      }

aoi_sub = {
        "coordinates": [
          [
            [
              127.2849268134674,
              36.585555110317785
            ],
            [
              127.2849268134674,
              36.555772187725154
            ],
            [
              127.32201071768088,
              36.555772187725154
            ],
            [
              127.32201071768088,
              36.585555110317785
            ],
            [
              127.2849268134674,
              36.585555110317785
            ]
          ]
        ],
        "type": "Polygon"
      }

# 좌표 정보 추출
coordinates = aoi_sub['coordinates'][0]

# 좌표의 최소/최대 위경도 구하기
min_lon = min(coord[0] for coord in coordinates)
max_lon = max(coord[0] for coord in coordinates)
min_lat = min(coord[1] for coord in coordinates)
max_lat = max(coord[1] for coord in coordinates)

del s2_list
s2_list = subset_raster(fp, lon=(min_lon, max_lon),
                            lat=(min_lat, max_lat))

# Check Shape of s2_list
for i in range(len(s2_list)):
    print("Image index:", i)
    for band_name, band_data in s2_list[i]['band_data'].items():
        print(f"Band {band_name} shape: {band_data.shape}")

s2_list = match(s2_list)
#%% Visualize Data ---------------------------------------------

def eo_disp(s2_list, band='B4', index=[0]):
    '''
    Parameters:
    -----------------------
    s2_list: image list
    band: Band name ('B1' to 'B12', 'NDVI', 'NDWI', 'REI', 'EVI2')
    index: list of index you want to display
    
    Returns:
    -----------------------
    Sentinel-2 Image Display
    '''
    for i in index:
        disp_band = s2_list[i]['band_data'][band]
        
        # 2% linear contrast stretch
        p2, p98 = np.percentile(disp_band, (2, 98))
        img_rescale = exposure.rescale_intensity(disp_band, in_range=(p2, p98))
        
        fig, axes = plt.subplots()
        axes.imshow(img_rescale, cmap='gray')
        axes.set_title(f'S2_{band}_Image_{i+1}')
        axes.axis('off')
        plt.tight_layout()
        plt.ion()
        plt.show()

def eo_rgb(s2_list, index=[0]):
    '''
    Parameters:
    -----------------------
    s2_list: image list
    bands: list of band names for Red, Green, and Blue channels respectively
    index: list of index you want to display
    
    Returns:
    -----------------------
    Sentinel-2 RGB Image Display
    '''
    bands=['B4', 'B3', 'B2']
    for i in index:
        red = s2_list[i]['band_data'][bands[0]]
        green = s2_list[i]['band_data'][bands[1]]
        blue = s2_list[i]['band_data'][bands[2]]
        
        # Stack bands to create an RGB image
        rgb_img = np.dstack((red, green, blue))
        
        # Scale bands to range 0-255
        p2, p98 = np.percentile(rgb_img, (2, 98)) 
        rgb_img = exposure.rescale_intensity(rgb_img, in_range=(p2, p98))
        rgb_img = (rgb_img * 255).astype(np.uint8)
        
        # Display RGB image
        fig, axes = plt.subplots()
        axes.imshow(rgb_img)
        axes.set_title(f'S2_RGB_Image_{i+1}')
        axes.axis('on')
        plt.tight_layout()
        plt.ion()
        plt.show()

eo_disp(s2_list, band='B2', index=[0])
eo_rgb(s2_list, index=[0,1])

#%% Pandas DataFrame ------------------------------------------

'''Single'''
# Create pandas DataFrame from VV,VH arrays
# df = pd.DataFrame({
#     'vv_band': s1_list[0]['band_data']['vv_band'].flatten(),
#     'vh_band': s1_list[0]['band_data']['vh_band'].flatten(),
#     'vv / vh': s1_list[0]['band_data']['vv / vh'].flatten()})

'''Time Series'''
df_list = []
for i, s2_img in enumerate([s2_list[0]]):
    temp_df = pd.DataFrame()
    for band in s2_img['band_data'].keys():
        temp_df[f'{band}_{i}'] = s2_img['band_data'][band].flatten()
    df_list.append(temp_df)
df = pd.concat(df_list, axis=1)

print(df.head())

#%% Rasterize Shape File --------------------------------------

# Load Shapefile
ref_shp = gpd.read_file(r'/Users/hoyeong/Documents/PYTHON/PBML/Image/Label/Label.shp')

# Convert Shapefile to Raster
# ref_shp['L1_CODE'] = pd.to_numeric(ref_shp['L1_CODE'])

shapes = ((geom,value) for geom, value in zip(ref_shp.geometry, ref_shp.L1_CODE))
rasterized_shp = rasterize(shapes,
                           out_shape=s2_list[0]['band_data']['B1'].shape,
                           transform=s2_list[0]['meta']['transform'],
                           fill=0)
# Count NaN Values
np.count_nonzero(rasterized_shp == 0)

# Extract labels for your training pixels
train_labels = rasterized_shp.flatten()

# Create DataFrame 
df['label'] = train_labels
df = df.copy()

X = df.drop('label',axis=1)
y = df['label']

# Straify (Random Sampling from each Labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)

#%% Hyper-Parameter Tuning ------------------------------------
from sklearn.model_selection import RandomizedSearchCV

params = {'n_estimators' : [100,200,300],
          'max_depth' : [10,20,30,None],
          'min_samples_leaf' : [1,2,4],
          'min_samples_split' : [2,5,10]}

rf_clf = RandomForestClassifier(random_state = 100, n_jobs = -1)
rand_cv = RandomizedSearchCV(rf_clf, param_distributions = params, cv = 5, verbose=3, n_jobs = -1)
rand_cv.fit(X_train, y_train)

print('Best Hyper-Parameters: ', rand_cv.best_params_)
print('Best Accuracy Score: {:.4f}'.format(rand_cv.best_score_))

clf = rand_cv.best_estimator_
#%% If you know parameters ----------------------------------

clf = RandomForestClassifier(n_estimators=100, min_samples_split=20, 
                             min_samples_leaf= 6, max_depth= 20, n_jobs=-1)
clf.fit(X_train, y_train)


#%% Evaluation ------------------------------------------------
y_pred = clf.predict(X_test)
print('Model Accuracy: ', accuracy_score(y_test,y_pred))

pred = clf.predict(X)
print('pred Model Accuracy: ', accuracy_score(y,pred))
rf_result = pred.reshape(s2_list[0]['band_data']['B1'].shape)


print('Report :\n')
print(classification_report(y_test,y_pred))
print('Accuracy Score: ', accuracy_score(y_pred,y_test))
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))


#%% Visualize Image --------------------------------------------

# Default ------
plt.figure(figsize=(8,8))
plt.imshow(rf_result)
plt.title('Classified S2 Image - Single')
plt.tight_layout()
plt.axis('on')
output_path = r'/Users/hoyeong/Desktop/PBML_Project/Image/S2_timeseries_def2.jpeg'
plt.savefig(output_path, dpi=500)
plt.ion()
plt.show()


# Cmap --------
from matplotlib.patches import Patch
cmap_dict = {
    100: '#EA3323',
    200: '#EDE94E',
    300: '#324A30',
    400: '#559739',
    500: '#72297A',
    600: '#79CBC9',
    700: '#0502F0'
}
class_dict = {
    100: 'Used Area',
    200: 'Agricultural Land',
    300: 'Forest Areas',
    400: 'Grassland',
    500: 'Wetland',
    600: 'Barren',
    700: 'Water'
}
cmap = ListedColormap([cmap_dict[x] for x in sorted(cmap_dict.keys())])
plt.imshow(rf_result, cmap=cmap)
plt.title('Classified S2 Image - Single')
legend_labels = [Patch(facecolor=cmap_dict[key], edgecolor=cmap_dict[key], label=class_dict[key]) for key in cmap_dict.keys()]
plt.legend(handles=legend_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
plt.tight_layout()
plt.axis('on')
output_path = r'/Users/hoyeong/Desktop/PBML_Project/Image/S2_single_cmap2.jpeg'
plt.savefig(output_path, dpi=500)
plt.ion()
plt.show()

# Confusion Mat -----
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title(f'Confusion Matrix')
plt.tight_layout()
out_path = r'/Users/hoyeong/Desktop/PBML_Project/Image/S2_timeseries_confmat.jpeg'
plt.savefig(out_path, dpi=600)
plt.ion()
plt.show()


#%% Predict other time ----------------------------------------

pred2 = clf.predict(X)
pred2_result = pred2.reshape(s2_list[0]['band_data']['B1'].shape)
print('pred Accuracy: ', accuracy_score(y,pred2))

plt.figure(figsize=(8,8))
plt.imshow(pred2_result)
# plt.title('Classified S1 Image - Time Series')
plt.tight_layout()
plt.axis('on')
# output_path = r'/Users/hoyeong/Desktop/PBML_Project/Image/S1_timeseries_def2.jpeg'
# plt.savefig(output_path, dpi=500)
plt.ion()
plt.show()

plt.figure(figsize=(8,8))
plt.imshow(pred2_result)
# plt.title('Classified S1 Image - Time Series')
plt.tight_layout()
plt.axis('on')
# output_path = r'/Users/hoyeong/Desktop/PBML_Project/Image/S1_timeseries_def2.jpeg'
# plt.savefig(output_path, dpi=500)
plt.ion()
plt.show()