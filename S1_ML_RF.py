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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#%% Read Sentinel-1 Image --------------------------------------
fp = glob.glob(r'/Users/hoyeong/Documents/PYTHON/PBML/Image/TIFF/TF/*.tif')
fp.sort()

s1_list = []
for img in fp:
    with rasterio.open(img) as src:
            raster_data = {
                # read(2) : VV | read(1): VH 
                'band_data': {'vv_band': src.read(2),
                              'vh_band': src.read(1),
                              'vv / vh': src.read(2) - src.read(1)},
                'meta': src.meta,
                'bounds': src.bounds,
                'name': src.name.split('/')[-1].split('.')[0]
            }
            s1_list.append(raster_data)

#%% Subset Sentinel-1 Image ------------------------------------

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
    im_list_clip=[]
    for file in fp:
        with rasterio.open(file) as src:
            p = Proj(src.crs)
            xmin, ymin = p(lon[0], lat[0])
            xmax, ymax = p(lon[1], lat[1])
            window = longlat2window(lon,lat,src)
            new_transform = src.window_transform(window)
            vv_band = np.array(src.read(2, window=window))
            vh_band = np.array(src.read(1, window=window))
            ratio_band = vv_band - vh_band
            raster = {'band_data': {'vv_band': vv_band,
                                    'vh_band': vh_band,
                                    'vv / vh': ratio_band},
                      'meta': {'driver': src.meta['driver'],
                               'dtype': src.meta['dtype'],
                               'nodata': {'vv_band': np.sum(np.isnan(vv_band)),
                                          'vh_band': np.sum(np.isnan(vh_band)),
                                          'vv / vh': np.sum(np.isnan(ratio_band))},
                               'width': np.array(src.read(1,window=window)).shape[1],
                               'height': np.array(src.read(1,window=window)).shape[0],
                               'count': 3,
                               'crs': src.meta['crs'],
                               'transform': new_transform},
                      'bounds': [xmin,ymin,xmax,ymax],
                      'name': src.name.split('/')[-1].split('.')[0]}
        im_list_clip.append(raster)
    return im_list_clip

def temporal_average(im_list):
    band_names = ['vv_band', 'vh_band', 'vv / vh']
    averaged = {'band_data':{}}
    for band in band_names:
        stacked_bands = np.stack([img['band_data'][band] for img in im_list])
        # Calculate the mean along the new axis (axis=0) created by stacking
        averaged['band_data'][band] = np.mean(stacked_bands, axis=0)
    averaged['meta'] = {'driver': 'GTiff',
                         'dtype': 'float32',
                         'width': averaged['band_data']['vv_band'].shape[1],
                         'height':averaged['band_data']['vv_band'].shape[0],
                         'crs': im_list[0]['meta']['crs'],
                         'transform': im_list[0]['meta']['transform']}
    averaged['bounds'] = im_list[0]['bounds']
    averaged['name'] = 'Temporal Average'
    return averaged

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

def sarrgb(vv_arr,vh_arr):
    '''
    ---------
    vv_arr: 'VV' band numpy array
    vh_arr: 'VH' band numpy array
    Returns
    ---------
    RGB Image using (vv,vv,vh)
    '''
    def equalize_hist(channel):
        hist, bin_edges = np.histogram(channel, bins=256, range=(0, 1))
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]  # normalize
        equalized_channel = np.interp(channel, bin_edges[:-1], cdf)
        return equalized_channel
    
    # 10*nplog10
    R = vv_arr
    G = vv_arr
    B = vh_arr
    # Normalize
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    G = (G - np.min(G)) / (np.max(G) - np.min(G))
    B = (B - np.min(B)) / (np.max(B) - np.min(B))
    # Histogram Equalization
    R = equalize_hist(R)
    G = equalize_hist(G)
    B = equalize_hist(B)
    # Stack
    RGB = np.stack((R, G, B), axis=-1)
    return RGB

filtered_RGB = sarrgb(s1_avg['band_data']['vv_band'], s1_avg['band_data']['vh_band'])
fig, axes = plt.subplots()
axes.imshow(filtered_RGB)
axes.set_title('Filtered_RGB')
plt.axis('off')
plt.tight_layout()
# output_path = '/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/Image/ATSF_RGB.png'
# plt.savefig(output_path, dpi=300)
plt.ion()
plt.show()

Original_RGB = sarrgb(s1_list[0]['band_data']['vv_band'], s1_list[0]['band_data']['vh_band'])
fig, axes = plt.subplots()
axes.imshow(Original_RGB)
axes.set_title('Original_RGB')
plt.axis('off')
plt.tight_layout()
# output_path = '/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/Image/Original_RGB.png'
# plt.savefig(output_path, dpi=300)
plt.show()


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

del s1_list
s1_list = subset_raster(fp, lon=(min_lon, max_lon),
                            lat=(min_lat, max_lat))

# Check Shape of s1_list
for i in range(len(s1_list)):
    print(s1_list[i]['band_data']['vh_band'].shape)

s1_list = match(s1_list)

#%% Temporal Average -------------------------------------------

s1_avg = temporal_average(s1_list)

#%% Visualize Data ---------------------------------------------

def sar_disp(s1_list, band='vv', index=[0]):
      '''
      Parameters:
      -----------------------
      im_list: image list
      band: 'vv', 'vh', 'vv / vh'
      index: list of index you want to display
      
      Returns:
      -----------------------
      SAR Image Display
      '''
      for i in index:    
        if band=='vv': 
            disp_band = np.clip(s1_list[i]['band_data']['vv_band'], -25, 0)
            disp_band = ((disp_band + 25) * (255 / 25)).astype(np.uint8)
        elif band=='vh':
            disp_band = np.clip(s1_list[i]['band_data']['vh_band'], -25, 0)
            disp_band = ((disp_band + 25) * (255 / 25)).astype(np.uint8)
        else:
            disp_band = np.clip(s1_list[i]['band_data']['vv / vh'], -25, 0)
            disp_band = ((disp_band + 25) * (255 / 25)).astype(np.uint8)
            
        fig, axes = plt.subplots()
        axes.imshow(disp_band, cmap='gray')
        axes.set_title('S1_%s_Image_%d' %(band,i+1))
        axes.axis('on')
        plt.tight_layout()
        plt.ion()
        plt.show()

sar_disp(s1_list, band='vv', index=[4])
sar_disp([s1_avg], band='vv')

#%% Pandas DataFrame ------------------------------------------

# # Create pandas DataFrame from VV,VH arrays
# df = pd.DataFrame({
#     'vv_band': s1_avg['band_data']['vv_band'].flatten(),
#     'vh_band': s1_avg['band_data']['vh_band'].flatten(),
#     'vv / vh': s1_avg['band_data']['vv / vh'].flatten()})

df = pd.DataFrame()
for i, s1_img in enumerate([s1_list[0]]):
    df[f'vv_band_{i}'] = s1_img['band_data']['vv_band'].flatten()
    df[f'vh_band_{i}'] = s1_img['band_data']['vh_band'].flatten()
    df[f'vv_vh_ratio_{i}'] = s1_img['band_data']['vv / vh'].flatten()

print(df.head())

#%% Rasterize Shape File --------------------------------------

# Load Shapefile
ref_shp = gpd.read_file(r'/Users/hoyeong/Documents/PYTHON/PBML/Image/Label/Label.shp')

# Convert Shapefile to Raster
# ref_shp['L2_CODE'] = pd.to_numeric(ref_shp['L2_CODE'])

shapes = ((geom,value) for geom, value in zip(ref_shp.geometry, ref_shp.L1_CODE))
rasterized_shp = rasterize(shapes,
                           out_shape=s1_list[0]['band_data']['vv_band'].shape,
                           transform=s1_list[0]['meta']['transform'],
                           fill=0)
# Count NaN Values
np.count_nonzero(rasterized_shp == 0)

# Extract labels for your training pixels
train_labels = rasterized_shp.flatten()

# Create DataFrame 
df['label'] = train_labels

#%% Data Preprocessing ----------------------------------------
# df = df.dropna()
# df = df[df.label != 0]

#%% Train test split ------------------------------------------

# # Single Image 
# X = df[['vv_band', 'vh_band', 'vv / vh']]
# y = df['label']

# # Multitemporal Image
feature_columns = [f"{name}_{i}" for i in range(1) for name in ["vv_band", "vh_band", "vv_vh_ratio"]]
X = df[feature_columns]
y = df['label']

# Straify (Random Sampling from each Labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)

#%% Hyper-Parameter Tuning ------------------------------------
from sklearn.model_selection import RandomizedSearchCV

params = {'n_estimators' : [100,200,300],
          'max_depth' : [10,20,30,None],
          'min_samples_leaf' : [1,2,4],
          'min_samples_split' : [2,5,10]}

rf_clf = RandomForestClassifier(random_state = 100, n_jobs = 4)
rand_cv = RandomizedSearchCV(rf_clf, param_distributions = params, cv = 5, verbose=3, n_jobs = 4)
rand_cv.fit(X_train, y_train)

print('Best Hyper-Parameters: ', rand_cv.best_params_)
print('Best Accuracy Score: {:.4f}'.format(rand_cv.best_score_))

#%% Training and Prediction ----------------------------------

clf = rand_cv.best_estimator_

# If you know parameters
clf = RandomForestClassifier(n_estimators=100, min_samples_split=10, 
                             min_samples_leaf= 1, max_depth= 10, n_jobs=-1)
clf.fit(X_train, y_train)

# Predict labels
y_pred = clf.predict(X_test)
print('y_pred Model Accuracy: ', accuracy_score(y_test,y_pred))

# Predict Entire Image
pred = clf.predict(X)
print('pred Model Accuracy: ', accuracy_score(y,pred))
rf_result = pred.reshape(s1_list[0]['band_data']['vv_band'].shape)

print('Report :\n')
print(classification_report(y_test,y_pred))
print('Accuracy Score: ', accuracy_score(y_pred,y_test))
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))

#%% Visualize Result -----------------------------------------

# Default ------
plt.figure(figsize=(8,8))
plt.imshow(rf_result)
plt.title('Classified S1 Image - Time Series')
plt.tight_layout()
plt.axis('off')
output_path = r'/Users/hoyeong/Desktop/PBML_Project/Image/S1_timeseries_def1.jpeg'
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
norm = BoundaryNorm(sorted(cmap_dict.keys()), len(cmap_dict.keys()))
plt.imshow(rf_result, cmap=cmap, norm=norm)
plt.title('Classified S1 Image - Single')
legend_labels = [Patch(facecolor=cmap_dict[key], edgecolor=cmap_dict[key], label=class_dict[key]) for key in cmap_dict.keys()]
plt.legend(handles=legend_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
plt.tight_layout()
plt.axis('on')
output_path = r'/Users/hoyeong/Desktop/PBML_Project/Image/S1_single_cmap2.jpeg'
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
out_path = r'/Users/hoyeong/Desktop/PBML_Project/Image/S1_timeseries_confmat.jpeg'
plt.savefig(out_path, dpi=600)
plt.ion()
plt.show()

#%% Predict other time ----------------------------------------

pred2 = clf.predict(X)
pred2_result = pred2.reshape(s1_list[0]['band_data']['vv_band'].shape)
print('pred Model Accuracy: ', accuracy_score(y,pred2))
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
plt.imshow(rasterized_shp)
# plt.title('Classified S1 Image - Time Series')
plt.tight_layout()
plt.axis('on')
# output_path = r'/Users/hoyeong/Desktop/PBML_Project/Image/S1_timeseries_def2.jpeg'
# plt.savefig(output_path, dpi=500)
plt.ion()
plt.show()