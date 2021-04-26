# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import cdsapi

import numpy as np
import pandas as pd
import pandas_profiling
import geopandas
import netCDF4
import xarray as xarr # pandas based library for 
            # labeled data with N-D tensors at each dimension
import salem

import matplotlib.pyplot as plt
# %matplotlib inline 
import cartopy
import cartopy.crs as ccrs
import seaborn as sns

# %%
# # This cell needs to be run only once, when the agroclimatic indicators are downloaded from the source

# c = cdsapi.Client()

# c.retrieve(
#     'sis-agroclimatic-indicators',
#     {
#         'origin': 'era_interim_reanalysis',
#         'variable': [
#             'biologically_effective_degree_days', 'frost_days', 'heavy_precipitation_days',
#             'ice_days', 'maximum_of_daily_maximum_temperature', 'maximum_of_daily_minimum_temperature',
#             'mean_of_daily_maximum_temperature', 'mean_of_daily_mean_temperature', 'mean_of_daily_minimum_temperature',
#             'mean_of_diurnal_temperature_range', 'minimum_of_daily_maximum_temperature', 'minimum_of_daily_minimum_temperature',
#             'precipitation_sum', 'simple_daily_intensity_index', 'summer_days',
#             'tropical_nights', 'very_heavy_precipitation_days', 'wet_days',
#         ],
#         'experiment': 'historical',
#         'temporal_aggregation': '10_day',
#         'period': '198101_201012',
#         'format': 'zip',
#     },
#     'agroclimindicators.zip')

# %%
# Read the data Path where it is stored on the Computer
#data_dir = input('Path to the data\n')
data_dir = os.path.join('C:\\'
                        'Users',
                        'kurt_',
                        'Data',
                        'agroclimate','')

# %% [markdown]
# ### Importing the Indicators
# <div class="span5 alert alert-info">
# Now importing the agroclimatic indicators from the disk where they are stored as a single netcdf file per an indicator. All the files will be merged into a single xarray dataset which will be divided into chunks and parallelization with dask will be enabled to speed up the operations
# </div>

# %% tags=[]
# Import data as xarray dataset from the directory
dask = True
if dask:
    # Import with dask
    agroclim = xarr.open_mfdataset(data_dir+'*.nc', parallel=True, 
                              combine='by_coords', chunks={'time': 20}
                             , engine='netcdf4')
    print(f'The chunk size for time dimension is {agroclim.chunks["time"][0]}\n')
    print(f'dataset, thus, have {len(agroclim.time)/agroclim.chunks["time"][0]} chunks')
else:
    # Import without dask for debugging
    agroclim = xarr.open_mfdataset(data_dir+'*.nc', parallel=False, 
                          combine='by_coords', engine='netcdf4')

# %% [markdown]
# ### Exploring the indicators dataset

# %%
#print(clim.data_vars)
#print(clim.coords)
agroclim

# %% [markdown]
# agroclim dataset have 2 spatial, one time coordinates and 15 variables. Now let's see what each of these variables are.

# %% tags=[]
for var in agroclim:
    print(f'{var}: {agroclim[var].attrs}')
# Let's select the first time step and plot the 2m-air temperature

# Let's check the dimensions
for dim in agroclim.dims:
    dimsize = agroclim.dims[dim]
    print(f'\nData has {dimsize} {dim} ')
    if dim == 'latitude':
        print(f' latitudes: from {float(agroclim[dim].min())} degree South',
     f'to {float(agroclim[dim].max())} degree North')
    if dim == 'longitude':
        print(f' Longitudes: from {float(agroclim[dim].max())} degree East',
     f'to {float(agroclim[dim].min())} degree West')
    if dim == 'time':
        print(f'time: from {pd.to_datetime(agroclim["time"].min().values)} to {pd.to_datetime(agroclim["time"].max().values)} ')

# %% [markdown]
# # Preparing The Crop Production Data
# Crop Data is obtained from:
#
#
# ### Alfalfa Hay
# Alfalfa hay is produced mostly in North-Western States. Among them it is produced throughout all Montana and in most of the Idaho which makes them more convenient for agroclimatic analysis.
#
# Here is the map that shows where Alfala hay is produced
# Source: https://www.nass.usda.gov/Charts_and_Maps/Crops_County/al-ha.php

# %% [markdown]
# * Masking climate data only to keep the relavant states using **Salem**
# * Geospatial data for the state boundaries are from US Census
# * Let's examine the shape file for US States using __Geopandas__
#

# %%

from matplotlib.gridspec import GridSpec
import matplotlib as mpl

mpl.rc('figure', figsize = (18,12))
fig = plt.figure()
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])

a = plt.imread("https://www.nass.usda.gov/Charts_and_Maps/graphics/AL-HA-RGBChor.png")
ax1.imshow(a)
c = plt.imread("https://www.nass.usda.gov/Charts_and_Maps/graphics/CR-PL-RGBChor.png")
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(c)
s = plt.imread("https://www.nass.usda.gov/Charts_and_Maps/graphics/SB-PL-RGBChor.png")
ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(s)
for ax in [ax1,ax2,ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
plt.tight_layout()
plt.show()
plt.close()

mpl.rcParams.update(mpl.rcParamsDefault)

# %% [markdown]
# ## Embedding Geospatial Coordinates For Statewise operations

# %% tags=[]
# Let's read the geospatial data for the states
path_geo = 'C:\\Users\\kurt_\\Data\\usstates\\'
geo_usa = geopandas.read_file(path_geo)
print(type(geo_usa))
print('The coordinate Reference System Info:')
print(geo_usa.crs)
geo_usa.head()

# %% tags=[]
# Let's see the state boundaries on a map to see
# if there is an error

# Getting rid of oversees territories from the map
geo_usa = geo_usa[geo_usa.STATEFP.apply(lambda x: int(x)) < 60]
#Let's remove the Alaska too
geo_usa = geo_usa[geo_usa.NAME != 'Alaska']
fig,ax = plt.subplots(figsize=(16, 12))
geo_usa.plot(ax=ax, cmap='OrRd')
ax.set_xlim(-127,-65)
ax.set_ylim(22,55)
ax.set_yticks([])
ax.set_xticks([])
ax.axis("off")
geo_usa.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=7),axis=1);
#plt.tight_layout()
plt.show()
plt.close()

# %% [markdown]
# <div class="span5 alert alert-info">
# We will now embedd the geogrophical coordinates data of the states to our climate data. Then we will plot 2-d temperature variable of two states on a random day to make sure that everything is fine.
# </div>

# %%
# Plotting a random time step just to see the data on a map
alfala_states = ['Montana', 'Idaho']
fig = plt.figure(figsize=(10, 8))
# plotting on a map using cartopy
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.add_feature(cartopy.feature.RIVERS)
ax.add_feature(cartopy.feature.STATES)

# plotting using xarray plot method
# Montana and Idaho For Alfala Barley
MT_coord = salem.read_shapefile(path_geo+'cb_2018_us_state_500k.shp')
MT_coord = MT_coord[(MT_coord.NAME.isin(alfala_states))]
mnt_sub = agroclim.salem.subset(shape=MT_coord, margin=10)
MT_coord.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=7),axis=1);
# Let's plot the daily average tempreture on a random time
randm_day = mnt_sub['TG'].isel( time=np.random.randint(len(mnt_sub.time)))
randm_day.salem.roi(shape=MT_coord).plot(ax=ax)
#Montana_anm = clim_loc['TG'].isel( time=np.random.randint(len(clim_loc.time))) - clim_loc['TG'].mean(dim='time')
#Montana_anm.plot(ax=ax)
plt.show()
plt.close()

# %% [markdown]
# ## Preparing Crop Production Data
# Data link: https://quickstats.nass.usda.gov/results/347988B6-8746-305D-9147-D1A31FE09FD2

# %% [markdown]
# <div class="span5 alert alert-info">
# It's time get our crop production data which contain various fields. The most important fields for this work are the State, Commodity, and the yield (Value). However, we do not need all the eentries as some of them are irrelevant for our purpose or using them make it too complex for the scope of this work. Therefore, we will use most of the columns, even if we wouldn't need them eventually, to filter out those entries we want to get rid of.
# </div>

# %% tags=[]
#  Now Reading All the Crop Data that we are interested in to a dataframe
path = "Data/"
# Let's read and merge all the crops data into a single dataframe
# This way data wrangling steps will be less cumbersome
files = [path+"Alfala.csv", path+"Corn.csv", path+"Soybean.csv"]
df = (pd.read_csv(f) for f in files)
df_crop = pd.concat(df, ignore_index=True)
df_crop.head(3).T

# %%
df_crop.info()

# %% [markdown]
# We are not interested in most of these columns. The only relevant columns are these: `State`, `State ANSI` (may be helpful for a regression model), `Commodity`, `Data Item`(that is the column associated with the Value column along with State colums), `Domain`, `Value` (This is what we are trying to predict). However, we might need to check them first to decide which rows to include for our model. 

# %% [markdown]
# * **Value column is object type. We need to convert it to Float.**
# * Also Let's drop all other nan values. None of the columns we are interested in has any nan values.
#

# %% tags=[]
f = lambda x: x.replace(',', '')
df_crop['Value'] = df_crop['Value'].apply(f)
# Before converting to numeric let's see if there is non-numeric values and what they are
def IsNumeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
S = lambda s: isinstance(s, (int, float))
_isdigit = df_crop['Value'].apply(IsNumeric)
_r = df_crop[~_isdigit]['Value'].value_counts()
print(f'There are {_r[0]} entries with the value of "{_r.index[0]}"' )

# %% [markdown]
# **According to the [Quick Stats Glossary](https://quickstats.nass.usda.gov/src/glossary.pdf) published by USDA for this dataset "D" corresponds to:** 
#
# > **Withheld to avoid disclosing data for individual operations.**
#
# Thus all we can do is dropping these rows.
#

# %%
df_crop['Value'] = pd.to_numeric(df_crop['Value'], errors='coerce')
df_crop['Value'].isnull().sum()

# %% [markdown]
# Now Let's drop all nan values together. Let's first check the with null values in the dataframe

# %%
# Missing Values
nan=pd.DataFrame(df_crop.isnull().sum().sort_values(ascending=False), columns = ['NULL values'])
nan

# %% tags=[]
#first get rid of the nan values in Value column only
df_crop.dropna(subset=['Value'], inplace=True)
# Now we can drop all the columns with nan values at once
df_crop.dropna(axis=1, inplace=True)
# Fixing the index
df_crop.reset_index(drop=True, inplace=True)
print(df_crop.info())
df_crop.head(3)


# %% [markdown]
# **There are some constant values and some entries we are not interested in. Let's remove them from the df_crop**

# %%
# First let's take a look at the number of unique 
# values in each column in a bar plot
plt.style.use('ggplot')
df_crop.nunique().plot(kind='bar', logy=True)
plt.xlabel('Features')
plt.ylabel('Count')
plt.show()
plt.close()
# Find those columns with constant values

# %% [markdown]
# **We need only Year entries for Period. Let's see what else is in there**

# %%
df_crop['Period'].value_counts()

# %% [markdown]
# Looks like some values are just forecast entries not observations. We don't need any of those.

# %%
# dropping all rows with forecast entries given in Period column
mask_year = df_crop['Period'] == 'YEAR'
df_crop = df_crop[mask_year]
df_crop.reset_index(drop=True, inplace=True)
df_crop.nunique()

# %% [markdown]
# **Let's take a look at the Domain Catogories.**
# * For this study some of the domains are irrelavant, like "ECONOMIC CLASS"
# * Domain Catogories should be consistent among all "Data Item" values

# %% tags=[]
print(df_crop['Domain Category'].nunique())
df_crop['Domain Category'].value_counts().head(10)

# %% [markdown]
# We have two sources of records which are given in the `Program` field. Let's check these resources.

# %%
df_crop['Program'].value_counts()

# %%
mask_DomCat = df_crop['Domain Category'] != 'NOT SPECIFIED'
print(df_crop[mask_DomCat].Commodity.value_counts())
print(df_crop[mask_DomCat].Program.value_counts())
df_crop[mask_DomCat]['Domain Category'].value_counts()

# %% [markdown]
# There are 69 Domain Categories, but only Soybeans from CENSUS have Categories other than `NOT SPECIFIED`.
# **Hence, for now I seperate all those Domain Categories to deal with them later since they require special handling.**
#
# Note that these entries are from CENCUS records. Thus it is already more reasonable not to mix data from two different source **("Survey and Cencus")**

# %%
# creating new dataframe for soybeans from CENSUS
df_soybean_census = df_crop[mask_DomCat]
df_soybean_census.head()

# %%
# Crerating a new dataframe for Survey only crops
df_crop_srv = df_crop[~mask_DomCat]
df_crop_srv.head()

# %%
# Let's what commodities left in the df_crop_srv
df_crop_srv.Commodity.value_counts()

# %%
df_crop_srv.head(3)

# %% [markdown]
# We are only interested some of these columns. Let's drop the ones we will not use.
#
# **We do not need constant value columns**

# %% tags=[]
# Getting rid of the columns with constant values
nunq = df_crop_srv.nunique()
dropped = []
for clm in df_crop_srv:
    if nunq.loc[clm] == 1:
        df_crop_srv = df_crop_srv.drop(clm, axis=1)
        dropped.append(clm)
print(df_crop_srv.nunique(),'\n\n')
print(f'dropped columns: {dropped}')

# %%
# Now fixing the index
df_crop_srv = df_crop_srv.reset_index(drop=True)
df_crop_srv.info()

# %%
df_crop_srv.sample(3)

# %% tags=[]
# # Let's get rid of the rows where Period is smth other than Year
# period_to_rid = set(hay['Period']).difference(['YEAR'])
# print(f'The following period entries will be removed:\n {period_to_rid}')
# Period_rows = hay['Period'].isin(period_to_rid)
# hay = hay[~Period_rows]
# hay['Period'].value_counts()

# %% [markdown]
# ## Pre-Processing & Exploring
# Now that we have crop data ready for analysis, we should now prepare the climate data. 
# * Climate variables should be averaged for each state
# * Then we need to aggregate all climate features thorugh each year.
#     * Some features require averging, such as mean temperature, some require to be summed such as 'Biologically Effective Degree Days'
#     * Note that each feature in climate data can have completely different effect depending on the season or month. For example while frost days can have a devastating effect on April it might have no effect at all on January. Therefore for some climate variables I should treat each month as a separate feature. 
#     

# %% [markdown]
# ## Getting State Climate Data
#

# %%
df_crop_srv['Data Item'].value_counts()


# %%
def crop_trend_plot(crop, method=False):
    """method is given when only certain quantities are needed.
    It should be a keyword or keywords which can be used to filter only
    intended quantity(or quantities). e.g. method='YIELD' will 
    only plot Yield in any Unit available.
    Multiple keywords should be separated with space  """
    # creating a list of 
    meas = ["PRODUCTION, MEASURED IN TONS", "PRODUCTION, MEASURED IN BU",
            "YIELD, MEASURED IN TONS / ACRE", "YIELD, MEASURED IN BU / ACRE",
            "ACRES HARVESTED"]
    if method:
        ls = []
        for item in meas:
            if method in item:
                ls.append(item)
        meas = ls
    # SOYBEANS from census were removed, and SURVEY records do not have all the Data Item values as the others
    if crop != 'SOYBEANS':
        crop_type = [" - ",", IRRIGATED - ", ", NON-IRRIGATED - ", ", ORGANIC - "]
    elif crop == 'SOYBEANS':
        crop_type = [" - "]
    
    dt_type = [i+j for i in crop_type for j in meas]
    plotting = [crop+x for x in dt_type]
    for dt in plotting:
        alf_prd = df_crop[df_crop['Data Item'] == dt]
        # plot only if alf_prd has at least one entry
        if len(alf_prd):
            sns.lineplot(data = alf_prd, x='Year', y='Value', hue='State')
            plt.title(dt, fontsize=10)
            plt.show()
            plt.close()
        # elif len(alf_prd) == 0:
        #     print(f"{dt} NOT Exist")


# %% [markdown]
# ## Alfalfa Hay

# %% tags=[]
crop_trend_plot("HAY, ALFALFA", method='YIELD')

# %% [markdown]
# ## Corn Grain

# %% tags=[]
crop_trend_plot("CORN, GRAIN", 'YIELD')

# %% [markdown]
# * Corn grain irrigated has less variance comparing to non-irrigated. This might be because irrigation helps the crops much better tolerate any warm and dry growing season

# %% [markdown]
# ## Soybeans

# %%
crop_trend_plot("SOYBEANS", 'YIELD')

# %% [markdown]
# ## Dealing with Trends for Alfala
# * There is a clear trend in Alfala production and yield except for Non-Irrigated Alfala
# * Significant trend in Alfala is perhaps related to development in agricultral technologies
# * Warming climate may also have a positive impact. Irrigated crops might benefit from warmer temperature due to increased photosynthesis. (** REF. **)
# * Non-Irrigated crops may or may not tolarate warmer climate depending on soil water availability and some other factors. This would explain the difference between Irrigated and Non-irrigated Alfala yield trends. 
# * **The priority of this project is to investigate how crop yield is affected by weather conditions. Therefore interannual variability in yield is more important for this work than decadal variability where the latter is related not only Climate but also development in agricultural technologies and human beheviours.**
#
# * Therofore, I will first **detrend the predictor (climate) variables and response (yield).**  

# %%
a= "SOYBEANS, IRRIGATED - ACRES HARVESTED"
ah = "ACRES HARVESTED"
S = lambda s: ah in s
test = df_crop['Data Item'].apply(S)
df_crop[test]

# %% [markdown]
# ## Getting Climate For Product Zone

# %%
month_length = agroclim.time.dt.days_in_month
month_length

# %% [markdown]
# Some variables are aggregated as sum of all the occurence 
# during default time interval (10 days), while some are averaged
# or their extreme value is recorded. The aggregation method is
# recorded in "cell_methods" attribute of the variable. Using cell_methods
# for each variable monthly aggregations will be calculated. 
#
# Then, All the climate parameters will be spatially averaged through the state level.
#
#

# %% tags=[]
# First resampling entire dataset. Later individual variables will be resampled 
# based on their cell_method
climate = agroclim.resample(time="1MS", keep_attrs=True).sum()
# Determining cell_method
for var in climate:
    rsmpl = dict()
    # Some variables, such as time_bounds, does not have cell_methods attribute
    # Don't raise error for those, just skip.
    # collecting all variables' resampled data in a dict or such would make a 
    # large single object. I could use chunks but why not keep them as
    # variable with their unique name
    try:
        # variables need to be summed already aggregated as sum
        if "mean" in agroclim[var].cell_methods:
            vars()[var] = agroclim[var].resample(time="1MS",keep_attrs=True).mean()
        elif "maximum" in agroclim[var].cell_methods:
            vars()[var] = agroclim[var].resample(time="1MS",keep_attrs=True).max()
        elif "minimum" in agroclim[var].cell_methods:
            vars()[var] = agroclim[var].resample(time="1MS",keep_attrs=True).min()
    except:
        pass

# Now let's correct resampling of each variables
for x in agroclim:
    try:
        eval(x)
    except NameError:
        continue
    climate[x] = eval(x)
# free up some memory
#del agroclim

# %%
climate.dims

# %% [markdown]
# ## Calculating the climotological state means 

# %%
plt.style.use('seaborn-colorblind')
crop_states = ['Wisconsin'] #list(df_crop.State.unique())
crop_states = [x.title() for x in crop_states]
fig = plt.figure(figsize=(12,10))
# plotting on a map using cartopy
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.add_feature(cartopy.feature.RIVERS)
ax.add_feature(cartopy.feature.STATES)

# plotting using xarray plot method
# Reading state shape files from 
us_states = salem.read_shapefile(path_geo+'cb_2018_us_state_500k.shp')
state_coord = us_states[(us_states.NAME.isin(crop_states))]
# Extracting only the region of interest from climate data
state_clim = climate.salem.subset(shape=state_coord, margin=5)
state_coord.apply(lambda x: ax.annotate(s=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=7),axis=1);
# Let's plot the daily average tempreture on a random time
randm_day = state_clim['TG'].isel( time=np.random.randint(len(state_clim.time)))
# randm_day.salem.roi(shape=state_coord).plot(ax=ax)
randm_day = randm_day.salem.roi(shape=state_coord)
plt.pcolormesh(randm_day.lon, randm_day.lat, randm_day)
cb = plt.colorbar(shrink=0.4) # use shrink to make colorbar smaller
cb.set_label(f"{agroclim['TG'].long_name} in {agroclim['TG'].units}")
plt.title(randm_day.time.values)
plt.show()
plt.close()


# %%
for var in agroclim:
    print(f'{var}: {agroclim[var].attrs}')
# Let's select the first time step and plot the 2m-air temperature

# Let's check the dimensions
for dim in agroclim.dims:
    dimsize = agroclim.dims[dim]
    print(f'\nData has {dimsize} {dim} ')
    if dim == 'latitude':
        print(f' latitudes: from {float(agroclim[dim].min())} degree South',
     f'to {float(agroclim[dim].max())} degree North')
    if dim == 'longitude':
        print(f' Longitudes: from {float(agroclim[dim].max())} degree East',
     f'to {float(agroclim[dim].min())} degree West')
    if dim == 'time':
        print(f'time: from {pd.to_datetime(agroclim["time"].min().values)} to {pd.to_datetime(agroclim["time"].max().values)} ')

# %%
for state in list(df_crop.State.unique()): #['Wisconsin']:#
    state_coord = us_states[(us_states.NAME == state.title())]
    # Let's extract the state of interest and save as a separate dataset
    state_clim = climate.salem.subset(shape=state_coord)
    # Now let's take a spatial mean for entire state
    state_clim = state_clim.salem.roi(shape=
                state_coord).mean(dim=['lat','lon'])
    # Detrending all the features so that we can study the interannual affect of 
    # weather to crop yields.
    plot_month_group('TNx',12,1,2,3,4,5,6,7,8,9,10,11)


# %% [markdown]
# Let's write a function for plotting only a certain month in each year. We will use each month group as a single feature

# %%
plt.style.use('default')
def plot_month_group(var, *months_to_plot ):
    #plt.style.use('seaborn-paper')
    # plt.style.use('tableau-colorblind10')
    plt.style.use([ 'tableau-colorblind10'])
    # 
    """ var is the variable to plot in the dataset. It should be given as a string
    months_to_plot should be an integer corresponding to a month. 
    Multiple months should be separated with commas. 
    e.g. plot_month_group('TXn',12,1,2) 

    """
    
    for mon in state_clim[var].groupby("time.month"):
        if mon[1]['time.month'][0] in months_to_plot:
            mon[1].plot(label=f'Month: {mon[1]["time.month"].to_series()[0]}')
    plt.legend()
    plt.show()
    plt.close()


# %%
plot_month_group('TXn',12,1,2,3,4,5,6,7,8,9,10,11)

# %%
for state in list(df_crop.State.unique()): #['Wisconsin']:#
    state_coord = us_states[(us_states.NAME == state.title())]
    # Let's extract the state of interest and save as a separate dataset
    state_clim = climate.salem.subset(shape=state_coord)
    # Now let's take a spatial mean for entire state
    state_clim = state_clim.salem.roi(shape=
                state_coord).mean(dim=['lat','lon'])
    # Detrending all the features so that we can study the interannual affect of 
    # weather to crop yields.
    plot_month_group('TNx',12,1,2,3,4,5,6,7,8,9,10,11)

# %% tags=[]
df_crop_srv.profile_report(explorative=True, html={'style': {'full_width': True}})


# %%
# #Turkey
# #clim_loc = clim.where((clim.lat > 30) & (clim.lat < 50) & (clim.lon >20 ) & (clim.lon < 45), drop=True)
# #shdf = salem.read_shapefile(salem.get_demo_file('world_borders.shp'))
# #shdf = shdf.loc[shdf['CNTRY_NAME'] == 'Turkey']

# feature_cols = [ 'Rainfall' ,'Temperature','Usage amount']
# target_v = df['water level']
# X = df[feature_cols] 
# y = target_v 

# from sklearn.model_selection import TimeSeriesSplit
# tss = TimeSeriesSplit(n_splits = 3)
# for train_index, test_index in tss.split(X):
#     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# mean = np.mean((df.values), axis=-1, keepdims=True)
# detrended = df - mean

# %% [markdown]
# # Modelling
# <div class="span5 alert alert-info">
# 1. There are two main ways to predict the crop yield. The first one is from simple time series analysis of crop yield data, and building a time series model such as ARIMA. This method is straightforward, does not require any variable other than the yield itself and time as the single dimension. However, this method does not provide any physical inside for the problem and assumes that all the conditions relavent to crop production will be the same in the future. Despite the weakneses it can still provide a good starting point and can be useful to see how will the yield change in the future all the conditions stay the same as the past.
#
# 2. Second method would be building regression models to predict the crop yield from the actual physical parameters. This method is superior to time series analysis in terms of providing more actionable results, such as if we determine that the most important parameter is the rain amount during the growing season we could suggest irrigation to increase. 
#
# We will use both of these methods and compare the results at the end
#     </div>

# %% [markdown]
# ## 1. ARIMA Model

# %%
# Impoting the libraries required for this section
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

from sklearn.metrics import mean_squared_error

# %% [markdown]
# <div class="span5 alert alert-warning">We will use corn, grain yield data to build an ARIMA model. For this purpose, we do not need the agro-climate indicators.
# Let's remember how the corn data looks.
#     </div>

# %% tags=[]
crop_trend_plot("CORN, GRAIN", 'YIELD')

# %% [markdown]
# Let's start with only 'ILLINOIS'.

# %%
# getting the only corn - yield rows
corn_mask = df_crop_srv['Data Item'] == \
    "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"
df_corn = df_crop_srv[corn_mask].reset_index(drop=True)
# creating dataframe for soybens in Illinois 
df_corn_il = df_corn[df_corn.State == 'ILLINOIS'].sort_values(by='Year').reset_index(drop=True)
df_corn_il.info()

# %%
# Converting Year column to datetime object and setting as index
df_corn_il['Year'] = pd.to_datetime(df_corn_il['Year'], format='%Y')
df_corn_il.set_index('Year', inplace=True)

# %%
# Plotting corn yield for Illinois alone
# _ = df_corn_il['Value'].pct_change().plot(title="Corn Yield - ILLINOIS")
corn_yield = df_corn_il['Value']
_ = yield_soy.plot(title="Corn Yield - ILLINOIS")
# _ = df_corn_il.plot(x='Year', y='Value')

# %% [markdown]
# ### Model Identification
# Before we fit a model, we need to check if the data is stationary. There is abviously a strong trend starting from 1940s, but is it also a random walk? What does it take to make the model stationary?
# * Test the null hypothesis that the model is random walk with Dicky-Fuller Test.
# * Test the hypothesis that model is stationary with KPSS tests.
# * Make the model stationary taking difference of the values.
# * Plot the auto correlation function, and partial auto correlation function of the data to identify possible model order.
# * Build multiple ARIMA models and with different orders and find the best one in terms of AIC and BIC scores

# %%
adf = adfuller(corn_yield)[1]
kpss_ = kpss(corn_yield, nlags="auto")[1]
print(f"Dickey-Fuller test p-value is {round(adf,3)}")
print(f"KPSS test p-value is {round(kpss_, 3)}")

# %% [markdown]
# * Based on the Dickey-Fuller test we can not reject the null hypothesis which is the series has a unit root.
# * The null hypothesis of the KPSS test is the opposite, which is the process is trend stationary. Since the p-value is smaller than 0.05 we can reject the null hypothesis in favor of the alternative. 
#
# Thus both of the test suggest that the series is non-stationary. The easist way to get rid of the trend and make the data stationary is to take the lagged difference of the values. In most of the cases this will take care of non-stationarity. Let's try a differencing for a few lag.

# %% [markdown]
# The auto correlation plot tails off while the partial autocorrelation cuts off at lag 2. This suggest AR(2) model. However, we should check the acf and pacf plots after removing the trend in the next step. 

# %%
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# x= np.linspace(1, len(yield_soy), num=len(yield_soy) ).reshape(-1,1)
# model.fit(x, yield_soy)
# trend = model.predict(x)
# _ = plt.plot(yield_soy.values,label='Yield')
# _ = plt.plot(trend, label='trend')
# _ = plt.legend()
# _ = plt.title('The Linear Trend')

# %%
# # Now removing the trend from the data
# y = yield_soy.values - trend
# y = pd.Series(y, index=yield_soy.index)
# _ = plt.plot(y)

# %% [markdown]
# Let's see whether removing the trend helped to make the data stationary or not.

# %%
adf = adfuller(y)[1]
kpss_ = kpss(y, nlags="auto")[1]
print(f"Dickey-Fuller test p-value is {round(adf,4)}")
print(f"KPSS test p-value is {round(kpss_, 4)}")

# %%
for i in range(4):
    if i == 0:
        y_diff = yield_soy
    elif i > 0:
        y_diff = yield_soy.diff(periods=i).dropna()
        
    print(f'p-value of randomness \
    for period={i} = {adfuller(y_diff)[1]}')
    
    print(f'p-value of stationarity \
    for period={i} = {kpss(y_diff, nlags="auto")[1]}')

# %% [markdown]
# #### Test Results
# **Dickey-Fuller Test:** The first order differencing is sufficient with p-value << 0.05
# **KPSS Test:** the first order difference is a weak stationary with p = 0.072. 
# The succesive orders make it worse. This is probably because of the variance that is changing with time. Let's see the data after differencing.

# %%
_ = yield_soy.diff().dropna().plot(title='Corn Yield - $1^{st}$ Order Difference')

# %% [markdown]
# Look's like the differencing took care of trend but the variance changes in time. The easiest way to make the variance constont is taking the log of the timeseries first

# %%
# Taking the first difference
ylog = np.log(yield_soy)
ylog_diff = ylog.diff().dropna()
# plotting y
_ = ylog_diff.plot(title = 
                   ' $ln {\ (Corn\ Yield)}$ -  $1^{st}$ Order Difference' )

# %%
# Let's make the stationarity tests again.
adf = adfuller(ylog_diff)[1]
kpss_ = kpss(ylog_diff, nlags="auto")[1]
print(f"Dickey-Fuller test p-value is {round(adf,4)}")
print(f"KPSS test p-value is {round(kpss_, 4)}")

# %% [markdown]
# This looks better in terms of variance and overall stationarity of the data. We will fit an ARIMA model now. Thus we actually do not need to take the difference but we should take the logarithm first.

# %%
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))

# Plot ACF and PACF
plot_acf(ylog_diff, lags=15, zero=False, ax=ax1)
plot_pacf(ylog_diff, lags=15, zero=False, ax=ax2)

# Show plot
plt.tight_layout()
plt.show()


# %% [markdown]
# * The auto correlation plot cuts off after the first lag 1.
# * PACF tails off. 
#
# <div class="span5 alert alert-info">
# An ARIMA model with the order (0,1,1) might fit for the time series.
#
# Recall the model choosing criteria based on ACF and PACF plots: </div>
#
# <table><tbody><tr><th></th><th>AR(p)</th><th>MA(q)</th><th>ARMA(p,q)</th></tr><tr><td>ACF</td><td>Tails off</td><td>Cuts off after lag q</td><td>Tails off</td></tr><tr><td>PACF</td><td>Cuts off after lag p</td><td>Tails off</td><td>Tails off</td></tr></tbody></table>
#

# %% [markdown]
# ### Model Selection
# Three criteria will be used for model selection.
# 1. MSE error based on timestep-wise comparison between test data and one-step prediction ARIMA model.
# 2. Akaike information criteria
# 3. Bayesian information criteria

# %%
# Import mean_squared_error and ARIMA

# Make a function called evaluate_arima_model to find the MSE of a single ARIMA model 
def evaluate_arima_model(data, arima_order):
    # Needs to be an integer because it is later used as an index.
    # Use int()
    split = int(len(data) * 0.8) 
    # Make train and test variables, with 'train, test'
    train, test = data[0:split], data[split:len(data)]
    past=[x for x in train]
    # make predictions
    predictions = list()
    for i in range(len(test)):#timestep-wise comparison between test data and one-step prediction ARIMA model. 
        model = ARIMA(past, order=arima_order)
        model_fit = model.fit()
        future = model_fit.forecast()[0]
        predictions.append(future)
        past.append(test[i])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    # Return the error
    return error


# %%
# Function to evaluate different ARIMA models with several different p, d, and q values.
def evaluate_models(dataset, p_values, d_values, q_values):
    score_dict = {'order':[], 'mse':[]}
    best_score, best_cfg = float("inf"), None
    # Iterate through p_values
    for p in p_values:
        # Iterate through d_values
        for d in d_values:
            # Iterate through q_values
            for q in q_values:
                # p, d, q iterator variables in that order
                order = (p, d, q)
                try:
                    # Make a variable called mse for the Mean squared error
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    score_dict['order'].append(order)
                    score_dict['mse'].append(mse)
                    print(f'ARIMA{order} MSE={round(mse, 4)}')
                except:
                    continue
    return pd.DataFrame(score_dict), print('Best ARIMA%s MSE=%.4f' % (best_cfg, best_score))


# %%
# Now, we choose a couple of values to try for each parameter.
# let's try up to 3 for each parameter
p_values = [i for i in range(3)] # from pacf plot the best p might be 3
d_values = [i for i in range(3)] # p-val of kpss lower but still close to 0.05. Let's try up to d=2
q_values = [i for i in range(3)] # Most likely we have model with MA order 0 since pacf plot cuts off

# %% [markdown]
# Now fitting corn_yield. Note that, this is the initial time series in which the differencing has not been done.

# %%
# Finally, we can find the optimum ARIMA model for our data.
import warnings
warnings.filterwarnings("ignore")
scores = evaluate_models(ylog, p_values, d_values, q_values)

# %% [markdown]
# The best ARIMA model based on MSE is ARIMA(0,1,1). Let's compare the models based on AIC and BIC scores to be more confident about our final model.

# %%
# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over d values from 0-2
    for d in range(3):    
    # Loop over q values from 0-2
        for q in range(3):
            # create and fit ARIMA(p,d,q) model
            try:
                model = ARIMA(ylog, order=(p, d, q))
                results = model.fit(disp=0)

                # Append order and results tuple
                order_aic_bic.append((p, d, q, results.aic, results.bic))
            except:
                continue

# %%
# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p', 'd', 'q', 'AIC', 'BIC'])

# Print order_df in order of increasing AIC
order_df.sort_values('AIC').reset_index(drop=True).head()

# %%
# Print order_df in order of increasing BIC
order_df.sort_values('BIC').reset_index(drop=True).head()

# %% [markdown]
# All three criteria, (BIC, AIC and MSE) point to same model: ARIMA(0,1,1), just like we anticipated from the ACF and PACF plots.

# %% [markdown]
# All three criteria, (BIC, AIC and MSE) points to same model: ARIMA(0,1,1), just like we anticipated from the ACF and PACF plots. 
#
# * The best model based on BIC is: ARIMA(0,1,1)
# * The best model based on AIC ARIMA(2, 1, 2), but ARIMA(0, 1, 1) score is very close to best score
# * The Best models based on the MSE are ARIMA(2,1,2), but ARIMA(0,1,1) is the simplest model with one of the lowest score
# Since the ARIMA(0,1,1) is a much simplier model than ARIMA(2,1,2) and still performance almost as well as the best model based on AIC and MSE criteria, I will fit the series with ARIMA(0,1,1)

# %%
arima = ARIMA(ylog,order=(0,1,1))
model = arima.fit()
forecast = model.forecast(25)
y_pred = model.predict()

# %%
model.summary()

# %% [markdown]
# <p>Here is a reminder of the tests in the model summary:</p>
#
# <table>
#   <tbody><tr>
#     <th>Test</th>
#     <th>Null hypothesis</th>
#     <th>P-value name</th>
#   </tr>
#   <tr>
#     <td>Ljung-Box</td>
#     <td>There are no correlations in the residual<br></td>
#     <td>Prob(Q)</td>
#   </tr>
#   <tr>
#     <td>Jarque-Bera</td>
#     <td>The residuals are normally distributed</td>
#     <td>Prob(JB)</td>
#   </tr>
# </tbody></table>

# %% [markdown]
# PRob(Q) = 0.13 > 0.05. We should reject the null hypothesis and deduce that there are correlations in the residuals. Moreover, Prob(JB) < 0.05 i.e. residuals not normally distribute based on the Jarque-Bera test. 

# %% [markdown]
# ### Model Diagnostics

# %%
_ = fitted.plot_predict()

# %%
import statsmodels.api as sm
import scipy.stats as stats
# Plot residual errors
residuals = pd.DataFrame(model.resid)
fig, ax = plt.subplots(2,2,figsize=(12,8))
# residuals.plot.scatter(x=residuals.index, y=residuals.values, title="Residuals", ax=ax[0], linestyle=None, marker='.')
ax[0,0].scatter(x=residuals.index, y=residuals.values, marker='.')
ax[0,0].plot(residuals.index, np.zeros(len(residuals)), 'k--')
plot_acf(residuals, ax=ax[0,1], zero=False)
# residuals.plot(kind='kde', title='Density', ax=ax[1,0])
sns.kdeplot(residuals.values.reshape(-1,), ax=ax[1,0])
ax[1,0].hist(residuals,density=True)
# sm.qqplot(residuals, line='45',ax=ax[1,1])
stats.probplot(residuals.values.reshape(-1,), dist="norm", plot=plt)
plt.tight_layout()
plt.show()

# %%
from scipy.stats import shapiro
stat, p = shapiro(residuals)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print("Data follows Normal Distribution")
else:
    print("Data does not follow Normal Distribution")

# %%
from scipy.stats import anderson
result = anderson(residuals.values.reshape(-1,))
print('stat=%.3f' % (result.statistic))
for i in range(len(result.critical_values)):
 sl, cv = result.significance_level[i], result.critical_values[i]
 if result.statistic < cv:
  print('Data follows Normal at the %.1f%% level' % (sl))
 else:
  print('Data does not follows Normal at the %.1f%% level' % (sl))

# %% [markdown]
# <div class="span5 alert alert-warning">
# Model have some outliers, the residuals are not normally distributed. The only good news is, there is no correlation in the residuals.
#     </div>

# %% [markdown]
# Let's see the model with the un-modified yield data, plus with a dynamic forecast.

# %%
# Create Training and Test
train = ylog[:119]
test = ylog[119:]
print(f'Test percent: {100*round(len(test)/len(train), 2)}')


# %%
# Build Model
# model = ARIMA(train, order=(3,2,1))    
def plot_dynamic_pred(dt, dtrain, dtest, order):
    model = ARIMA(dtrain, order=order)  
    fitted = model.fit(disp=-1)  
    ypred = model.predict(dtrain, dynamic=True)
    ypred = pd.Series(ypred,index=dtrain.index[order[1]:])
    forecast_period= len(test)+25
    # Forecast
    fc, se, conf = fitted.forecast(forecast_period, alpha=0.05)  # 95% conf

    # Make as pandas series

    date_range = pd.date_range(dt.index[-len(test)], periods = forecast_period, 
                  freq='Y').strftime("%Y-%m-%d").tolist()
    date_range = pd.to_datetime(date_range)

    fc_series = pd.Series(fc, index=date_range)
    lower_series = pd.Series(conf[:, 0], index=date_range)
    upper_series = pd.Series(conf[:, 1], index=date_range)

    # Plot
    _=plt.figure(figsize=(12,5), dpi=100)
    _ = plt.plot(dtrain, label='training')
    # _ = fitted.plot_predict()
    _= plt.plot(dtest, label='test')
    _= plt.plot(fc_series, label='forecast', c='k')
    _= plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='k', alpha=.15)

    _=plt.title('Dynamic Forecast vs Actual Values')
    _=plt.legend(loc='upper left', fontsize=8)
    # plt.show()
plot_dynamic_pred(ylog,train, test, (0,1,1))

# %% [markdown]
# <div class="span5 alert alert-info">
#     Finally let's try the original data without taking logarithm as we did before to fix the variance change in time. In the previous model, we had a trouble finding the best model order. We chose ARIMA(0,1,1), but the residuals were not normal. Whereas any more complicated models actually make worse of both the normallity of the residuals and other criteria like mse. 

# %% [markdown]
# This time let's use sklearns' SARIMAX method, which is equivalent its ARIMA model but have some more features.

# %%
# Let's test mse for ARIMA models with original data
y = corn_yield
import warnings
warnings.filterwarnings("ignore")
scores = evaluate_models(y, p_values, d_values, q_values)

# %%
# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over d values from 0-2
    for d in range(3):    
    # Loop over q values from 0-2
        for q in range(3):
            # create and fit ARIMA(p,d,q) model
            try:
                model = ARIMA(y, order=(p, d, q))
                results = model.fit(disp=0)

                # Append order and results tuple
                order_aic_bic.append((p, d, q, results.aic, results.bic))
            except:
                continue

# %%
# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p', 'd', 'q', 'AIC', 'BIC'])

# Print order_df in order of increasing AIC
order_df.sort_values('AIC').reset_index(drop=True).head()

# %%
order_df.sort_values('BIC').reset_index(drop=True).head()

# %% [markdown]
# All three criteria we used point to same model, again, but this time the order of the best model is (0,2,2)

# %%
# Calling SARIMAX and fitting to original data
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 2, 2))

results = mod.fit()

# one-step ahead prediction
pred = results.get_prediction(start=pd.to_datetime('1870-01-01'), dynamic=False)
pred_ci = pred.conf_int()

# Dynamic Prediction
pred_dynamic = results.get_prediction(start=pd.to_datetime('1945-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=50)

# Get confidence intervals of forecasts
pred_uc_ci = pred_uc.conf_int()

# %%
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# %%
from scipy.stats import shapiro
stat, p = shapiro(residuals)
print('stat=%.3f, p=%.5f' % (stat, p))
if p > 0.05:
    print("Data follows Normal Distribution")
else:
    print("Data does not follow Normal Distribution")

# %%
fig, ax = plt.subplots(figsize=(10,5))
ax = corn_yield.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
pred_uc.predicted_mean.plot(ax=ax, label='Forecast', c='k')
ax.fill_between(pred_uc_ci.index,
                pred_uc_ci.iloc[:, 0],
                pred_uc_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')
ax.set_ylabel('Corn, Grain - Yield')
plt.legend()

plt.show()
 


# %% [markdown]
# #### Plot Dynamic prediction

# %%
# Create Training and Test and plot dynamic prediction
train = y[:119]
test = y[119:]
plot_dynamic_pred(y,train, test, (0,2,2))

# %% [markdown]
# ### Conclusion For ARIMA Models
#
# <div class="span5 alert alert-info">
#     Arima models for the corn, grain production for ILLINOIS, can be considered as having limited prediction capability. I tried two models, one with the original data and the other one with the natural logarithm of the actual values to make the data more stationary by cancelling the temporal variance change. However, both of the model has suffered from non-normal residual distriution at the end. Both of the model have residuals without autocorrelation at any lag. 
#     
# Overall, I would not suggest using ARIMA model for corn, grain yield. Although here I showed analysis for ILLINOIS, I made the same analysis for a few other states, none of which showed any promises for an ARMA model. Also not shown here is the soy bean yields, which have very similar results. 
#
# Perhaps,the better way to model crop yield would be using the agro-climatic indicators as features and building a regression model. In the next chapter I will do this. However, this time the major limitation is the lack of sufficiently long time series data for agro-climatic indicators, which starts at 1980s. The major challenge would be cross-validation and testing the model.
#     </div>

# %% [markdown]
# # Regression Model For the Crop Yield vs. Agro-Climatic Indicators

# %%

# %%

# %%
