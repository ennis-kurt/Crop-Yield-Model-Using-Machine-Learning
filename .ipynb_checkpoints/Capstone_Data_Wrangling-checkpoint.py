# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
#import cdsapi
import pandas as pd
#import netcdf4
import xarray as xarr # pandas based library for 
            # labeled data with N-D tensors at each dimensions

import matplotlib.pyplot as plt
# %matplotlib inline 
import cartopy
import cartopy.crs as ccrs

import pandas_profiling
import seaborn as sns


# %%
def pull_data(_start, _end):
    """ Downloads the data from CDS Servers in netcdf format 
    in to the working directory
    
    _start: the starting year type: integer
    _end: the end year type: integer
    does not return anything.
    """
    year_range = pd.date_range(start=str(_start), end=str(_end), freq='YS').year

    years = []
    for item in year_range:
        years.append(str(item))
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': '2m_temperature',
            'year': years,
            'month': [
                '01', '02', '06',
                '07', '08', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                49, -128, 23,
                -66,
            ],
            'format': 'netcdf',
        },
        'download.nc')
for _start, _end in [(1979, 1992), (1993,2019), (2020, 2020) ]:
    pull_data(_start, _end )

# %%
# Read the data Path where it is stored on the Computer
#data_dir = input('Path to the data\n')
data_dir = os.path.join('C:\\'
                        'Users',
                        'kurt_',
                        'Data',
                        'crop_climotology','')
data_dir

# %% tags=[]
# Import data as xarray dataset from the directory
dask = True
if dask:
    # Import with dask
    clim = xarr.open_mfdataset(data_dir+'*.nc', parallel=True, 
                              combine='by_coords', chunks={'time': 20}
                             , engine='netcdf4')
    print(f'The chunk size for time dimension is {clim.chunks["time"][0]}\n')
    print(f'dataset, thus, have {len(clim.time)/clim.chunks["time"][0]} chunks')
else:
    # Import without dask for debugging
    clim = xarr.open_mfdataset(data_dir+'*.nc', parallel=False, 
                          combine='by_coords', engine='netcdf4')

# %%
#print(clim.data_vars)
#print(clim.coords)
clim

# %% tags=[]
for var in clim:
    print(f'{var}: {clim[var].attrs}')
# Let's select the first time step and plot the 2m-air temperature

# Let's check the dimensions
for dim in clim.dims:
    dimsize = clim.dims[dim]
    print(f'\nData has {dimsize} {dim} ')
    if dim == 'latitude':
        print(f' latitudes: from {float(clim[dim].min())} degree South',
     f'to {float(clim[dim].max())} degree North')
    if dim == 'longitude':
        print(f' Longitudes: from {float(clim[dim].max())} degree East',
     f'to {float(clim[dim].min())} degree West')
    if dim == 'time':
        print(f'time: from {pd.to_datetime(clim["time"].min().values)} to {pd.to_datetime(clim["time"].max().values)} ')

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
geo_usa.apply(lambda x: ax.annotate(s=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=7),axis=1);
#plt.tight_layout()
plt.show()
plt.close()

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
mnt_sub = clim.salem.subset(shape=MT_coord, margin=10)
MT_coord.apply(lambda x: ax.annotate(s=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=7),axis=1);
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

# %% tags=[]
#  Now Reading All the Crop Data that we are interested in to data
path = "C:\\Users\\kurt_\\Dropbox\\code\\dell-github\\SpringBoard\\Capstone2\\Data\\"
files = [path+"Alfala.csv", path+"Corn.csv", path+"Soybean.csv"]
df = (pd.read_csv(f) for f in files)
df_crop = pd.concat(df, ignore_index=True)
print(df_crop.info())
df_crop.head(3)

# %%
# Missing Values
nan=pd.DataFrame(df_crop.isnull().sum().sort_values(ascending=False), columns = ['NULL values'])
nan

# %% [markdown]
# ### Cleaning Data
# Handling the missing values and getting rid of unrelavant columns:
#
# * First ** Value column is object type. We need to convert it to Float. **
# * Column CV (%) has many missing values. Let's drop that column.
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
# According to the [Quick Stats Glossary](https://quickstats.nass.usda.gov/src/glossary.pdf) published by USDA for this dataset "D" corresponds to: 
#
# > Withheld to avoid disclosing data for individual operations.
#
# Thus all we can do is dropping these rows.
#

# %%
df_crop['Value'] = pd.to_numeric(df_crop['Value'], errors='coerce')
df_crop['Value'].isnull().sum()

# %% [markdown]
# Now Let's drop all nan values together.

# %% tags=[]
#first get rid of the nan values in Value column only
df_crop.dropna(subset=['Value'], inplace=True)
# Now we can drop all the columns with nan values at once
df_crop.dropna(axis=1, inplace=True)
print(df_crop.info())
df_crop.head(3)


# %% [markdown]
# ** There are some constant values and some entries we are not interested in. Let's remove them from the df_crop **

# %%
# First let's take a look at the number of unique 
# values in each column in a bar plot except the values column
plt.style.use('ggplot')
df_crop.nunique().plot(kind='bar', logy=True)
plt.xlabel('Features')
plt.ylabel('Count')
plt.show()
plt.close()
# Find those columns with constant values

# %% [markdown]
# ** We need only Year entries for Period. Let's see what else is in there  **

# %% [markdown]
# Now dropping all rows with forecast entries given in Period

# %%
xf = df_crop['Period'] == 'YEAR'
df_crop = df_crop[xf]
df_crop.nunique()

# %% [markdown]
# ** Let's take a look at Domain Catogories. **
# * For this study some of the domains are irrelavant, like "ECONOMIC CLASS"
# * Domain Catogories should be consistent among all "Data Item"s

# %% tags=[]
print(df_crop['Domain Category'].value_counts(),'\n\n')
y = (df_crop['Domain Category'] == 'NOT SPECIFIED') & (df_crop['Program'] == 'SURVEY')
df_crop[~(y)].Commodity.value_counts()

# %% [markdown]
# There are 69 Domain Categories, but only Soybeans from CENSUS have Categories other than "NOT SPECIFIED".
# ** Hence, for now I seperate all those Domain Categories to deal with them later since they require special handling. **
#
# Note that these entries are for CENCUS records. Thus it is already more reasonable to not mix data from two different source ("Survey and Cencus")

# %% tags=[]
df_crop = df_crop[y]
df_crop2 = df_crop[~y]
df_crop.info()

# %% [markdown]
# We are only interested some of these columns. Let's drop the ones we will not use.
#
# ** We do not need constant value columns **

# %% tags=[]
# Getting rid of features with constant values
nunq = df_crop.nunique()
for clm in df_crop:
    if nunq.loc[clm] == 1:
        df_crop.drop(clm, axis=1, inplace=True)
print(df_crop.nunique(),'\n\n')
df_crop.reset_index(drop=True, inplace=True)
df_crop.info()

# %%
df_crop.sample(5)

# %% tags=[]
# # Let's get rid of the rows where Period is smth other than Year
# period_to_rid = set(hay['Period']).difference(['YEAR'])
# print(f'The following period entries will be removed:\n {period_to_rid}')
# Period_rows = hay['Period'].isin(period_to_rid)
# hay = hay[~Period_rows]
# hay['Period'].value_counts()

# %% [markdown]
# ## Pre-Processing & Exploring the Data
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
df_crop['Data Item'].value_counts()


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

    crop_type = [" - ",", IRRIGATED - ", ", NON-IRRIGATED - ", ", ORGANIC - "]
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
crop_trend_plot("HAY, ALFALFA")

# %% [markdown]
# ## Corn Grain

# %% tags=[]
crop_trend_plot("CORN, GRAIN")

# %% [markdown]
# ## Soybeans

# %%
crop_trend_plot("SOYBEANS")

# %% [markdown]
# ## Dealing with Trends for Alfala
# * There is a clear trend in Alfala production and yield except for Non-Irrigated Alfala
# * Significant trend in Alfala is perhaps related to development in agricultral technologies
# * Warming climate may also have a positive impact. Irrigated crops might benefit from warmer temperature due to increased photosynthesis. (** REF. **)
# * Non-Irrigated crops may or may not tolarate warmer climate depending on soil water availability and some other factors. This would explain the difference between Irrigated and Non-irrigated Alfala yield trends. 
# * The priority of this project is to investigate how crop yield is affected by weather conditions. Therefore interannual variability in yield is more important for this work than decadal variability where the latter is related not only Climate but also development in agricultural technologies and human beheviours.
#
# * Therofore, I will first ** detrend the predictor (climate) variables and response (yield). **  

# %%
a= "SOYBEANS, IRRIGATED - ACRES HARVESTED"
ah = "ACRES HARVESTED"
S = lambda s: ah in s
test = df_crop['Data Item'].apply(S)
df_crop[test]

# %%
df_crop[df_crop['Program'] == 'CENSUS']
df_crop[df_crop['Program'] == 'CENSUS']['Data Item'].value_counts()

# %% [markdown]
# ## Getting Climate For Product Zone

# %%
month_length = clim.time.dt.days_in_month
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
climate = clim.resample(time="1MS", keep_attrs=True).sum()
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
        if "mean" in clim[var].cell_methods:
            vars()[var] = clim[var].resample(time="1MS",keep_attrs=True).mean()
        elif "maximum" in clim[var].cell_methods:
            vars()[var] = clim[var].resample(time="1MS",keep_attrs=True).max()
        elif "minimum" in clim[var].cell_methods:
            vars()[var] = clim[var].resample(time="1MS",keep_attrs=True).min()
    except:
        pass

# Now let's correct resampling of each variables
for x in clim:
    try:
        eval(x)
    except NameError:
        continue
    climate[x] = eval(x)
# free up some memory
#del clim

# %%
climate.dims

# %% [markdown]
# ## Calculating the climotological state means 

# %%

# %%
state_clim.DTR

# %%
randm_day["time.month" == 2]

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
cb.set_label(f"{clim['TG'].long_name} in {clim['TG'].units}")
plt.title(randm_day.time.values)
plt.show()
plt.close()


# %%
for var in clim:
    print(f'{var}: {clim[var].attrs}')
# Let's select the first time step and plot the 2m-air temperature

# Let's check the dimensions
for dim in clim.dims:
    dimsize = clim.dims[dim]
    print(f'\nData has {dimsize} {dim} ')
    if dim == 'latitude':
        print(f' latitudes: from {float(clim[dim].min())} degree South',
     f'to {float(clim[dim].max())} degree North')
    if dim == 'longitude':
        print(f' Longitudes: from {float(clim[dim].max())} degree East',
     f'to {float(clim[dim].min())} degree West')
    if dim == 'time':
        print(f'time: from {pd.to_datetime(clim["time"].min().values)} to {pd.to_datetime(clim["time"].max().values)} ')

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
plt.style.available

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
plot_month_group('TNx',12,1,2,3,4,5,6,7,8,9,10,11)

# %% tags=[]
df_crop.profile_report(explorative=True, html={'style': {'full_width': True}})


# %%

# %%
#Turkey
#clim_loc = clim.where((clim.lat > 30) & (clim.lat < 50) & (clim.lon >20 ) & (clim.lon < 45), drop=True)
#shdf = salem.read_shapefile(salem.get_demo_file('world_borders.shp'))
#shdf = shdf.loc[shdf['CNTRY_NAME'] == 'Turkey']

feature_cols = [ 'Rainfall' ,'Temperature','Usage amount']
target_v = df['water level']
X = df[feature_cols] 
y = target_v 

from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits = 3)
for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

mean = np.mean((df.values), axis=-1, keepdims=True)
detrended = df - mean
