# Crop Yield Prediction Using Machine Learning
## Introduction

While machine learning algorithms has been used in many fields the researchers in weather and climate 
applications have just recently started to take advantage of powerful ML algorithms. One of the interesting 
topics is to predict the crop production yield with the meteorological variables. Although there have been 
several applications of similar models for highly localized areas it would be very informative to build
predictive models for large geographical scales, such as for a state. Such models would be helpful for 
developing climate change adaptation strategies for agriculture. 

Weather and climate are among the most important factors that affect agriculture especially crop 
production. There is no question that a long warm and dry spell would cause a stress on crops and possibly 
reduce the yield on harvest for many products. Similarly freezing weather or very heavy rain that cause 
runoff during growing season could possibly damage the seeds or the plants after emerging. There are 
several other weather events/conditions, some with obvious positive or negative effect and some with a net 
effect which depends on when and where they happen, and which product is planted. However, even 
considering the obvious factors, building a successful statistical model for crop yield is challenging to say the 
least. This is because, the agro-climatic indicators are highly correlated while most of the machine learning 
algorithms requires non-collinearity. Moreover, finding a long enough measurement for crop yield and the 
climate variables for building a model using ML algorithms is difficult. This is especially true for large scale 
modelling. 

Therefore, the question is whether building a reliable, large scale crop production yield model is possible with 
the currently available yield and agro-climatic data sets? Here in this project, despite all these challenges, I 
will try to build predictive models to answer this question.

## Data Wrangling

In this work I use 3 different data sets each of which have different structures. Each data sets contains more 
information then needed in this work. The goal in this section is to prepare all three data sets by cleaning and 
transforming them to get the relevant information’s: crop yield as a response variable and the agro-climatic 
indicators as features for the selected states. Due to structural differences 3 different data analysis tools 
were needed to be used. These datasets are:

1. The crop yield data for alfalfa, corn, and soybeans for each state in the U.S. The goal is to clean the 
data and get the State, Year and Value (Yield) columns for the selected states. Pandas package will 
be used for analysis and manipulation. Data source: United States Department of Agriculture -
National Agricultural Statistics Service.
2. The data for the coordinates of the state boundaries. This data is used to select the region of interest 
in the climate dataset. Geopandas and Salem packages are used for this data.3. Agro-climatic indicators, which include fundamental climate variables, such as temperature and rain, 
and the ones that are derived from them such as biologically available degree days. This dataset is a 
time series of 26 agro-climatic indicators with additional two spatial coordinates (see Figure 1). 
Xarray package with dask integration is used to analysis and manipulation. The entire dataset comes 
as three separate files in netcdf format, with either dekadal (10 day), seasonal or annual aggregation 
of certain variables. Dekadal data has a restrictive size to process on a regular home computer. 
Hence the data is read in to 54 chunks. The goal is to 
• annually aggregate all the indicators and merge them into a single xarray dataset, 
• using the state boundaries data, get the spatial averages of each variables for selected states,
• convert the final data to pandas dataframe,
• merge the climate indicators and crop yield data into a single pandas dataframe
