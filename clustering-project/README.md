# Zillow data logerror clustering project
A school project to build a machine learning model to predict logerror using clustering methods and regression and deliver a report on the indicators used in the model.

## About the project

The project will attempt to develop a clear understanding of where Zillow's model struggles to predict accurately.  Then hopefully use that insight to accurately predict Zillow's error.

### Steps to reproduce
First download a copy of each file or clone this repository.

To run this project you will need to fulfill one of the following requirements:
 > An environment file named `env.py` in the same directory as the other files.  The enviornment file should be structured the same as below and should contain your unique credentials to CodeUp's SQL server.
 ```py
hostname='data.codeup.com'
username='your_username'
password='your_password'
```
> A csv file named `data.csv` in the same directory as the other files.  You can download a copy of the data from [Kaggle](https://www.kaggle.com/competitions/zillow-prize-1/data)

You will also need to ensure the proper libraries are installed in your python environment.  You can install the libraries easily by running the following command in your python shell:
```py
%pip install numpy
%pip install pandas
%pip install matplotlib
%pip install seaborn
%pip install plotly
%pip install scipy
%pip install sklearn
```
___________

### Goals

Zillow, a real-estate listing website, is looking to improve their model that predicts property value.  By applying a model that can accurately predict their model's error, they can adjust the predictions of their existing model, resulting in a more accurate prediction.

*Agile Story Format:*
> As a `real-estate listing website`, 
> I want `predictions of my model's logerror`
> so that I can `reduce my model's logerror`.


### Deliverables

Since the company wants two different things that means that we will need at least two deliverables in our final product:  
 1. A storytelling document that explains at least one driver of property value and makes recommendations on how to discover further.
 2. A machine learning model that predicts the property value that will perform well in production.

Because this is a classroom project there are some additional requirements that will need to be met.  

For the storytelling document:
 - It will need to be delivered in a jupyter notebook named `Final_Report.ipynb`
    - Between five and seven visualizations included.
    - It must include 2 statistical tests

For the machine learning model:
 - It should be demonstrated in the `Final_Report.ipynb` notebook.
 - It should be scripted easily reproducible in a `model.py` helper library.
 - Since Zillow is looking for inspiration for improving their own model, I will try to create models that use finesse gained from exploration, rather than a brute-force methodology.



### Initial Hypothesis
Just like with any scientific endeavor, it is important to form a hypothesis when approaching a Data Science project.  The hypothesis gives us something concrete to start testing against.  How truthful the initial hypothesis turns out to be is ultimately unimportant since this is an iterive process and there will likely be many hypotheses throughout the life of the project.  The important part is to give us a starting point.  For the purposes of this project my initial hypothesis will be:

> Certain groups of homes are more affected by logerror than others.


### Data Dictionary   
| Original Column Name | Renamed Column Name | Status | Notes | Column Description |
|---|---|---|---|---|
| '`airconditioningtypeid`' |  | Dropped | Too many nulls |  Type of cooling system present in   the home (if any) |
| '`architecturalstyletypeid`' |  | Dropped | 99.8% Null |  Architectural style of the home   (i.e. ranch, colonial, split-level, etc…) |
| '`assessmentyear`' |  | Dropped | All values the same | The year of the property tax assessment  |
| '`basementsqft`' |  | Dropped | 99.9% Null |  Finished living area below or   partially below ground level |
| '`bathroomcnt`' |  | Dropped | Duplicates 'calc_bath' |  Number of bathrooms in home   including fractional bathrooms |
| '`bedroomcnt`' |  | Kept |  |  Number of bedrooms in home  |
| '`buildingclasstypeid`' |  | Dropped | 100% Null | The building framing type (steel frame, wood frame, concrete/brick)  |
| '`buildingqualitytypeid`' |  | Dropped | Too many nulls |  Overall assessment of condition of   the building from best (lowest) to worst (highest) |
| '`calculatedbathnbr`' | `calc_bath` | Kept |  |  Number of bathrooms in home   including fractional bathroom |
| '`calculatedfinishedsquarefeet`' | `structure_sqft` | Kept |  |  Calculated total finished living   area of the home  |
| '`censustractandblock`' |  | Dropped | Same data as lat/long + fips |  Census tract and block ID combined   - also contains blockgroup assignment by extension |
| '`decktypeid`' |  | Dropped | 99% Null | Type of deck (if any) present on parcel |
| '`finishedfloor1squarefeet`' |  | Dropped | Too many nulls |  Size of the finished living area   on the first (entry) floor of the home |
| '`finishedsquarefeet12`' |  | Dropped | Same as structure_sqft | Finished living area |
| '`finishedsquarefeet13`' |  | Dropped | 100% Null | Perimeter  living area |
| '`finishedsquarefeet15`' |  | Dropped | 100% Null | Total area |
| '`finishedsquarefeet50`' |  | Dropped | Too many nulls |  Size of the finished living area   on the first (entry) floor of the home |
| '`finishedsquarefeet6`' |  | Dropped | 99% Null | Base unfinished and finished area |
| '`fips`' |  | Kept |  |  Federal Information Processing   Standard code -  see   https://en.wikipedia.org/wiki/FIPS_county_code for more details |
| '`fireplacecnt`' |  | Dropped | Too many nulls |  Number of fireplaces in a home (if   any) |
| '`fireplaceflag`' |  | Dropped | 99.8% Null |  Is a fireplace present in this   home  |
| '`fullbathcnt`' |  | Kept |  |  Number of full bathrooms (sink,   shower + bathtub, and toilet) present in home |
| '`garagecarcnt`' |  | Dropped | Too many nulls |  Total number of garages on the lot   including an attached garage |
| '`garagetotalsqft`' |  | Dropped | Too many nulls |  Total number of square feet of all   garages on lot including an attached garage |
| '`hashottuborspa`' |  | Dropped | Too many nulls |  Does the home have a hot tub or   spa |
| '`heatingorsystemtypeid`' |  | Kept | Nulls imputed as 2 (Central) |  Type of home heating system |
| '`landtaxvaluedollarcnt`' | `tax_land` | Kept | Hoping to use in ensemble model | The assessed value of the land area of the parcel |
| '`latitude`' |  | Kept | Perhaps the most trustworthy datapoint in the set |  Latitude of the middle of the   parcel multiplied by 10e6 |
| '`longitude`' |  | Kept | Perhaps the most trustworthy datapoint in the set |  Longitude of the middle of the   parcel multiplied by 10e6 |
| '`lotsizesquarefeet`' | `lot_sqft` | Kept |  |  Area of the lot in square feet |
| '`numberofstories`' |  | Dropped | Too many nulls |  Number of stories or levels the   home has |
| '`parcelid`' |  | Kept |  |  Unique identifier for parcels   (lots)  |
| '`poolcnt`' |  | Dropped | Too many nulls |  Number of pools on the lot (if   any) |
| '`poolsizesum`' |  | Dropped | Too many nulls |  Total square footage of all pools   on property |
| '`pooltypeid10`' |  | Dropped | 99% Null |  Spa or Hot Tub |
| '`pooltypeid2`' |  | Dropped | Too many nulls |  Pool with Spa/Hot Tub |
| '`pooltypeid7`' |  | Dropped | Too many nulls |  Pool without hot tub |
| '`propertycountylandusecode`' |  | Dropped | Difficult to derive meaning from |  County land use code i.e. it's   zoning at the county level |
| '`propertylandusetypeid`' |  | Dropped | All values the same |  Type of land use the property is   zoned for |
| '`propertyzoningdesc`' |  | Dropped | Too many nulls |  Description of the allowed land   uses (zoning) for that property |
| '`rawcensustractandblock`' |  | Dropped | Same data as lat/long + fips |  Census tract and block ID combined   - also contains blockgroup assignment by extension |
| '`regionidcity`' |  | Kept |  |  City in which the property is   located (if any) |
| '`regionidcounty`' |  | Dropped | Duplicates fips | County in which the property is located |
| '`regionidneighborhood`' |  | Dropped | Too many nulls | Neighborhood in which the property is located |
| '`regionidzip`' |  | Kept |  |  Zip code in which the property is   located |
| '`roomcnt`' |  | Kept |  |  Total number of rooms in the   principal residence |
| '`storytypeid`' |  | Dropped | 99.9% Null |  Type of floors in a multi-story   house (i.e. basement and main level, split-level, attic, etc.).  See tab for details. |
| '`structuretaxvaluedollarcnt`' | `tax_structure` | Kept | Hoping to use in ensemble model | The assessed value of the built structure on the parcel |
| '`taxamount`' |  | Dropped | Leaks data about target variable | The total property tax assessed for that assessment year |
| '`taxdelinquencyflag`' |  | Kept | Made into Boolean with Null imputed as false | Property taxes for this parcel are past due as of 2015 |
| '`taxdelinquencyyear`' |  | Engineered | See 'years_tax_delinquent' | Year for which the unpaid propert taxes were due  |
| '`taxvaluedollarcnt`' | `tax` | Target | This is our target variable | The total tax assessed value of the parcel |
| '`threequarterbathnbr`' |  | Dropped | Too many nulls |  Number of 3/4 bathrooms in house   (shower + sink + toilet) |
| '`typeconstructiontypeid`' |  | Dropped | 99.8% Null |  What type of construction material   was used to construct the home |
| '`unitcnt`' |  | Kept | Nulls imputed as 1 |  Number of units the structure is   built into (i.e. 2 = duplex, 3 = triplex, etc...) |
| '`yardbuildingsqft17`' |  | Dropped | Too many nulls | Patio in  yard |
| '`yardbuildingsqft26`' |  | Dropped | 99.8% Null | Storage shed/building in yard |
| '`yearbuilt`' |  | Kept | Also engineered. See 'age' |  The Year the principal residence   was built  |
|  | `years_tax_delinquent` | Engineered | Nulls imputed as 0 | Calculated by subtacting 'taxdelinquencyyear' from 2017 |
|  | `age` | Engineered |  | Calculated by subtracting 'yearbuilt' from 2017 |
|  | `bathroom_sum` | Engineered |  | Calculated by taking 3/4 of 'threequarterbathnbr' and adding it to   'fullbathcnt' |
| `logerror` |  | Dropped | Leaks data about target variable | Log of the error produced by Zillows existing model |
| `transactiondate` |  | Dropped | Not useful in context of this project | Date of the transaction which zillow's model compared its prediction |                                                 |

### The Plan
- Acquire (`wrangle.ipynb`)
    - Collect the data into working environment. 
- Prepare (`wrangle.ipynb`)
    - Examine each column and build a data dictionary. (Above)
    - Decide on how to handle null values.
    - Transform data to make it useable for exploration and modeling.
        - Renaming columns
        - Assigning the right datatype for each column
    - Split data into 3 datasets for training, testing, and validation.
- Trim (`wrangle.ipynb`)
    - Decide how to handle outliers
    - Decide how to handle duplicates and bad records
- Pave the way (`wrangle.py`)
    - Create functions to make repeating the processes above repeatable and easy.
- Explore (`explore.ipynb`)
    - Use data visualization and statistical testing to confirm or reject hypotheses.
    - Develop new hypotheses along the way.
    - Identify features and/or engineer new ones that may be useful for modeling.
    - Create clustering models and determine their usefulness in predicting the target variable (`logerror`)
- Model (`model.ipynb`)
    - Develop baseline prediction to compare future models against.
    - Develop models and compare them to the baseline model.
    - Select a favorite model and validate it's performance on the validate sample.
- Script (`model.py`)
    - Create a helper library to make reproducing the models developed repeatable and easy.
- Deliver (`Report.ipynb`)
    - Create a professional and presentable notebook that clearly summarizes the work done.