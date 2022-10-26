# Codeup Curriculum Access Logs Project
By Daniel Ford, Jarad Angel, Joshua Mayes

## Readme Outline
- [Project Description](#project_desc)  
    - [Scenario](#scenario)
    - [Goals](#goals)
        - [Deliverables](#deliverables)
    - [Project Dependencies](#dependencies)

<!-- - [Project Planning](#plan)   -->

- [About the data](#data)
    - Scope
    - Acquiring
    - Preparing
    - Data Dictionary







# About the project <a name="project_desc"></a>

This project is a group classroom project in which we are "roleplaying" as an analyst who has been emailed by a colleague with some questions about the statistics for CodeUp's curriculum access logs.  


## Scenario <a name="scenario"></a>

A colleague has emailed our group with a few specific questions.  They would like the questions answered in time for a board meeting, and there is an assumption that they will be speaking about these questions in that board meeting.  The questions they would like answered include:

1. Which lesson appears to attract the most traffic consistently across cohorts (per program)?
2. Is there a cohort that referred to a lesson significantly more than other cohorts seemed to gloss over?
3. Are there students who, when active, hardly access the curriculum? If so, what information do you have about these students?
4. Is there any suspicious activity, such as users/machines/etc. accessing the curriculum who shouldn’t be? Does it appear that any web-scraping is happening? Are there any suspicious IP addresses?
5. At some point in 2019, the ability for students and alumni to access both curriculums (web dev to ds, ds to web dev) should have been shut off. Do you see any evidence of that happening? Did it happen before?
6. What topics are grads continuing to reference after graduation and into their jobs (for each program)?
7. Which lessons are least accessed?
8. Anything else I should be aware of?

> __Agile Story__  
    As a `speaker at a board meeting`  
    I want `an analytical report of how CodeUp's curriculum is used`  
    So that `I can enable business decisions`    

## Goals <a name="goals"></a>

Use Exploratory Data Analysis (EDA) techniques to generate the following reports.
- Most accessed and least accessed lesson for each cohort separated by program type
- Students who accessed the lessons less than 10 times during enrollment
- The top 3 topics which are accessed by students with an ended enrollment.
- List of suspicious events discovered during analysis
- Report of users accessing lessons outside of their enrolled curriculum.


### Deliverables <a name="deliverables"></a>

- Final report: (Report.ipynb) A jupyter notebook that collects and displays the work done to answer these questions.
- Response email: An html email that briefly answers the questions.

## Reproducing this project <a name="requirements"></a>

{Are there any special considerations one must take to run this project on another machine?  Usually yes.  The most common considerations have been filled out already below.}

### Dependencies

This project makes use of several technologies that will need to be installed
* [![python-shield](https://img.shields.io/badge/Python-3-blue?&logo=python&logoColor=white)
    ](https://www.python.org/)
* [![jupyter-shield](https://img.shields.io/badge/Jupyter-notebook-orange?logo=jupyter&logoColor=white)
    ](https://jupyter.org/)
* [![numpy-shield](https://img.shields.io/badge/Numpy-grey?&logo=numpy)
    ](https://numpy.org/)
* [![pandas-shield](https://img.shields.io/badge/Pandas-grey?&logo=pandas)
    ](https://pandas.pydata.org/)
* [![matplotlib-shield](https://img.shields.io/badge/Matplotlib-grey.svg?)
    ](https://matplotlib.org)
* [![seaborn-shield](https://img.shields.io/badge/Seaborn-grey?&logoColor=white)
    ](https://seaborn.pydata.org/)
* [![scipy-shield](https://img.shields.io/badge/SciPy-grey?&logo=scipy&logoColor=white)
    ](https://scipy.org/)
* [![sklearn-shield](https://img.shields.io/badge/_-grey?logo=scikitlearn&logoColor=white&label=scikit-learn)
    ](https://scikit-learn.org/stable/)

Dependencies can be installed quickly with just a few lines of code.
```
%pip install notebook
%pip install numpy
%pip install pandas
%pip install matplotlib
%pip install seaborn
%pip install scipy
%pip install sklearn
```


# About the data <a name="data"></a>

The log data comes from CodeUp's SQL server, with each row representing a log entry for the curriculum's access logs.  

IP Geolocation information is acquired from www.geoplugin.net's free public API, which contains the current location information for each IP address.

## Scope

The data set consists of 900K rows containing log entries from 5500 unique IP addresses accessed CodeUp's curriculum between January 2018 and April 2021.


## Acquiring

The logs were accessed via CodeUp's SQL server using the curriculum_logs database.  The necessary code is compiled in `acquire.py`

The IP geolocation information was acquired via API calls to www.geoplugin.net. The necessary code is compiled in `ip_tk.py`

## Preparing

The steps below were taken:
- Ensuring that the data has the appropriate datatypes such setting dates to be recognized as datetime objects.
- The program ID was translated to a program name by using CodeUp's alumni page, which displays the program name when a cohort is selected.
- Modules and lessons were inferred by parsing the endpoint string.  Anything left of the first slash ("/") was inferred as the module's name, and anything to the right was inferred as the lesson name.
- 

## Data Dictionary