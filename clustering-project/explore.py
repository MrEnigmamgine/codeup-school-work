import itertools
import seaborn as sns
import matplotlib.pyplot as plt

def plot_categorical_and_continuous_vars(df, categorical_cols, continuous_cols):
    """Spits out a bunch of plots to speed up exploration."""
    # Save ourselves some time on huge datasets
    if len(df) > 100_000:
        df = df.sample(100_000)
    
    # Some quickmaths to determine each figure size.
    units = 4
    width = 3*units
    height = len(categorical_cols)*units-1

    # Make a new subplots figure for each continuous variable
    for con in continuous_cols:
        fig, ax = plt.subplots(nrows=len(categorical_cols), ncols=3, figsize=(width,height))

        # Make a row consisting of a bar, box, and violin plot for each categorical variable
        for row, cat in enumerate(categorical_cols):
            # For readability, get a sorted list of each categorical bucket
            sort_order = df[cat].sort_values().unique().tolist()

            plot1 = sns.barplot(data=df, x=cat, y=con, ax=ax[row][0], order=sort_order)
            plot1.set_ylabel(cat)   # Only the leftmost plot gets a ylabel, labeling the entire row (not the y axis!)
            plot1.set_xlabel(None)  # Remove the xlabel, that goes in the ylabel
            # Rotate xtick labels so that they can't overlap each other.
            for item in plot1.get_xticklabels():
                item.set_rotation(90)
            
            plot2 = sns.boxplot(data=df, x=cat, y=con, ax=ax[row][1], order=sort_order)
            plot2.set_ylabel(None)
            plot2.set_xlabel(None)
            for item in plot2.get_xticklabels():
                item.set_rotation(90)
            
            plot3 = sns.violinplot(data=df, x=cat, y=con, ax=ax[row][2], order=sort_order)
            plot3.set_ylabel(None)
            plot3.set_xlabel(None)
            for item in plot3.get_xticklabels():
                item.set_rotation(90)
        # Title the figure with the continuous variable so we know what we are looking at.
        fig.suptitle(con)
        fig

def plot_variable_pairs(df):
    cols = list(df.dtypes[(df.dtypes == 'int64')| (df.dtypes == 'float64')].index)
    pairs = list(itertools.combinations(cols, 2))
    for pair in pairs:
        plt.scatter(df[pair[0]], df[pair[1]])
        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.title(pair)
        plt.show()