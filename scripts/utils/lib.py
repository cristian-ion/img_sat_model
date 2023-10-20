import matplotlib.pyplot as plt  # plotting
import seaborn as sns


def histogram_boxplot(data, feature, figsize=(12, 7), bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """

    f2, (ax_box2, ax_hist2, ax_hist3) = plt.subplots(
        nrows=3,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.2, 0.4, 0.4)},
        figsize=figsize,
    )  # creating the 2 subplots

    # boxplot will be created and a triangle will indicate the mean value of the column
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")

    # for histogram
    if bins:
        sns.histplot(
            data=data, x=feature, kde=False, ax=ax_hist2, bins=bins, stat="count"
        )
    else:
        sns.histplot(data=data, x=feature, kde=False, ax=ax_hist2, stat="count")

    if bins:
        sns.histplot(
            data=data, x=feature, kde=True, ax=ax_hist3, bins=bins, stat="density"
        )
    else:
        sns.histplot(data=data, x=feature, kde=True, ax=ax_hist3, stat="density")

    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

    ax_hist3.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist3.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc is True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot
