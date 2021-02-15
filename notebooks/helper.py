import seaborn as sns
import matplotlib.pyplot as plt

def get_demographic_plots(df_exp):
    fig, axs = plt.subplots(4, 3, figsize = (15,15))
    
    plt1 = sns.violinplot(y = df_exp["age"], ax = axs[0, 0])
    plt1.set(xlabel = '', ylabel = 'Age groups')

    genders = df_exp.sex.value_counts()
    plt1 = sns.barplot(x = genders.index, y = genders.values, ax = axs[0, 1])
    plt1.set(xlabel = '', ylabel = 'Gender')

    hhsize = df_exp.hhsize.value_counts()
    plt1 = sns.barplot(x = hhsize.index, y = hhsize.values, ax = axs[0, 2])
    plt1.set(xlabel = '', ylabel = 'Household Size')

    numchild = df_exp.num_child.value_counts()
    plt1 = sns.barplot(x = numchild.index, y = numchild.values, ax = axs[1, 0])
    plt1.set(xlabel = '', ylabel = 'Number of children')

    blsurbn = df_exp.blsurbn.value_counts()
    plt1 = sns.barplot(x = blsurbn.index, y = blsurbn.values, ax = axs[1, 1])
    plt1.set(xlabel = '', ylabel = 'Area of living')

    educatio = df_exp.educatio.value_counts()
    plt1 = sns.barplot(x = educatio.index, y = educatio.values, ax = axs[1, 2])
    plt1.set(xlabel = '', ylabel = 'Level of education')

    race = df_exp.race.value_counts()
    plt1 = sns.barplot(x = race.index, y = race.values, ax = axs[2, 0])
    plt1.set(xlabel = '', ylabel = 'Race')

    empstat = df_exp.empstat.value_counts()
    plt1 = sns.barplot(x = empstat.index, y = empstat.values, ax = axs[2, 1])
    plt1.set(xlabel = '', ylabel = 'Employment Status')

    emptype = df_exp.emptype.value_counts()
    plt1 = sns.barplot(x = emptype.index, y = emptype.values, ax = axs[2, 2])
    plt1.set(xlabel = '', ylabel = 'Employment Type')

    occup = df_exp.occup.value_counts()
    plt1 = sns.barplot(x = occup.index, y = occup.values, ax = axs[3, 0])
    plt1.set(xlabel = '', ylabel = 'Occupation')

    marital = df_exp.marital.value_counts()
    plt1 = sns.barplot(x = marital.index, y = marital.values, ax = axs[3, 1])
    plt1.set(xlabel = '', ylabel = 'Marital Status')
    
    plt1 = sns.violinplot(y = df_exp['income'], ax = axs[3, 2])
    plt1.set(xlabel = '', ylabel = 'Income')

    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation = 90)

    return plt.tight_layout()

def get_expenditure_boxplots(df_exp_values):
    fig, axs = plt.subplots(len(list(df_exp_values)) // 2 + 1, 2, figsize = (20, 100))
    rindex = 0
    for index, column in enumerate(list(df_exp_values)):
        cindex = index % 2
        if index > 0 and cindex == 0:
            rindex += 1
        vars()[f'plt{index}'] = sns.boxplot(x=column, data=df_exp_values, ax = axs[rindex, cindex])
    return plt