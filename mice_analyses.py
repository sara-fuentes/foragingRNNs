import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats
from scipy.special import erf
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import statsmodels.formula.api as smf
import os
import matplotlib.patches as mpatches


def get_regressors(df, num_trial_back=10):
    # latencies & times
    df['day'] = pd.to_datetime(df['date']).dt.date
  # Prepare df columns
    select_columns = ['session', 'outcome', 'side', 'iti_duration']  # Usa una lista per i nomi delle colonne
    # print columns
    print(df.columns)
    df_glm = df.loc[:, select_columns].copy()

    df_glm['outcome_bool'] = np.where(df_glm['outcome'] == "correct", 1, 0)

    # conditions to determine the choice of each trial:
    # if outcome "0" & side "right", choice "left" ;
    # if outcome "1" & side "left", choice "left" ;
    # if outcome "0" & side "left", choice "right" ;
    # if outcome "1" & side "right", choice "right";
    # define conditions
    conditions = [
        (df_glm['outcome_bool'] == 0) & (df_glm['side'] == 'right'),
        (df_glm['outcome_bool'] == 1) & (df_glm['side'] == 'left'),
        (df_glm['outcome_bool'] == 0) & (df_glm['side'] == 'left'),
        (df_glm['outcome_bool'] == 1) & (df_glm['side'] == 'right'),
    ]

    choice = [
        'left',
        'left',
        'right',
        'right'
    ]

    df_glm['choice'] = np.select(conditions, choice, default='other')

    # calculate correct_choice regressor L+
    # if outcome_bool 0,  L+: incorrect (0)
    # if outcome_bool 1, choice "right", L+: correct (1) because right,
    # if outcome bool 1, choice "left", L+: correct (-1) because left,

    # define conditions
    conditions = [
        (df_glm['outcome_bool'] == 0),
        (df_glm['outcome_bool'] == 1) & (df_glm['choice'] == 'right'),
        (df_glm['outcome_bool'] == 1) & (df_glm['choice'] == 'left'),
    ]

    r_plus = [
        0,
        1,
        -1,
    ]

    df_glm['r_plus'] = np.select(conditions, r_plus, default='other')
    df_glm['r_plus'] = pd.to_numeric(df_glm['r_plus'], errors='coerce')

    # calculate wrong_choice regressor L- (1 correct R, -1 correct L, 0 incorrect)

    # define conditions
    conditions = [
        (df_glm['outcome_bool'] == 1),
        (df_glm['outcome_bool'] == 0) & (df_glm['choice'] == 'right'),
        (df_glm['outcome_bool'] == 0) & (df_glm['choice'] == 'left'),
    ]

    r_minus = [
        0,
        1,
        -1,
    ]

    df_glm['r_minus'] = np.select(conditions, r_minus, default='other')
    df_glm['r_minus'] = pd.to_numeric(df_glm['r_minus'], errors='coerce')

    # Convert choice from int to num (R= 1, L=-1); define conditions
    conditions = [
        (df_glm['choice'] == 'right'),
        (df_glm['choice'] == 'left'),
    ]

    choice_num = [
        1,
        0,
    ]

    df_glm['choice_num'] = np.select(conditions, choice_num, default='other')
    # TODO: use code for RNNs here
    # Creating columns for previous trial results (both dfs)
    regr_plus = ''
    regr_minus = ''
    for i in range(1, num_trial_back + 1):
        df_glm[f'r_plus_{i}'] = df_glm.groupby('session')['r_plus'].shift(i)
        df_glm[f'r_minus_{i}'] = df_glm.groupby('session')['r_minus'].shift(i)
        regr_plus += f'r_plus_{i} + '
        regr_minus += f'r_minus_{i} + '
    regressors = regr_plus + regr_minus[:-3]

    df_glm['choice_num'] = pd.to_numeric(df_glm['choice_num'], errors='coerce')

    return df_glm, regressors

# Define a custom converter function to clean and convert iti_duration values
def clean_iti_duration(value):
    try:
        # Remove periods and convert to float
        return float(value.replace('.', ''))
    except ValueError:
        # Handle the case where conversion fails
        return None
    
def plot_GLM(ax, GLM_df, alpha=1):
    orders = np.arange(len(GLM_df))

    # filter the DataFrame to separate the coefficients
    r_plus = GLM_df.loc[GLM_df.index.str.contains('r_plus'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.contains('r_minus'), "coefficient"]
    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(r_plus)], r_plus, marker='o', color='indianred', alpha=alpha)
    ax.plot(orders[:len(r_minus)], r_minus, marker='o', color='teal', alpha=alpha)

    # Create custom legend handles with labels and corresponding colors
    legend_handles = [
        mpatches.Patch(color='indianred', label='r+'),
        mpatches.Patch(color='teal', label='r-')
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_handles)
    # ax.axhline(y=intercept, label='Intercept', color='black')
    ax.axhline(y=0, color='gray', linestyle='--')

    ax.set_ylabel('GLM weight')
    ax.set_xlabel('Previous trials')


if __name__ == '__main__':
    num_bins_iti = 4
    # TODO:
    # 1. Move part of code to get_regressors
    # 1.1. Generalize code for mice using the code for RNNs
    # 2. Compute GLM conditioning on mouse (there are 3 mice in the dataset)
    # 3. Introduce ITI as a regressor
    data_folder = 'C:\\Users\\saraf\\OneDrive\\Documentos\\IDIBAPS\\foraging RNNs\\mice\\'
    # data_folder= '/home/molano/Dropbox/Molabo/foragingRNNs/mice/'
    filename ='global_trials_sara.csv'
    # Read the CSV file and apply the custom converter function to the iti_duration column
    # try:
    #     df = pd.read_csv(str(data_folder) + str(filename), 
    #                      sep=';', 
    #                      converters={'iti_duration': clean_iti_duration},
    #                      low_memory=False)
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    iti_bins = [0, 2, 6, 12, 20]


# Read the CSV file and specify that a column (e.g., 'column_name') should be read as a float
    df = pd.read_csv(str(data_folder) + str(filename), sep=';', low_memory=False, dtype={'iti_duration': float})
    # get only trials with iti
    df = df[df['task'] != 'S4'] # quita las sessiones sin ITI
    df = df[df['subject'] != 'manual'] #
    mice_counter = 0
    f, axes = plt.subplots(1, len(df['subject'].unique()), figsize=(15, 5), sharey=True)
    for mice in df['subject'].unique():
        print(mice)
        df_mice = df.loc[df['subject'] == mice]
        # get 3 equipopulated bins of iti values
        df_mice['iti_bins'] = pd.cut(df_mice['iti_duration'], iti_bins)
        for iti_index in range(num_bins_iti):
            iti = [iti_bins[iti_index], iti_bins[iti_index + 1]]
            df_glm_mice, regressors = get_regressors(df=df_mice)
            # TODO: filter by iti
            df_glm_mice = df_glm_mice.loc[df_glm_mice['iti_bins'] == pd.Interval(left=iti[0], right=iti[1])]
            mM_logit = smf.logit(formula='choice_num ~ ' + regressors, data=df_glm_mice).fit()
            GLM_df = pd.DataFrame({
                'coefficient': mM_logit.params,
                'std_err': mM_logit.bse,
                'z_value': mM_logit.tvalues,
                'p_value': mM_logit.pvalues,
                'conf_Interval_Low': mM_logit.conf_int()[0],
                'conf_Interval_High': mM_logit.conf_int()[1]
            })
            # alpha = 1 if iti_index == 0 else subtract 0.3 for each iti_index
            alpha = 1 - 0.3 * iti_index
            # subplot title with name of mouse
            axes[mice_counter].set_title(mice)
            plot_GLM(axes[mice_counter], GLM_df, alpha=alpha)
        mice_counter += 1
    plt.show()

    # "variable" and "regressors" are columnames of dataframe
    # you can add multiple regressors by making them interact: "+" for only fitting separately,
    # "*" for also fitting the interaction
    # Apply glm
    # mM_logit = smf.logit(formula='choice_num ~ ' + regressors, data=GLM_df).fit()
 
    # prints the fitted GLM parameters (coefs), p-values and some other stuff
    # results = mM_logit.summary()
    # print(results)
    # # save param in df
    # m = pd.DataFrame({
    #     'coefficient': mM_logit.params,
    #     'std_err': mM_logit.bse,
    #     'z_value': mM_logit.tvalues,
    #     'p_value': mM_logit.pvalues,
    #     'conf_Interval_Low': mM_logit.conf_int()[0],
    #     'conf_Interval_High': mM_logit.conf_int()[1]
    # })

    # axes = plt.subplot2grid((50, 50), (39, 0), rowspan=11, colspan=32)
    # orders = np.arange(len(m))

    # # filter the DataFrame to separately the coefficients
    # r_plus = m.loc[m.index.str.contains('r_plus'), "coefficient"]
    # r_minus = m.loc[m.index.str.contains('r_minus'), "coefficient"]
    # intercept = m.loc['Intercept', "coefficient"]

    # plt.plot(orders[:len(r_plus)], r_plus, label='r+', marker='o', color='indianred')
    # plt.plot(orders[:len(r_minus)], r_minus, label='r-', marker='o', color='teal')
    # plt.axhline(y=intercept, label='Intercept', color='black')


    # axes.set_ylabel('GLM weight', label_kwargs)
    # axes.set_xlabel('Prevous trials', label_kwargs)
    # plt.legend()


    #### last PLOT : CUMULATIVE TRIAL RATE
    # axes = plt.subplot2grid((50, 50), (39, 36), rowspan=11, colspan=14)

    # df['start_session'] = df.groupby(['subject', 'session'])['STATE_center_light_START'].transform('min')
    # # Remove commas if necessary and convert to float
    # df['start_session'] = df['start_session'].str.replace(',', '').astype(float)
    # df['end_session'] = df.groupby(['subject', 'session'])['STATE_drink_delay_END'].transform('max')
    # df['session_lenght'] = (df['end_session'] - df['start_session']) / 60
    # df['STATE_center_light_START'] = df['STATE_center_light_START'].str.replace(',', '').astype(float)    
    # df['current_time'] = df.groupby(['subject', 'session'])['STATE_center_light_START'].transform(
    #     lambda x: (x - x.iloc[0]) / 60)  # MINS

    # max_timing = round(df['session_lenght'].max())
    # max_timing = int(max_timing)
    # sess_palette = sns.color_palette('Greens', 5)  # color per day

    # for idx, day in enumerate(df_mice['day'].unique()):
    #     subset = df_mice.loc[df_mice['day'] == day]
    #     n_sess = len(subset['session'].unique())
    #     try:
    #         hist_ = stats.cumfreq(subset['current_time'], numbins=max_timing,
    #                                 defaultreallimits=(0, subset['current_time'].max()), weights=None)
    #     except Exception as e:
    #         hist_ = stats.cumfreq(subset['current_time'], numbins=max_timing, defaultreallimits=(0, max_timing),
    #                                 weights=None)
    #     hist_norm = hist_.cumcount / n_sess
    #     bins_plt = hist_.lowerlimit + np.linspace(0, hist_.binsize * hist_.cumcount.size, hist_.cumcount.size)
    #     sns.lineplot(x=bins_plt, y=hist_norm, color = sess_palette[idx % len(sess_palette)], ax=axes, marker='o', markersize=4)

    # # axes.set_ylabel('Cum. nº of trials', label_kwargs)
    # # axes.set_xlabel('Time (mins)', label_kwargs)

    # # legend
    # lines = [Line2D([0], [0], color=sess_palette[i], marker='o', markersize=7, markerfacecolor=sess_palette[i]) for
    #             i in
    #             range(len(sess_palette))]
    # axes.legend(lines, np.arange(-5, 0, 1), title='Days', loc='center', bbox_to_anchor=(0.1, 0.85))
    # plt.show()