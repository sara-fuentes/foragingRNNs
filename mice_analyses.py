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

def get_regressors(df):
    # latencies & times
    df['day'] = pd.to_datetime(df['date']).dt.date
    # df['center_response_time'] = df['STATE_center_light_END'] - df['STATE_center_light_START']
    # df['response_time'] = df['STATE_side_light_END'] - df['STATE_side_light_START']
    # df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    # df['centre_response_latency'] = df['center_response_time']
    # df['Port5In_START'] = df['Port5In_START'].astype(str)
    # df['Port2In_START'] = df['Port2In_START'].astype(str)
    # df['first_response_right'] = df['Port5In_START'].str.split(',').str[0].astype(float)
    # df['first_response_left'] = df['Port2In_START'].str.split(',').str[0].astype(float)
    # df['center_median_response_time'] = df['centre_response_latency'].median()  # median latency to first response
    # df['response_latency_median'] = df['response_time'].median()  # median latency to first response
    # df['probability_r'] = np.round(df['probability_r'], 1)

    # # teat well the NANs and  List of conditions for knowing which one was the first choice in each trial
    # df = df.replace(np.nan, 0)

    # # List of conditions for teat well the NANs
    # conditions = [
    #     (df.first_response_left == 0) & (df.first_response_right == 0),
    #     df.first_response_left == 0,
    #     df.first_response_right == 0,
    #     df.first_response_left <= df.first_response_right,
    #     df.first_response_left > df.first_response_right,
    # ]
    # choices = ["no_response",
    #            "right",
    #            "left",
    #            "left",
    #            "right"]
    # df["first_trial_response"] = np.select(conditions, choices)

    # df["correct_outcome_bool"] = df["first_trial_response"] == df["side"]
    # df["correct_outcome_int"] = np.where(df["first_trial_response"] == df["side"], 1,
    #                                      0)  # (1 = correct choice, 0= incorrect side)

    return df


if __name__ == '__main__':
    # TODO:
    # 1. Move part of code to get_regressors
    # 1.1. Generalize code for mice using the code for RNNs
    # 2. Compute GLM conditioning on mouse (there are 3 mice in the dataset)
    # 3. Introduce ITI as a regressor
    data_folder= '/home/molano/Dropbox/Molabo/foragingRNNs/mice/'
    filename ='global_trials_100524.csv'
    df = pd.read_csv(str(data_folder) + str(filename), sep=';', low_memory=False)
    df = get_regressors(df)
    # Prepare df columns
    # Converting the 'outcome' column to boolean values
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
    # if outcome_bool 1,  L-: correct (0)
    # if outcome_bool 0 & choice "right", L-: incorrect (1) because right,
    # if outcome bool 0 & choice "left", L-: incorrect (-1) because left,

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
    for i in range(1, 21):
        df_glm[f'r_plus_{i}'] = df_glm.groupby('session')['r_plus'].shift(i)
        df_glm[f'r_minus_{i}'] = df_glm.groupby('session')['r_minus'].shift(i)

    df_glm['choice_num'] = pd.to_numeric(df_glm['choice_num'], errors='coerce')

    # "variable" and "regressors" are columnames of dataframe
    # you can add multiple regressors by making them interact: "+" for only fitting separately,
    # "*" for also fitting the interaction
    # Apply glm
    mM_logit = smf.logit(
        formula='choice_num ~ r_plus_1 + r_plus_2 + r_plus_3 + r_plus_4 + r_plus_5 + r_plus_6+ r_plus_7+ r_plus_8'
                '+ r_plus_9 + r_plus_10 + r_plus_11 + r_plus_12 + r_plus_13 + r_plus_14 + r_plus_15 + r_plus_16'
                '+ r_plus_17 + r_plus_18 + r_plus_19+ r_plus_20'
                '+ r_minus_1 + r_minus_2 + r_minus_3 + r_minus_4 + r_minus_5 + r_minus_6+ r_minus_7+ r_minus_8'
                '+ r_minus_9 + r_minus_10 + r_minus_11 + r_minus_12 + r_minus_13 + r_minus_14 + r_minus_15 '
                '+ r_minus_16 + r_minus_17 + r_minus_18 + r_minus_19+ r_minus_20',
        data=df_glm).fit()

    # prints the fitted GLM parameters (coefs), p-values and some other stuff
    results = mM_logit.summary()
    print(results)
    # save param in df
    m = pd.DataFrame({
        'coefficient': mM_logit.params,
        'std_err': mM_logit.bse,
        'z_value': mM_logit.tvalues,
        'p_value': mM_logit.pvalues,
        'conf_Interval_Low': mM_logit.conf_int()[0],
        'conf_Interval_High': mM_logit.conf_int()[1]
    })

    axes = plt.subplot2grid((50, 50), (39, 0), rowspan=11, colspan=32)
    orders = np.arange(len(m))

    # filter the DataFrame to separately the coefficients
    r_plus = m.loc[m.index.str.contains('r_plus'), "coefficient"]
    r_minus = m.loc[m.index.str.contains('r_minus'), "coefficient"]
    intercept = m.loc['Intercept', "coefficient"]

    plt.plot(orders[:len(r_plus)], r_plus, label='r+', marker='o', color='indianred')
    plt.plot(orders[:len(r_minus)], r_minus, label='r-', marker='o', color='teal')
    plt.axhline(y=intercept, label='Intercept', color='black')


    # axes.set_ylabel('GLM weight', label_kwargs)
    # axes.set_xlabel('Prevous trials', label_kwargs)
    plt.legend()


    #### last PLOT : CUMULATIVE TRIAL RATE
    axes = plt.subplot2grid((50, 50), (39, 36), rowspan=11, colspan=14)

    df['start_session'] = df.groupby(['subject', 'session'])['STATE_center_light_START'].transform('min')
    df['end_session'] = df.groupby(['subject', 'session'])['STATE_drink_delay_END'].transform('max')
    df['session_lenght'] = (df['end_session'] - df['start_session']) / 60
    df['current_time'] = df.groupby(['subject', 'session'])['STATE_center_light_START'].transform(
        lambda x: (x - x.iloc[0]) / 60)  # MINS

    max_timing = round(df['session_lenght'].max())
    max_timing = int(max_timing)
    sess_palette = sns.color_palette('Greens', 5)  # color per day

    for idx, day in enumerate(df.day.unique()):
        subset = df.loc[df['day'] == day]
        n_sess = len(subset.session.unique())
        try:
            hist_ = stats.cumfreq(subset.current_time, numbins=max_timing,
                                    defaultreallimits=(0, subset.current_time.max()), weights=None)
        except:
            hist_ = stats.cumfreq(subset.current_time, numbins=max_timing, defaultreallimits=(0, max_timing),
                                    weights=None)
        hist_norm = hist_.cumcount / n_sess
        bins_plt = hist_.lowerlimit + np.linspace(0, hist_.binsize * hist_.cumcount.size, hist_.cumcount.size)
        sns.lineplot(x=bins_plt, y=hist_norm, color = sess_palette[idx % len(sess_palette)], ax=axes, marker='o', markersize=4)

    # axes.set_ylabel('Cum. nº of trials', label_kwargs)
    # axes.set_xlabel('Time (mins)', label_kwargs)

    # legend
    lines = [Line2D([0], [0], color=sess_palette[i], marker='o', markersize=7, markerfacecolor=sess_palette[i]) for
                i in
                range(len(sess_palette))]
    axes.legend(lines, np.arange(-5, 0, 1), title='Days', loc='center', bbox_to_anchor=(0.1, 0.85))
    plt.show()