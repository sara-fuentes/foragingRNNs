import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
main_folder = '/home/molano/Dropbox/Molabo/foragingRNNs/'
# load dataframe training_data.csv
folder = main_folder+'w0.01_mITI400_xITI800_f100_d100_prb0.2'
# folder = main_folder+'w1e-05_mITI400_xITI800_f100_d100_prb0.2' # associated to '_mITI_400_pr_0208'
# folder = main_folder+'w1e-05_mITI200_xITI400_f100_d100_prb0.1' # associated to ''
for exp in ['_w1e-02']: # , '_bias_corrected_th04']:  # , 'long_ITI_0208']:
    df = pd.read_csv(main_folder+'/training_data'+exp+'.csv')
    # get list of net seeds from df
    net_seeds = df['net_seed'].unique()
    env_seed = '123'
    f_training = plt.figure(figsize=(4, 4))
    # create df with networks stats
    df_net = pd.DataFrame(columns=['net_seed', 'seq_len', 'blk_dur', 'lr', 'mean_perf'])
    for ns in net_seeds:
        # build file name envS_XX_netS_XX
        f = folder+'/envS_'+env_seed+'_netS_'+str(ns)

        # get columns seq_len, blk_dur and lr corresponding to ns
        seq_len = df.loc[df['net_seed'] == ns, 'seq_len'].values[0]
        blk_dur = df.loc[df['net_seed'] == ns, 'blk_dur'].values[0]
        lr = df.loc[df['net_seed'] == ns, 'lr'].values[0]
        # load params.txt with parameters
        # params = np.loadtxt(f+'/params.txt')
        # load npz data
        
        try:
            data = np.load(f+'/data.npz')
            mean_final_perf = np.mean(data['mean_perf_list'][-100:])
            df_net = df_net.append({'net_seed': ns, 'seq_len': seq_len, 'blk_dur': blk_dur, 'lr': lr, 'mean_perf': mean_final_perf},
                                    ignore_index=True)
            if mean_final_perf>0.1:
                roll = 50
                mean_performance_smooth = np.convolve(data['mean_perf_list'],
                                              np.ones(roll)/roll, mode='valid')
                plt.plot(mean_performance_smooth, color='black', lw=0.5, alpha=0.5)
                plt.xlabel('Training epoch')
                plt.ylabel('Mean performance')
                # add dashed line at 0.25, 0.5 and 0.75
                plt.axhline(0.25, color='k', linestyle='--', alpha=0.5, lw=0.5)
                plt.axhline(0.5, color='k', linestyle='--', alpha=0.5, lw=0.5)
                plt.axhline(0.75, color='k', linestyle='--', alpha=0.5, lw=0.5)
                print(ns)
        except FileNotFoundError:
            print('File not found')

    df_net['param_combination'] = df_net['lr'].astype(str) + '_' + df_net['blk_dur'].astype(str) + '_' + df_net['seq_len'].astype(str)
    f_final_perf, ax = plt.subplots(figsize=(2, 4))
    sns.boxplot(data=df_net, x='param_combination', y='mean_perf', ax=ax)
    sns.stripplot(data=df_net, x='param_combination', y='mean_perf', color='black', size=4, ax=ax)
    # plot dashed line at 0.25, 0.5 and 0.75
    plt.axhline(0.25, color='k', linestyle='--')
    plt.axhline(0.5, color='k', linestyle='--')
    plt.axhline(0.75, color='k', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # xlabel
    plt.xlabel('Parameter configuration')
    # ylabel
    plt.ylabel('Mean performance')
    plt.show()
    # save figure
    f_final_perf.savefig(folder+'/perf_boxplot'+exp+'.png')
    f_final_perf.savefig(folder+'/perf_boxplot'+exp+'.svg')
    f_training.savefig(folder+'/training'+exp+'.png')
    f_training.savefig(folder+'/training'+exp+'.svg')
