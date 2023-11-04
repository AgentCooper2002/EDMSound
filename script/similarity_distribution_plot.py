import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

def load_sim_scores(gen_score_path, self_score_path=None):
    gen_score_list = []
    for path in gen_score_path:
        with open(path, 'rb') as file:
            gen_score_dict = pickle.load(file)
            gen_score_list.append(gen_score_dict)
    with open(self_score_path, 'rb') as file:
        self_score_dict = pickle.load(file)
    return {'gen_score': gen_score_list,
            'self_score': self_score_dict['sim'],
            'self_label': self_score_dict['label']}
    # return {'gen_score': gen_score_dict['sim'],
    #         'gen_label': gen_score_dict['label'],
    #         'self_score': self_score_dict['sim'],
    #         'self_label': self_score_dict['label']}
    


def plot_his_top_one_score(scores_dict, model_label, gen_models, axs, log_scale):
    bin_width= 0.005
    nbins = math.ceil(1 / bin_width)
    bins = np.linspace(0.001, 1, nbins)
    self_score = scores_dict['self_score']
    self_dataset_score = np.round(np.percentile(self_score, 95), 3)
    if log_scale:
        bins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))


    axs[0].hist(self_score, bins, alpha=1, label=f'Training dataset', density=True, histtype='step', linewidth=2, linestyle='dashed')
    for i, model in enumerate(gen_models):
        gen_score = scores_dict['gen_score'][i]['sim']
        print(len(gen_score), len(self_score))
        gen_dataset_score = np.round(np.percentile(gen_score, 95) - self_dataset_score, 3)

   
            
        axs[0].hist(gen_score, bins, alpha=1, label=f'{model}',density=True, histtype='step', linewidth=2)
    axs[0].set_xscale('log')  
    axs[0].set(xticks= [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
           xticklabels=['{:,.1f}'.format(x) for x in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]],
           xlim=[0.45, 1])
    axs[0].legend(loc='upper left')
    axs[0].set_yticks([])
    axs[0].set_xlabel('similarity score')
    axs[0].set_ylabel('distribution')

    
def plot_his_class_base(scores_dict, model_label, gen_models, class_dict, axs, log_scale):
    bin_width= 0.001
    nbins = math.ceil(1 / bin_width)
    bins = np.linspace(0.001, 1, nbins)
    if log_scale:
        bins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    for i in range(len(class_dict)):
        class_self_idx = np.where(scores_dict['self_label'] == i)[0]
        self_score = scores_dict['self_score']
        class_self_scores = self_score[class_self_idx]
        class_self_dataset_score = np.round(np.percentile(class_self_scores, 95), 3)
        axs[i+1].hist(class_self_scores, bins, alpha=0.6, label=f'Training dataset, {class_self_dataset_score}', density=True, histtype='step', linewidth=2, linestyle='dashed')
        for j, model in enumerate(gen_models):
            class_gen_idx = np.where(scores_dict['gen_score'][j]['label'] == i)[0]
            gen_score = scores_dict['gen_score'][j]['sim']
            class_gen_scores = gen_score[class_gen_idx]
            class_gen_dataset_score = np.round(np.percentile(class_gen_scores, 95) - class_self_dataset_score, 3)
        # print(len(class_self_scores), len(class_gen_scores), i)
        # axs[i+1].hist(class_gen_scores, bins, alpha=0.4, label=f'sim(gen,train), {class_dict[i]}, {model}', density=True)
        # axs[i+1].hist(class_self_scores, bins, alpha=0.6, label=f'sim(train,train), {class_dict[i]}, {model}', density=True)
        
            axs[i+1].hist(class_gen_scores, bins, alpha=0.4, label=f'{model}, {class_gen_dataset_score}', density=True, histtype='step', linewidth=2)
            # path = os.path.join(self.logger.save_dir, f'sim_histogram_{class_dict[i]}.pdf')
            # plt.savefig(path, dpi=500, bbox_inches='tight')
        axs[i+1].set_title(class_dict[i])
        axs[i+1].legend(loc='upper left')
        
        # axs[i+1].set_xlim(0.5, 1)
    # axs[i+1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        
def plot_sim_distribution(scores_dict, class_dict, model_label, gen_models, log_scale=True, class_base=True):
    font = {'size': 18}
    matplotlib.rc('font', **font)
    if class_base:
        fig, axs = plt.subplots(2, 4, sharex=True, tight_layout=True, figsize=(30, 12))
    else:
        fig, axs = plt.subplots(1, sharex=True, tight_layout=True, figsize=(10, 4))
    plot_his_top_one_score(scores_dict, model_label, gen_models, np.ravel(axs), log_scale)
    if class_base:
        plot_his_class_base(scores_dict, model_label, gen_models, class_dict, np.ravel(axs), log_scale)
    path = 'clap_ft_EDM.pdf'
    plt.savefig(path, dpi=500, bbox_inches='tight')
    

if __name__ == '__main__':
    class_dict = {0: 'DogBark', 1: 'Footstep', 2: 'GunShot', 3: 'Keyboard', 
                               4: 'MovingMotorVehicle', 5: 'Rain', 6: 'Sneeze_Cough'}
    # clap_ft_tri_0.2
    self_path = '/home/yutong/EDMsound/logs/eval/runs/clap_ft_tri_0.2_self/tensorboard/sim_score_dict.pickle'
    gen_path_our = '/home/yutong/EDMsound/logs/eval/runs/clap_ft_tri_0.2_gen_EDMSound/tensorboard/sim_score_dict.pickle'
    gen_path_TA = '/home/yutong/EDMsound/logs/eval/runs/audioMAE_ft_tri_0.2_gen_TA03/tensorboard/sim_score_dict.pickle'
    gen_path_TB = '/home/yutong/EDMsound/logs/eval/runs/audioMAE_ft_tri_0.2_gen_TB14/tensorboard/sim_score_dict.pickle'
    gen_path_TA08 = '/home/yutong/EDMsound/logs/eval/runs/audioMAE_ft_tri_0.2_gen_TA08/tensorboard/sim_score_dict.pickle'
    # gen_models = ['EDMSound', 'Scheibler et al', 'Jung et al', 'Yi et al']
    gen_models = ['EDMSound']
    # gen_path = [gen_path_our, gen_path_TA, gen_path_TB, gen_path_TA08]
    gen_path = [gen_path_our]
    score_dict = load_sim_scores(gen_path, self_path)
    plot_sim_distribution(score_dict, class_dict, 'audioMAE_ft_tri_0.2', gen_models, class_base=False)