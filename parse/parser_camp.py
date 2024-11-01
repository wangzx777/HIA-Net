# coding=utf-8
import os
import argparse
from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

dataset_name = 'SEED'
model_name = 'CAMP'

# hyper params default setting
params = {
    'dataset_name': dataset_name,
    'experiment_root': '../result_sdafsl/' + dataset_name + '/' + model_name + '/' + current_time,  # output result root path
    'n_epochs': 50,  # training epochs, samplecnn: 100(200 for SEED), tcn: 6
    'learning_rate': 1e-4,  # SimpleCNN: 0.001, TCN: 2e-4 mudafn:0.01 sudafn:0.01, ResCBAM:0.001 ResCBAM-att:0.0001, SUDAFN-ResCBAM: 0.001, SDA-FSL:0.001, MDA-FSL:0.001
    'lr_scheduler_step': 10,
    'lr_scheduler_gamma': 0.5,
    'n_episodes': 20,  # number of episodes/iterations in an epoch
    'n_classes_per_episode_src': 3,  # number of classes used in an episode of source
    'n_supports_per_class_src': 5,  # numbers of support samples of each class of source
    'n_querys_per_class_src': 20,  # CAN BE MODIFIED. number of query samples of each class of source
    'n_classes_per_episode_tgt': 3,  # number of classes used in an episode of target
    'n_supports_per_class_tgt': 5,  # number of support samples of each class of target
    'n_querys_per_class_tgt': 20,  # CAN BE MODIFIED.number of query samples of each class of target
    'manual_seed': 42,
    'cuda': 0,  # cuda ID
    'optim': 'Adam',  # which optimizer to use general:Adam mudafn:SGD sudafn:SGD FN:Adam
    'patience': 10,  # patience for early stopping
    'num_bands': 5,  # number of EEG bands, DEAP:4, SEED:5
    'num_sources': 12,  # number of source domains, total subject number of DEAP/SEED:32/15
    'num_cal': 120, # num_cal = (num_support + num_query) * num_classes
    'SDA':False
}



def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-encoder', '--encoder',
    #                     type=str,
    #                     help='which encoder you use to embed EEG',
    #                     default=params['encoder_name'])
    parser.add_argument('-dataset', '--dataset',
                        type=str,
                        help='dataset name',
                        default=params['dataset_name'])
    # parser.add_argument('-root', '--dataset_root',
    #                     type=str,
    #                     help='path to dataset',
    #                     default=params['dataset_root'])

    parser.add_argument('-cSrc', '--classes_per_it_src',
                        type=int,
                        help='number of random classes per episode of source',
                        default=params['n_classes_per_episode_src'])

    parser.add_argument('-nsSrc', '--num_support_src',
                        type=int,
                        help='number of samples per class to use as support of source',
                        default=params['n_supports_per_class_src'])

    parser.add_argument('-nqSrc', '--num_query_src',
                        type=int,
                        help='number of samples per class to use as query of source',
                        default=params['n_querys_per_class_src'])

    parser.add_argument('-cTgt', '--classes_per_it_tgt',
                        type=int,
                        help='number of random classes per episode of target',
                        default=params['n_classes_per_episode_tgt'])

    parser.add_argument('-nsTgt', '--num_support_tgt',
                        type=int,
                        help='number of samples per class to use as support of target',
                        default=params['n_supports_per_class_tgt'])

    parser.add_argument('-nqTgt', '--num_query_tgt',
                        type=int,
                        help='number of samples per class to use as query of target',
                        default=params['n_querys_per_class_tgt'])

    # parser.add_argument('-kfold', '--k_fold',
    #                     type=int,
    #                     help='k-fold (session) cross validation',
    #                     default=params['k_fold'])
    parser.add_argument('--cuda',
                        type=int,
                        help='enables cuda',
                        default=params['cuda'])

    parser.add_argument('--seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=params['manual_seed'])
    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=params['n_episodes'])


    parser.add_argument('-optim', '--optim',
                        type=str,
                        help='which optimizer to use',
                        default=params['optim'])
    parser.add_argument('-patience', '--patience',
                        type=str,
                        help='patience for early stopping',
                        default=params['patience'])

    parser.add_argument('-num_bands', '--num_bands',
                        type=int,
                        help='number of EEG bands',
                        default=params['num_bands'])
    parser.add_argument('-num_sources', '--num_sources',
                        type=int,
                        help='number of source domains',
                        default=params['num_sources'])
    parser.add_argument('-num_cal', '--num_cal',
                        type=int,
                        help='number of calibration sample ',
                        default=params['num_cal'])

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default=params['experiment_root'])
    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=params['n_epochs'])

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=params['learning_rate'])

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=params['lr_scheduler_step'])

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=params['lr_scheduler_gamma'])
    parser.add_argument('--SDA',
                        type=bool,
                        help='whether use target CE loss',
                        default=params['SDA'])




    # elif mode=='finetune':
    #     # parser.add_argument('--split', default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
    #     parser.add_argument('--model_root', default=params['model_root'], type=str, help='the path of the pretrained models')
    #     parser.add_argument('--experiment_root', default=params['experiment_root'], type=str, help='the path of the finetune models')
    #     parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
    #     parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')
    #     parser.add_argument('--proto_init', action='store_true', help='initialize the adapted classifier via distances to prototypes')
    #     parser.add_argument('--lr_rate', default=0.0001, type=float,help ='Learning rate for fine-tuning')
    #     parser.add_argument('--ft_steps', default=100, type=int,help ='Number of fine-tuning steps')
    #     parser.add_argument('--freeze_backbone', action='store_true', help='Freeze the backbone network for finetuning')

    return parser

# def save_params(options):
#     opt = vars(options)
#     options.experiment_root = options.experiment_root + '_nCal' + str(options.num_cal) + '_seed' + str(options.manual_seed)
#     if options.message != None:
#         options.experiment_root = options.experiment_root + '_' + options.message
#
#     options.num_support_src = options.num_query_src = \
#         options.num_support_tgt = options.num_query_tgt = \
#         options.num_cal // 6
#
#     if not os.path.exists(options.experiment_root):
#         os.makedirs(options.experiment_root)
#
#     with open(os.path.join(options.experiment_root, 'hyper_params_setting.txt'), 'w') as f:
#         for (param_key, param_value) in opt.items():
#             f.write(param_key + ': ' + str(param_value) + '\n')# coding=utf-8
