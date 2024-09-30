import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM, TEMPO, ST_TimeLLM_1, ST_TimeLLM_2

from data_provider.data_factory_tempo import data_provider
import time
import random
import numpy as np
import os
from omegaconf import OmegaConf
import seaborn as sns
from numpy.random import choice
from torch.utils.data import Subset
import sys
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config

# basic config
parser.add_argument('--task_name', type=str,  default='long_term_forecast',
                                help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int,  default=1, help='status')
parser.add_argument('--model_id', type=str,  default='test', help='model id')
parser.add_argument('--model_comment', type=str,  default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str,  default='TEMPO',
                            help='model name, options: [Autoformer, DLinear]')
# model define

parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--datasets', type=str, default='ETTh1')
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
parser.add_argument('--target_data', type=str, default='ETTh1')
parser.add_argument('--root_path',type=str,default='./dataset/ETT-small',help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str,default='M',help = 'forecasting task, options:[M, S, MS]; '
                           'M:multivariate predict multivariate, S: univariate predict univariate, '
                           'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',help='freq for time features encoding, '
                            'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                            'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Hourly', help='subset for M4')
# model define
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=768, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=1, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='COS', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--llm_layers', type=int, default=1)
parser.add_argument('--percent', type=int, default=2)
parser.add_argument('--output_attn_map', action='store_true',
                    help='used for output attention map of patches and prototype tokens')#output_attn_map
parser.add_argument('--align_text', action='store_true', help='align trext or not')
parser.add_argument('--combination', type=str, default='late', help='combine components before go into model or not')
parser.add_argument('--decomp_level', type=int, default=1, help='decomposition level, '
                                                                '1 for TimeLLM, 2 for trend and seasonal, 3 for trend, seasonal and residual')
parser.add_argument('--decomp_method', type=str, default='STL', help='decomposition method for level as 3, '
                                                                       'STL change the original dataloader, TEMPO changes the model architecture')
parser.add_argument('--config_path', type=str, default='./configs/multiple_datasets.yml')
parser.add_argument('--electri_multiplier', type=int, default=1)
parser.add_argument('--traffic_multiplier', type=int, default=1)
parser.add_argument('--equal', type=int, default=1, help='1: equal sampling, 0: dont do the equal sampling')
args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

mses = []
maes = []
config = get_init_config(args.config_path)
accelerator.print(args)
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.target_data,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.d_ff, ii)

    train_data_name = args.datasets.split(',')
    print(train_data_name)
    train_datas = []
    val_datas = []
    min_sample_num = sys.maxsize
    for dataset_singe in args.datasets.split(','):
        print(dataset_singe)
        args.data = config['datasets'][dataset_singe].data
        args.root_path = config['datasets'][dataset_singe].root_path
        args.data_path = config['datasets'][dataset_singe].data_path
        args.data_name = config['datasets'][dataset_singe].data_name
        args.features = config['datasets'][dataset_singe].features
        args.freq = config['datasets'][dataset_singe].freq
        args.target = config['datasets'][dataset_singe].target
        if args.freq == 0:
            args.freq = 'h'

        print("dataset: ", args.data)
        train_data, train_loader = data_provider(args, 'train')
        if dataset_singe not in ['ETTh1', 'ETTh2', 'ILI', 'exchange']:
            min_sample_num = min(min_sample_num, len(train_data))

        # args.percent = 20
        vali_data, vali_loader = data_provider(args, 'val')
        # args.percent = 100

        # train_datas.append(train_data)
        val_datas.append(vali_data)

    for dataset_singe in args.datasets.split(','):
        print(dataset_singe)
        args.data = config['datasets'][dataset_singe].data
        args.root_path = config['datasets'][dataset_singe].root_path
        args.data_path = config['datasets'][dataset_singe].data_path
        args.data_name = config['datasets'][dataset_singe].data_name
        args.features = config['datasets'][dataset_singe].features
        args.freq = config['datasets'][dataset_singe].freq
        args.target = config['datasets'][dataset_singe].target
        if args.freq == 0:
            args.freq = 'h'

        print("dataset: ", args.data)
        train_data, train_loader = data_provider(args, 'train')
        if dataset_singe not in ['ETTh1', 'ETTh2', 'ILI', 'exchange'] and args.equal == 1:
            train_data = Subset(train_data, choice(len(train_data), min_sample_num))
        if args.electri_multiplier > 1 and args.equal == 1 and dataset_singe in ['electricity']:
            train_data = Subset(train_data, choice(len(train_data), int(min_sample_num * args.electri_multiplier)))
        if args.traffic_multiplier > 1 and args.equal == 1 and dataset_singe in ['traffic']:
            train_data = Subset(train_data, choice(len(train_data), int(min_sample_num * args.traffic_multiplier)))
        train_datas.append(train_data)

    if len(train_datas) > 1:
        train_data = torch.utils.data.ConcatDataset([train_datas[0], train_datas[1]])
        vali_data = torch.utils.data.ConcatDataset([val_datas[0], val_datas[1]])
        for i in range(2, len(train_datas)):
            train_data = torch.utils.data.ConcatDataset([train_data, train_datas[i]])

            vali_data = torch.utils.data.ConcatDataset([vali_data, val_datas[i]])

        # import pdb; pdb.set_trace()
        print("Way1", len(train_data))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    vali_loader = torch.utils.data.DataLoader(vali_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)

    args.data = config['datasets'][args.target_data].data
    args.root_path = config['datasets'][args.target_data].root_path
    args.data_path = config['datasets'][args.target_data].data_path
    args.data_name = config['datasets'][args.target_data].data_name
    args.features = config['datasets'][dataset_singe].features
    args.freq = config['datasets'][args.target_data].freq
    args.target = config['datasets'][args.target_data].target
    if args.freq == 0:
        args.freq = 'h'
    test_data, test_loader = data_provider(args, 'test')

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'ST_TimeLLM_1' or args.model == 'TimeLLM':
        model = ST_TimeLLM_1.Model(args).float()
    elif args.model == 'ST_TimeLLM_2':
        model = ST_TimeLLM_2.Model(args).float()
    elif args.model == 'ST_TimeLLM_3':
        model = ST_TimeLLM_3.Model(args).float()
    elif args.model == 'TEMPO':
        model = TEMPO.TEMPO(args).float()

    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    os.makedirs(path, exist_ok=True)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)


    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, seq_trend, seq_seasonal, seq_resid)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            accelerator.backward(loss)
            model_optim.step()

            scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator.device, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae_loss = vali(args, accelerator.device, model, test_data, test_loader, criterion, mae_metric)
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test MSE: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break
        
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        del_files(path)  # delete checkpoint files
        accelerator.print('success delete checkpoints')
