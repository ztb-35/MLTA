import argparse
from multiprocessing import freeze_support
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from models import Autoformer, DLinear, TimeLLM, ST_TimeLLM_1, ST_TimeLLM_2
from data_provider.data_factory_tempo import data_provider
import time
import random
import numpy as np
import os
import seaborn as sns
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
from utils.tools import EarlyStopping, adjust_learning_rate, vali, test, del_files

parser = argparse.ArgumentParser(description='Time-LLM')
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

#basic config
def main():
    parser.add_argument('--task_name', type=str,  default='long_term_forecast',
                                help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model_id', type=str,  default='test', help='model id')
    parser.add_argument('--model_comment', type=str,  default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str,  default='',
                                help='model name, options: [Autoformer, DLinear]')
    # model define

    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
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
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=1, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=1)
    parser.add_argument('--percent', type=int, default=1)
    parser.add_argument('--output_attn_map', action='store_true',
                        help='used for output attention map of patches and prototype tokens')#output_attn_map
    parser.add_argument('--decomp_level', type=int, default=3, help='decomposition level, '
                                                                    '1 for TimeLLM, 2 for trend and seasonal, 3 for trend, seasonal and residual')
    parser.add_argument('--decomp_method', type=str, default='STL', help='decomposition method for level as 3, '
                                                                           'STL change the original dataloader, TEMPO changes the model architecture')
    args = parser.parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    #deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    mses = []
    maes = []
    accelerator.print(args)
    for ii in range(args.itr):
        #setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_level{}_{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.decomp_level,
            args.decomp_method,
            args.des, ii)

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args,'val')
        test_data, test_loader = data_provider(args, 'test')
        attn_test_data, attn_test_loader = data_provider(args, 'test')

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        else:
            model = ST_TimeLLM_1.Model(args).float()
        path = os.path.join(args.checkpoints,setting + '-' + args.model_comment)  # unique checkpoint saving path#too long
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
                train_loader, vali_loader, test_loader, model, model_optim, scheduler)
            # train_loader, vali_loader, test_loader, model, model_optim = accelerator.prepare(
            #     train_loader, vali_loader, test_loader, model, model_optim)


        device = accelerator.device

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        best_score = 100
        counter = 0
        delta = 0.0001
        if not args.output_attn_map:
            for epoch in range(args.train_epochs):
                iter_count = 0
                train_loss = []
                model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in tqdm(
                        enumerate(train_loader), total=len(train_loader)):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                    seq_trend = seq_trend.float().to(device)
                    seq_seasonal = seq_seasonal.float().to(device)
                    seq_resid = seq_resid.float().to(device)
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                    # encoder - decoder

                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, seq_trend, seq_seasonal, seq_resid)
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:,f_dim:]
                    batch_y = batch_y[:, -args.pred_len:,f_dim:]
                    loss = criterion(outputs,batch_y)
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        accelerator.print( "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                                     loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                        accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed,left_time))
                        iter_count = 0
                        time_now = time.time()

                    if args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        accelerator.backward(loss)
                        model_optim.step()

                    if args.lradj == 'TST':
                        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args,
                                         printout=False)
                        scheduler.step()
                    # 获取当前学习率
                    current_lr = scheduler.get_last_lr()[0]
                    accelerator.print(f"Epoch: {epoch}, Step: {i}, Learning Rate: {current_lr}")

                accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion,mae_metric)
                test_loss, test_mae_loss = vali(args, accelerator,model, test_data,test_loader,criterion,mae_metric)
                accelerator.print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} MSE: {1:.7f} MAE: {2:.7f}".format
                                  (epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))  # loss=MSE,mae_loss=MAE
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

                if epoch in (1, 2, 5, 10):
                    if accelerator is not None:
                        model = accelerator.unwrap_model(model)
                        torch.save(model.state_dict(), path + '/' + '_epoch_' + str(epoch) + 'checkpoint')
                    else:
                        torch.save(model.state_dict(), path + '/' + '_epoch_' + str(epoch) + 'checkpoint')

                # score = vali_loss
                # if score > best_score + delta:
                #     counter += 1
                #     if accelerator is None:
                #         print(f'EarlyStopping counter: {counter} out of {args.patience}')
                #     else:
                #         accelerator.print(f'EarlyStopping counter: {counter} out of {args.patience}')
                #     if counter >= args.patience:
                #         model = accelerator.unwrap_model(model)
                #         torch.save(model.state_dict(), path + '/' + '_epoch_' + str(epoch) + 'checkpoint')
                #         break
                # else:
                #     best_score = score
                #     counter = 0
                early_stopping(vali_loss, model, path)
                if early_stopping.early_stop:
                    accelerator.print("Early stopping")
                    break

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        else:
            model = ST_TimeLLM_1.Model(args).float()

        if not args.output_attn_map:
            #Load the state_dict
            state_dict = torch.load(path + '/' + 'checkpoint')

            # Adjust the keys by removing 'module.' prefix
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model = accelerator.prepare(model)
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            accelerator.print("Best model: MSE: {0:.7f} MAE: {1:.7f}".format(test_loss, test_mae_loss))  # loss=MSE,mae_loss=MAE
            mses.append(test_loss)
            maes.append(test_mae_loss)

        if args.output_attn_map:
            for epoch in (1, 2, 5, 10, 100):
                #here epoch = 100 means the final svaed model(best model)
                if epoch != 100:
                    state_dict = torch.load(path + '/' + '_epoch_' + str(epoch) + 'checkpoint')
                else:
                    state_dict = torch.load(path + '/' + 'checkpoint')
                # Adjust the keys by removing 'module.' prefix
                new_state_dict = {'module.'+k: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
                model, test_loader = accelerator.prepare(model, attn_test_loader)
                model.eval()
                batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid = next(iter(test_loader))
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                seq_trend = seq_trend.float().to(device)
                seq_seasonal = seq_seasonal.float().to(device)
                seq_resid = seq_resid.float().to(device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                first_sample = batch_x[0, :,0]  # Shape will be (length,)
                # Plot the time series
                plt.figure(figsize=(10, 6))
                plt.plot(first_sample.cpu().detach().numpy(), label='Time Series Sample')
                plt.title('Time Series Data - First Sample')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                output_path = path+'/input.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')

                _, attn_map_list = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, seq_trend, seq_seasonal, seq_resid)
                if args.decomp_level == 1:
                    num_attn_map = {'attn_original': attn_map_list[0]}
                elif args.decomp_level == 2:
                    num_attn_map = {'attn_seasonal': attn_map_list[0], 'attn_trend': attn_map_list[1]}
                elif args.decomp_level == 3:
                    num_attn_map = {'attn_seasonal': attn_map_list[0], 'attn_trend': attn_map_list[1], 'attn_residual': attn_map_list[2]}

                for k, v in num_attn_map.items():
                    accelerator.print("v in num_attn_map.items(): ", str(k))
                    attn_heads_fused = v.mean(axis=1).mean(axis=0)
                    # Create a heatmap
                    plt.figure(figsize=(4, 20))  # Increase figure width to better display all columns
                    sns.heatmap(attn_heads_fused.cpu().detach().numpy(), cmap='viridis', linewidths=0)# Adjust the aspect ratio by setting the aspect of the Axes object
                    plt.gca().set_aspect('auto')
                    # Save the plot to a local path
                    output_path = path + '/' + str(k) + '_heatmap_epoch_' + str(epoch) + '.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
            break

    accelerator.print("predict length: ", args.pred_len)
    accelerator.print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
    accelerator.print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

if __name__ == '__main__':
    freeze_support()
    main()
