"""
Written by George Zerveas
Modified by Chau Tran (in the thank of Hoa Nguyen's work)

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json
import math

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer

from datasets.dataset import collate_superv
from datasets.wisdmdata import WISDMDataset

def main(config):

    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run
    torch.manual_seed(config['seed']) if config['seed'] is not None else None
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    logger.info("Device index: {}".format(torch.cuda.current_device())) if device == 'cuda' else None
    
    logger.info("Loading and preprocessing data ...")
    data_class         = data_factory[config['data_class']]   #WISDMData
    config['data_dir'] = f"{config['data_dir']}/run{config['wisdm_file_no']}"
    
    # Train Data
    train_data    = data_class(config['data_dir'], pattern='TRAIN', n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    train_indices = train_data.all_IDs
    # Val Data (same with test data)
    val_data    = data_class(config['data_dir'], pattern='TEST', n_proc=-1, config=config)
    val_indices = val_data.all_IDs
    # Test Data
    test_data    = data_class(config['data_dir'], pattern='TEST', n_proc=-1, config=config)
    test_indices = test_data.all_IDs
    
    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validating".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))
    
    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices)),
                       'test_indices': list(map(int, test_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_indices),
                       'val_indices': list(val_indices),
                       'test_indices': list(test_indices)}, f, indent=4)
    
    # Pre-process features
    if config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])
        train_data.feature_df.loc[train_indices] = normalizer.normalize(train_data.feature_df.loc[train_indices])
        val_data.feature_df.loc[val_indices]     = normalizer.normalize(val_data.feature_df.loc[val_indices])
        test_data.feature_df.loc[test_indices]   = normalizer.normalize(test_data.feature_df.loc[test_indices])

    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, train_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))


    # Initialize optimizer
    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer   = optim_class(model.parameters(), lr=config['wisdm_lr0'], weight_decay=weight_decay, 
                              betas = (config['wisdm_beta1'], config['wisdm_beta2']), eps = config['wisdm_epsilon']) #adding for wisdm dataset

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr      = config['wisdm_lr0']  # initial learning rate (lr0)
    config["epochs"]   = int(math.floor(config['wisdm_numTrainingSteps'] / len(train_indices)))
    lr_T = int(config['wisdm_decayDuration']*float(config['wisdm_numTrainingSteps']/config['batch_size'])) # total steps
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['wisdm_lr0'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model.to(device)

    loss_module = get_loss_module(config)
    if config['test_only'] == 'testset':  # Only evaluate and skip training
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices)
        test_loader  = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                            print_interval=config['print_interval'], console=config['console'])
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
        print_str = 'Test Summary: '
        for k, v in aggr_metrics_test.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        return
    
    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    train_dataset = dataset_class(train_data, train_indices)  # ClassiregressionDataset
    print('class train_dataset: ', len(train_dataset))
    train_loader  = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              # drop_last =True,
                              collate_fn=lambda x: collate_superv(x, max_len=model.max_len))
    val_dataset = dataset_class(val_data, val_indices)
    val_loader  = DataLoader(dataset=val_dataset,
                              batch_size=config['batch_size'],
                              shuffle=False,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              # drop_last =True,
                              collate_fn=lambda x: collate_superv(x, max_len=model.max_len))
    
    trainer        = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                 print_interval=config['print_interval'], console=config['console'])
    val_evaluator  = runner_class(model, val_loader, device, loss_module,
                                 print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    logger.info('Starting training...')
    
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = f'Epoch {int(epoch)} Training Summary: '
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar(f'{k}/train', v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

        utils.save_model(os.path.join(config['save_dir'], 'model_{}_{}.pth'.format(mark, config['wisdm_file_no'])), epoch, model, optimizer)

        # Learning rate scheduling
        if config['wisdm_is_lr_scheduling']:
            if (epoch > lr_T):
                lr = config['wisdm_lrDecay'] * config['wisdm_lr0']
            else:
                lr = config['wisdm_lr0'] - (1.0-config['wisdm_lrDecay'])*config['wisdm_lr0']*epoch/lr_T
            logger.info('[WISDM ] Learning rate updated to: {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if epoch == config['lr_step'][lr_step]:
                utils.save_model(os.path.join(config['save_dir'], 'model_{}_{}.pth'.format(epoch, config['wisdm_file_no'])), epoch, model, optimizer)
                lr = lr * config['lr_factor'][lr_step]
                if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                    lr_step += 1
                logger.info('Learning rate updated to: {}'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + str(config["wisdm_file_no"]) + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"], best_metrics, aggr_metrics_val, comment=config['comment'], file_no=str(config['wisdm_file_no']))

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    return best_value, best_metrics

if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    results_record = []
    data_dir = config['data_dir']
    for file_no in range(1,31):
        result = {}
        config['wisdm_file_no']  = file_no
        config['data_dir']       = data_dir
        best_value, best_metrics = main(config)
        result['best_value']     = best_value
        result['best_metrics']   = best_metrics
        results_record.append(result)
    
    print(results_record)