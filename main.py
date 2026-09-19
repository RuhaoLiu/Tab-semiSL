import argparse
import numpy as np
import os
import pandas as pd
import torch
from data_loader import load_mnist_data_torch
from supervised_models import xgb_model, mlp, lightgbm_model, catboost_model
from semi_adaptive import con_mask_semi
from utils import perf_metric
from vae_train import vae_train
from con_mask import con_mask, TsSelfModel


def supervised_model_training(x_train, y_train, x_test,
                              y_test, model_name, metric):
    """Train supervised learning models and report the results.

    Args:
      - x_train, y_train: training dataset
      - x_test, y_test: testing dataset
      - model_name: xgboost or catboost and so on
      - metric: acc or auc and so on

    Returns:
      - performance: prediction performance
    """

    # Train supervised model
    # XGBoost
    if model_name == 'xgboost':
        y_test_hat = xgb_model(x_train, y_train, x_test)
    elif model_name == 'lightgbm':
        y_test_hat = lightgbm_model(x_train, y_train, x_test)
    elif model_name == 'catboost':
        y_test_hat = catboost_model(x_train, y_train, x_test)
        # MLP
    elif model_name == 'mlp':
        mlp_parameters = dict()
        mlp_parameters['hidden_dim'] = 100
        mlp_parameters['epochs'] = 100
        mlp_parameters['activation'] = 'relu'
        mlp_parameters['batch_size'] = 100

        y_test_hat = mlp(x_train, y_train, x_test, mlp_parameters)

    # Report the performance
    performance = perf_metric(metric, y_test, y_test_hat)

    return performance


def ts_main(label_data_rate, model_sets, label_no, p_m, alpha, beta, gamma, datasets_name, seed):
    """Tab-semiSL Main function.

    Args:
      - label_data_rate: label data rate
      - model_sets: supervised model sets
      - label_no: amount of labeled data to be used
      - p_m: perturbation probability
      - alpha: hyper-parameter to control mask loss
      - beta: hyper-parameter to control contrastive loss
      - gamma: hyper-parameter to control unsupervised loss
      - datasets_name: name of datasets
      - seed: random seed

    Returns:
      - results: performances of supervised and Tan-semiSL performance
    """

    # Define outputs
    results = np.zeros([len(model_sets) + 1])

    # Define cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    x_train, y_train, x_unlab, x_test, y_test = load_mnist_data_torch(label_data_rate, seed)

    # Use subset of labeled data
    x_train = x_train[:label_no, :]
    y_train = y_train[:label_no, :]

    # Metric
    metric = 'acc'

    # Train supervised models
    for m_it in range(len(model_sets)):
        model_name = model_sets[m_it]
        results[m_it] = supervised_model_training(x_train, y_train, x_test,
                                                  y_test, model_name, metric)

    print('-----------Start training vae model-----------')

    # Generate augment data
    data_augment_parameters = dict()
    data_augment_parameters['hidden_dim'] = 400
    data_augment_parameters['latent_dim'] = 20
    data_augment_parameters['batch_size'] = 128
    data_augment_parameters['epochs'] = 50
    vae_model = vae_train(data_augment_parameters, x_unlab, datasets_name, device)

    print('-----------Save vae model-----------')

    # Vae model save dir
    vae_save_dir = f'./save_model/{args.datasets_name}/label_{args.label_no}/vae_model'
    vae_file = os.path.join(vae_save_dir, f'vae-encoder_model_{args.datasets_name}_{args.label_no}_{seed}.pth')

    # Save vae model
    os.makedirs(vae_save_dir, exist_ok=True)
    torch.save(vae_model.state_dict(), vae_file)

    # print('-----------Vae model saved-----------')
    print('-----------Start training self auto-encoder model-----------')

    # Train Tab-semiSL Self auto-encoder model
    ts_self_parameters = dict()
    ts_self_parameters['batch_size'] = 128
    ts_self_parameters['vae_latent_dim'] = 20
    ts_self_parameters['vae_hidden_dim'] = 400
    ts_self_parameters['epochs'] = 20
    ts_self_encoder = con_mask(vae_file, x_unlab, p_m, alpha, beta, ts_self_parameters, device)

    print('-----------Save self encoder model-----------')

    # Tab-semiSL Self auto-encoder model_save_dir
    cm_save_dir = f'./save_model/{args.datasets_name}/label_{args.label_no}/con_mask'
    encoder_file = os.path.join(cm_save_dir,
                                f'Self auto-encoder_model_{args.datasets_name}_{args.label_no}_{seed}.pth')

    # save Self auto-encoder model
    os.makedirs(cm_save_dir, exist_ok=True)
    torch.save(ts_self_encoder.state_dict(), encoder_file)

    print('-----------Self encoder model saved-----------')

    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    x_unlab = torch.tensor(x_unlab, dtype=torch.float32, device=device)

    print('-----------Start training predictor model-----------')

    # Train Tab-semiSL semi model
    ts_semi_parameters = dict()
    ts_semi_parameters['vae_hidden_dim'] = 400
    ts_semi_parameters['vae_latent_dim'] = 20
    ts_semi_parameters['predictor_hidden_dim'] = 512
    ts_semi_parameters['batch_size'] = 128
    ts_semi_parameters['iterations'] = 5000
    y_test_hat = con_mask_semi(x_train, y_train, x_unlab, x_test,
                               ts_semi_parameters, gamma, p_m, vae_file, encoder_file, device, args.datasets_name)

    print('-----------Training finished-----------')

    # Test Tab-semiSL semi model
    results[len(model_sets)] = perf_metric(metric, y_test, y_test_hat)
    print(np.round(results, 4))
    return results


def exp_main(args):
    """Main function for experiments.

    Args:
      - iterations: number of experiments iterations
      - label_no: amount of labeled data to be used
      - model_name: supervised model name (mlp, logit, or xgboost)
      - p_m: perturbation probability for self-supervised learning
      - alpha: hyper-parameter to control the mask loss
      - beta: hyperparameter to control the contrastive loss
      - gamma: hyperparameter to control the unsupervised loss
      - label_data_rate: ratio of labeled data
      - datasets_name: name of datasets
      - seed: random seed

    Returns:
      - results
    """
    # Define output
    results = np.zeros([args.iterations, 2])
    seed = 0
    # Iterations
    for it in range(args.iterations):
        print(f'-----------start {it} iteration-----------')
        seed += 10
        print(f'-----------seed {seed}-----------')
        results[it, :] = ts_main(args.label_data_rate,
                                 [args.model_name],
                                 args.label_no,
                                 args.p_m,
                                 args.alpha,
                                 args.beta,
                                 args.gamma,
                                 args.datasets_name,
                                 seed)

    # Print results
    print('Supervised Performance, Model Name: ' + args.model_name +
          ', Avg Perf: ' + str(np.round(np.mean(results[:, 0]), 4)) +
          ', Std Perf: ' + str(np.round(np.std(results[:, 0]), 4)))

    print('Tab-semiSL Performance' +
          ', Avg Perf: ' + str(np.round(np.mean(results[:, 1]), 4)) +
          ', Std Perf: ' + str(np.round(np.std(results[:, 1]), 4)))

    final_result = pd.DataFrame({
        'Model': [args.model_name, 'Tab-semiSL Performance'],
        'Avg_Perf': [np.mean(results[:, 0]), np.mean(results[:, 1])],
        'Std_Perf': [np.std(results[:, 0]), np.std(results[:, 1])]
    })

    output_save_dir = f'./output/{args.datasets_name}'
    os.makedirs(output_save_dir, exist_ok=True)
    output_file = os.path.join(output_save_dir, f'results_{args.datasets_name}_{args.label_no}.csv')
    final_result.to_csv(output_file, index=False)


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--iterations',
        help='number of experiments iterations',
        default=5,
        type=int)
    parser.add_argument(
        '--model_name',
        choices=['xgboost', 'catboost', 'lightgbm'],
        default='xgboost',
        type=str)
    parser.add_argument(
        '--label_no',
        help='amount of labeled data',
        default=500,
        type=int)
    parser.add_argument(
        '--p_m',
        help='perturbation probability for masked',
        default=0.5,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyper-parameter to control the mask loss',
        default=2.0,
        type=float)
    parser.add_argument(
        '--beta',
        help='hyper-parameter to control the contrastive loss',
        default=0.1,
        type=float)
    parser.add_argument(
        '--gamma',
        help='hyper-parameter to control the unsupervised loss',
        default=0.1,
        type=float)
    parser.add_argument(
        '--label_data_rate',
        help='ratio of labeled data',
        default=0.1,
        type=float)
    parser.add_argument(
        '--datasets_name',
        help='[MNIST,Income]',
        default='MNIST',
        type=str)

    args = parser.parse_args()

    # Calls main function
    results = exp_main(args)