import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader

from locs.utils.flags import build_flags
import locs.models.model_builder as model_builder
from locs.datasets.charged_data import ChargedData
import locs.training.train as train
import locs.training.train_utils as train_utils
import locs.training.evaluate as evaluate
import locs.utils.misc as misc


def plot_sample(model, dataset, num_samples, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1)
    use_gt_edges = params.get('use_gt_edges')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    batch_count = 0
    all_errors = []
    burn_in_steps = 10
    forward_pred_steps = 39

    colors = ['firebrick', 'forestgreen', 'dodgerblue', 'mediumvioletred', 'darkturquoise']
    pred_colors = ['lightsalmon', 'lightgreen', 'lightskyblue', 'palevioletred', 'lightskyblue']

    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        gt_edges = batch.get('edges', None)
        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            gt_predictions = inputs[:, burn_in_steps:burn_in_steps+forward_pred_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
                if gt_edges is not None and use_gt_edges:
                    gt_edges = gt_edges.cuda(non_blocking=True)
            if not use_gt_edges:
                gt_edges=None
            model_preds = model.predict_future(model_inputs, forward_pred_steps).cpu()
            #total_se += F.mse_loss(model_preds, gt_predictions).item()
            print("MSE: ", torch.nn.functional.mse_loss(model_preds, gt_predictions).item())
            batch_count += 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if isinstance(dataset, torch.utils.data.Subset):
            unnormalized_preds = dataset.dataset.unnormalize(model_preds)
            unnormalized_gt = dataset.dataset.unnormalize(inputs)
        else:
            unnormalized_preds = dataset.unnormalize(model_preds)
            unnormalized_gt = dataset.unnormalize(inputs)

        def update(frame):
            marker_range = 1.5 + 1.5 * np.arange(0, frame+1) / (frame+1)
            alpha_range = (0.1 + 0.9 * np.arange(0, frame+1) / (frame+1)) ** 2
            pred_range = (0.1 + 0.9 * np.arange(0, frame+1) / (frame+1)) ** 2 / 2
            ax.clear()
            for t in range(0, frame+1):
                for obj in range(5):
                    if t == frame:
                        ax.plot(*(unnormalized_gt[0, t, obj, :3]), 'o',
                                color=colors[obj], markersize=marker_range[t], alpha=alpha_range[t])

                    if t > 0:
                        ax.plot(*(unnormalized_gt[0, [t-1, t], obj, :3].T), '-',
                                color=colors[obj], alpha=alpha_range[t])
            if frame >= burn_in_steps:
                for t in range(burn_in_steps, frame+1):
                    tmp_fr = t - burn_in_steps
                    for obj in range(5):
                        if t == frame:
                            ax.plot(*(unnormalized_preds[0, tmp_fr, obj, :3]), 'h',
                                    color=pred_colors[obj], markersize=marker_range[t],
                                    alpha=pred_range[t])

                        if t > burn_in_steps:
                            ax.plot(*(unnormalized_preds[0, [tmp_fr-1, tmp_fr], obj, :3].T),
                                    '-', color=pred_colors[obj], alpha=alpha_range[t])
                        else:
                            first_pred = np.concatenate([unnormalized_gt[:, [t-1]], unnormalized_preds[:, [tmp_fr]]], 1)
                            ax.plot(*(first_pred[0, :, obj, :3].T), '-',
                                    color=pred_colors[obj], alpha=alpha_range[t])

            max_range = unnormalized_gt[0, :, :, :3].reshape(-1, 3).max(0)
            min_range = unnormalized_gt[0, :, :, :3].reshape(-1, 3).min(0)
            full_range = max_range - min_range
            ax.set_xlim(min_range[0] - 0.1 * full_range[0], max_range[0] + 0.1 * full_range[0])
            ax.set_ylim(min_range[1] - 0.1 * full_range[1], max_range[1] + 0.1 * full_range[1])
            ax.set_zlim(min_range[2] - 0.1 * full_range[2], max_range[2] + 0.1 * full_range[2])

        ani = animation.FuncAnimation(fig, update, interval=100, frames=burn_in_steps+forward_pred_steps)
        path = os.path.join(params['working_dir'], params["plot_sample_prefix"]
                            + 'pred_trajectory_%d.mp4'%batch_ind)
        ani.save(path, codec='mpeg4')
        plt.close(fig)


def save_samples(model, dataset, num_samples, params):
    gpu = params.get('gpu', False)
    batch_size = 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    burn_in_steps = params['test_burn_in_steps']
    forward_pred_steps = params['test_pred_steps']
    all_inputs = []
    all_outputs = []
    for batch in data_loader:
        inputs = batch['inputs']
        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
            model_preds = model.predict_future(model_inputs, forward_pred_steps).cpu()
        if isinstance(dataset, torch.utils.data.Subset):
            unnormalized_preds = dataset.dataset.unnormalize(model_preds)
            unnormalized_gt = dataset.dataset.unnormalize(inputs)
        else:
            unnormalized_preds = dataset.unnormalize(model_preds)
            unnormalized_gt = dataset.unnormalize(inputs)
        all_inputs.append(unnormalized_gt)
        all_outputs.append(unnormalized_preds)

    file_prefix = params['plot_sample_prefix']
    inputs_file_name = os.path.join(
        params['working_dir'], f'{file_prefix}all_inputs_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(inputs_file_name, all_inputs)

    outputs_file_name = os.path.join(
        params['working_dir'], f'{file_prefix}all_outputs_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(outputs_file_name, all_outputs)

    test_mse, _, _ = evaluate.eval_forward_prediction_unnormalized(
        model, dataset, burn_in_steps, forward_pred_steps, params,
        return_total_errors=True)
    error_file_name = os.path.join(params['working_dir'], f'{file_prefix}all_errors_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(error_file_name, test_mse.numpy())



if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--same_data_norm', action='store_true')
    parser.add_argument('--symmetric_data_norm', action='store_true')
    parser.add_argument('--vel_norm_norm', action='store_true')
    parser.add_argument('--no_data_norm', action='store_true')
    parser.add_argument('--error_out_name', default='{:s}prediction_errors_{:d}_{:d}_step.npy')
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--test_burn_in_steps', type=int, default=49)
    parser.add_argument('--test_pred_steps', type=int, default=20)
    parser.add_argument('--error_suffix')
    parser.add_argument('--val_teacher_forcing_steps', type=int, default=-1)
    parser.add_argument('--subject_ind', type=int, default=-1)
    parser.add_argument('--plot_samples', action='store_true')
    parser.add_argument('--report_error_norm', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--trans_only', action='store_true')
    parser.add_argument('--pos_representation', choices=['cart', 'polar'], default='polar')

    args = parser.parse_args()
    params = vars(args)

    misc.seed(args.seed)

    params['num_vars'] = 5
    params['input_size'] = 6
    params['input_time_steps'] = 49
    params['use_3d'] = True
    params['nll_loss_type'] = 'gaussian'
    params['plot_sample_prefix'] = 'inter_' if args.interactive else ""
    train_data = ChargedData(args.data_path, 'train', params)
    val_data = ChargedData(args.data_path, 'val', params)

    model = model_builder.build_model(params)
    if args.mode == 'train':
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer, val_writer)
    elif args.mode == 'eval':
        test_data = ChargedData(args.data_path, 'test', params)
        forward_pred = args.test_pred_steps

        if args.interactive:
            non_linear_mask = evaluate.get_non_linear_mask(test_data.torch_unnormalize(test_data.feats))
            interactive_dataset = torch.utils.data.Subset(
                test_data, non_linear_mask.nonzero().flatten())

            test_mse, test_pos_mse, test_vel_mse = evaluate.eval_forward_prediction_unnormalized(
                model, interactive_dataset, args.test_burn_in_steps, forward_pred, params,
                num_dims=3)
        else:
            test_mse, test_pos_mse, test_vel_mse = evaluate.eval_forward_prediction_unnormalized(
                model, test_data, args.test_burn_in_steps, forward_pred, params,
                num_dims=3)
        error_file_name = params['plot_sample_prefix'] + args.error_out_name.format(
            "norm_" if args.report_error_norm else "", args.test_burn_in_steps, args.test_pred_steps)
        print(error_file_name)
        path = os.path.join(args.working_dir, error_file_name)
        np.save(path, test_mse.cpu().numpy())
        pos_error_file_name = 'pos_' + error_file_name
        pos_path = os.path.join(args.working_dir, pos_error_file_name)
        np.save(pos_path, test_pos_mse.cpu().numpy())
        vel_error_file_name = 'vel_' + error_file_name
        vel_path = os.path.join(args.working_dir, vel_error_file_name)
        np.save(vel_path, test_vel_mse.cpu().numpy())
        test_mse_1 = test_mse[0].item()
        test_mse_10 = test_mse[9].item()
        test_mse_final = test_mse[-1].item()
        print("FORWARD PRED RESULTS:")
        print("\t1 STEP: ",test_mse_1)
        print("\t10 STEP: ",test_mse_10)
        print(f"\t{len(test_mse)} STEP: ",test_mse_final)

        print("POSITION FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_pos_mse[0].item())
        print("\t10 STEP: ", test_pos_mse[9].item())
        print(f"\t{len(test_mse)} STEP: ", test_pos_mse[-1].item())

        print("VELOCITY FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_vel_mse[0].item())
        print("\t10 STEP: ", test_vel_mse[9].item())
        print(f"\t{len(test_mse)} STEP: ", test_vel_mse[-1].item())

        if args.plot_samples:
            if args.interactive:
                plot_sample(model, interactive_dataset, args.test_burn_in_steps, params)
            else:
                plot_sample(model, test_data, args.test_burn_in_steps, params)
    elif args.mode == 'save_pred':
        test_data = ChargedData(args.data_path, 'test', params)
        if args.interactive:
            forward_pred = args.test_pred_steps

            non_linear_mask = evaluate.get_non_linear_mask(test_data.torch_unnormalize(test_data.feats))
            interactive_dataset = torch.utils.data.Subset(
                test_data, non_linear_mask.nonzero().flatten())
            save_samples(model, interactive_dataset, args.test_burn_in_steps, params)
        else:
            save_samples(model, test_data, args.test_burn_in_steps, params)
    elif args.mode == 'record_predictions':
        model.eval()
        burn_in = args.test_burn_in_steps
        forward_pred = params['input_time_steps'] - args.test_burn_in_steps
        test_data = ChargedData(args.data_path, 'test', params)
        if args.subject_ind == -1:
            val_data_loader = DataLoader(test_data, batch_size=params['batch_size'])
            all_predictions = []
            all_edges = []
            for batch_ind,batch in enumerate(val_data_loader):
                print("BATCH %d of %d"%(batch_ind+1, len(val_data_loader)))
                inputs = batch['inputs']
                if args.gpu:
                    inputs = inputs.cuda(non_blocking=True)
                with torch.no_grad():
                    predictions, edges = model.predict_future(inputs[:, :burn_in], forward_pred, return_edges=True, return_everything=True)
                    all_predictions.append(predictions)
                    all_edges.append(edges)
            if args.error_suffix is not None:
                out_path = os.path.join(args.working_dir, 'preds/', 'all_test_subjects_%s.npy'%args.error_suffix)
            else:
                out_path = os.path.join(args.working_dir, 'preds/', 'all_test_subjects.npy')

            predictions = torch.cat(all_predictions, dim=0)
            edges = torch.cat(all_edges, dim=0)

        else:
            data = test_data[args.subject_ind]
            inputs = data['inputs'].unsqueeze(0)
            if args.gpu:
                inputs = inputs.cuda(non_blocking=True)
            with torch.no_grad():
                predictions, edges = model.predict_future(inputs[:, :burn_in], forward_pred, return_edges=True, return_everything=True)
                predictions = predictions.squeeze(0)
                edges = edges.squeeze(0)
            out_path = os.path.join(args.working_dir, 'preds/', 'subject_%d.npy'%args.subject_ind)
        tmp_dir = os.path.join(args.working_dir, 'preds/')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        torch.save([predictions.cpu(), edges.cpu()], out_path)
