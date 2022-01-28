import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch.utils.data import DataLoader

from locs.utils.flags import build_flags
import locs.models.model_builder as model_builder
from locs.datasets.small_synth_data import SmallSynthData
import locs.training.train as train
import locs.training.train_utils as train_utils
import locs.training.evaluate as evaluate
import locs.utils.misc as misc


def eval_edges(model, dataset, params):

    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    eval_metric = params.get('eval_metric')
    num_edge_types = params['num_edge_types']
    skip_first = params['skip_first']
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=gpu)
    full_edge_count = 0.
    model.eval()
    correct_edges = 0.
    edge_count = 0.
    correct_0_edges = 0.
    edge_0_count = 0.
    correct_1_edges = 0.
    edge_1_count = 0.

    correct = num_predicted = num_gt = 0
    all_edges = []
    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        gt_edges = batch['edges'].long()
        with torch.no_grad():
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
                gt_edges = gt_edges.cuda(non_blocking=True)

            _, _, _, edges, _ = model.calculate_loss(inputs, is_train=False, return_logits=True)
            edges = edges.argmax(dim=-1)
            all_edges.append(edges.cpu())
            if len(edges.shape) == 3 and len(gt_edges.shape) == 2:
                gt_edges = gt_edges.unsqueeze(1).expand(gt_edges.size(0), edges.size(1), gt_edges.size(1))
            elif len(gt_edges.shape) == 3 and len(edges.shape) == 2:
                edges = edges.unsqueeze(1).expand(edges.size(0), gt_edges.size(1), edges.size(1))
            if edges.size(1) == gt_edges.size(1) - 1:
                gt_edges = gt_edges[:, :-1]
            edge_count += edges.numel()
            full_edge_count += gt_edges.numel()
            correct_edges += ((edges == gt_edges)).sum().item()
            edge_0_count += (gt_edges == 0).sum().item()
            edge_1_count += (gt_edges == 1).sum().item()
            correct_0_edges += ((edges == gt_edges)*(gt_edges == 0)).sum().item()
            correct_1_edges += ((edges == gt_edges)*(gt_edges == 1)).sum().item()
            correct += (edges*gt_edges).sum().item()
            num_predicted += edges.sum().item()
            num_gt += gt_edges.sum().item()
    prec = correct / (num_predicted + 1e-8)
    rec = correct / (num_gt + 1e-8)
    f1 = 2*prec*rec / (prec+rec+1e-6)
    all_edges = torch.cat(all_edges)
    return f1, prec, rec, correct_edges / (full_edge_count + 1e-8), correct_0_edges / (edge_0_count + 1e-8), correct_1_edges / (edge_1_count + 1e-8), all_edges


def plot_sample(model, dataset, num_samples, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1)
    use_gt_edges = params.get('use_gt_edges')
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    batch_count = 0
    all_errors = []
    burn_in_steps = 10
    forward_pred_steps = 40

    colors = ['firebrick', 'forestgreen', 'dodgerblue', 'mediumvioletred', 'darkturquoise']
    pred_colors = ['lightsalmon', 'lightgreen', 'lightskyblue', 'palevioletred', 'lightskyblue']

    total_steps = burn_in_steps + forward_pred_steps
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
        fig, ax = plt.subplots()
        unnormalized_preds = dataset.unnormalize(model_preds)
        unnormalized_gt = dataset.unnormalize(inputs)
        def update(frame):
            marker_range = 1.5 + 1.5 * np.arange(0, frame+1) / (frame+1)
            alpha_range = (0.1 + 0.9 * np.arange(0, frame+1) / (frame+1)) ** 2
            pred_range = (0.1 + 0.9 * np.arange(0, frame+1) / (frame+1)) ** 2 / 2
            ax.clear()
            for t in range(0, frame+1):
                for obj in range(3):
                    if t == frame:
                        ax.plot(*(unnormalized_gt[0, t, obj, :2]), 'o',
                                color=colors[obj], markersize=marker_range[t], alpha=alpha_range[t])

                    if t > 0:
                        ax.plot(*(unnormalized_gt[0, [t-1, t], obj, :2].T), '-',
                                color=colors[obj], alpha=alpha_range[t])
            if frame >= burn_in_steps:
                for t in range(burn_in_steps, frame+1):
                    tmp_fr = t - burn_in_steps
                    for obj in range(3):
                        if t == frame:
                            ax.plot(*(unnormalized_preds[0, tmp_fr, obj, :2]), 'h',
                                    color=pred_colors[obj], markersize=marker_range[t],
                                    alpha=pred_range[t])

                        if t > burn_in_steps:
                            ax.plot(*(unnormalized_preds[0, [tmp_fr-1, tmp_fr], obj, :2].T),
                                    '-', color=pred_colors[obj], alpha=alpha_range[t])
                        else:
                            first_pred = np.concatenate([unnormalized_gt[:, [t-1]], unnormalized_preds[:, [tmp_fr]]], 1)
                            ax.plot(*(first_pred[0, :, obj, :2].T), '-',
                                    color=pred_colors[obj], alpha=alpha_range[t])

            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
        ani = animation.FuncAnimation(fig, update, interval=100, frames=burn_in_steps+forward_pred_steps)
        path = os.path.join(params['working_dir'], 'pred_trajectory_%d.mp4'%batch_ind)
        ani.save(path, codec='mpeg4')
        plt.close(fig)

        fig2, ax2 = plt.subplots()
        ax2.set_xlim(-6, 6)
        ax2.set_ylim(-6, 6)
        markersizes = np.linspace(2, 5, burn_in_steps+forward_pred_steps)
        for frame in range(burn_in_steps+forward_pred_steps):
            ax2.plot(unnormalized_gt[0, frame, 0, 0], unnormalized_gt[0, frame, 0, 1], '-bo', markersize=markersizes[frame])
            ax2.plot(unnormalized_gt[0, frame, 1, 0], unnormalized_gt[0, frame, 1, 1], '-ro', markersize=markersizes[frame])
            ax2.plot(unnormalized_gt[0, frame, 2, 0], unnormalized_gt[0, frame, 2, 1], '-go', markersize=markersizes[frame])
            if frame >= burn_in_steps:
                tmp_fr = frame - burn_in_steps
                ax2.plot(unnormalized_preds[0, tmp_fr, 0, 0], unnormalized_preds[0, tmp_fr, 0, 1], '-bo', alpha=0.5, markersize=markersizes[frame])
                ax2.plot(unnormalized_preds[0, tmp_fr, 1, 0], unnormalized_preds[0, tmp_fr, 1, 1], '-ro', alpha=0.5, markersize=markersizes[frame])
                ax2.plot(unnormalized_preds[0, tmp_fr, 2, 0], unnormalized_preds[0, tmp_fr, 2, 1], '-go', alpha=0.5, markersize=markersizes[frame])
        path = os.path.join(params['working_dir'], 'pred_trajectory_%d.jpg'%batch_ind)
        fig2.savefig(path)
        plt.close(fig2)

        if batch_count >= num_samples:
            break


def save_samples(model, dataset, num_samples, params):
    gpu = params.get('gpu', False)
    batch_size = 1
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    burn_in_steps = params.get('test_burn_in_steps', 25)
    forward_pred_steps = 50 - burn_in_steps
    all_inputs = []
    all_outputs = []
    for batch in data_loader:
        inputs = batch['inputs']
        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
            model_preds = model.predict_future(model_inputs, forward_pred_steps).cpu()
        unnormalized_preds = dataset.unnormalize(model_preds)
        unnormalized_gt = dataset.unnormalize(inputs)
        all_inputs.append(unnormalized_gt)
        all_outputs.append(unnormalized_preds)
    inputs_file_name = os.path.join(
        params['working_dir'], f'all_inputs_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(inputs_file_name, all_inputs)
    outputs_file_name = os.path.join(
        params['working_dir'], f'all_outputs_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(outputs_file_name, all_outputs)

    test_mse, _, _ = evaluate.eval_forward_prediction_unnormalized(
        model, test_data, burn_in_steps, forward_pred_steps, params,
        return_total_errors=True)
    error_file_name = os.path.join(params['working_dir'], f'all_errors_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(error_file_name, test_mse.numpy())


if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--same_data_norm', action='store_true')
    parser.add_argument('--symmetric_data_norm', action='store_true')
    parser.add_argument('--vel_norm_norm', action='store_true')
    parser.add_argument('--no_data_norm', action='store_true')
    parser.add_argument('--error_out_name', default='%sprediction_errors_%dstep.npy')
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--test_burn_in_steps', type=int, default=10)
    parser.add_argument('--error_suffix')
    parser.add_argument('--val_teacher_forcing_steps', type=int, default=-1)
    parser.add_argument('--subject_ind', type=int, default=-1)
    parser.add_argument('--report_error_norm', action='store_true')
    parser.add_argument('--plot_samples', action='store_true')
    parser.add_argument('--trans_only', action='store_true')
    parser.add_argument('--pos_representation', choices=['cart', 'polar'], default='polar')

    args = parser.parse_args()
    params = vars(args)

    misc.seed(args.seed)

    params['num_vars'] = 3
    params['input_size'] = 4
    params['input_time_steps'] = 50
    params['nll_loss_type'] = 'gaussian'
    train_data = SmallSynthData(args.data_path, 'train', params)
    val_data = SmallSynthData(args.data_path, 'val', params)

    model = model_builder.build_model(params)
    if args.mode == 'train':
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer, val_writer)
    elif args.mode == 'eval':
        test_data = SmallSynthData(args.data_path, 'test', params)
        forward_pred = 50 - args.test_burn_in_steps
        test_mse, test_pos_mse, test_vel_mse = evaluate.eval_forward_prediction_unnormalized(model, test_data, args.test_burn_in_steps, forward_pred, params)
        error_file_name = args.error_out_name%("norm_" if args.report_error_norm
                                               else "", args.test_burn_in_steps)
        path = os.path.join(args.working_dir, error_file_name)
        np.save(path, test_mse.cpu().numpy())
        pos_error_file_name = 'pos_' + error_file_name
        pos_path = os.path.join(args.working_dir, pos_error_file_name)
        np.save(pos_path, test_pos_mse.cpu().numpy())
        vel_path = os.path.join(args.working_dir, 'vel_' + error_file_name)
        np.save(vel_path, test_vel_mse.cpu().numpy())
        test_mse_1 = test_mse[0].item()
        test_mse_15 = test_mse[14].item()
        test_mse_25 = test_mse[24].item()
        print("FORWARD PRED RESULTS:")
        print("\t1 STEP: ",test_mse_1)
        print("\t15 STEP: ",test_mse_15)
        print("\t25 STEP: ",test_mse_25)

        print("POSITION FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_pos_mse[0].item())
        print("\t15 STEP: ", test_pos_mse[14].item())
        print("\t25 STEP: ", test_pos_mse[24].item())

        print("VELOCITY FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_vel_mse[0].item())
        print("\t15 STEP: ", test_vel_mse[14].item())
        print("\t25 STEP: ", test_vel_mse[24].item())

        f1, prec, rec, all_acc, acc_0, acc_1, edges = eval_edges(model, val_data, params)
        print("Val Edge results:")
        print("\tF1: ",f1)
        print("\tPrecision: ",prec)
        print("\tRecall: ",rec)
        print("\tAll predicted edge accuracy: ",all_acc)
        print("\tFirst Edge Acc: ",acc_0)
        print("\tSecond Edge Acc: ",acc_1)
        out_dir = os.path.join(args.working_dir, 'preds')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'encoder_edges.npy')
        np.save(out_path, edges.numpy())
        f1_score_path = os.path.join(out_dir, 'f1_score.npy')
        np.save(out_path, f1)

        if args.plot_samples:
            plot_sample(model, test_data, args.test_burn_in_steps, params)

    elif args.mode == 'save_pred':
        test_data = SmallSynthData(args.data_path, 'test', params)
        save_samples(model, test_data, args.test_burn_in_steps, params)
    elif args.mode == 'record_predictions':
        model.eval()
        burn_in = args.test_burn_in_steps
        forward_pred = 50 - args.test_burn_in_steps
        test_data = SmallSynthData(args.data_path, 'test', params)
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
