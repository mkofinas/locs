import os
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch


def single_plot_trajectories(ax, x, y, y_pred):
    """
    x shape: T_gt x N x D
    y shape: T_pr x N x D
    y_pred shape: T_pr x N x D
    """
    colors = ['firebrick', 'forestgreen', 'dodgerblue', 'mediumvioletred', 'darkturquoise']
    pred_colors = ['lightsalmon', 'lightgreen', 'lightskyblue', 'palevioletred', 'lightskyblue']

    num_obj = x.size(1)
    num_gt_steps = x.size(0)
    num_pred_steps = y.size(0)
    full_traj = torch.cat([x, y], dim=0)
    num_steps = full_traj.size(0)

    color_range = np.arange(0, num_pred_steps+1) / (num_pred_steps)

    # Prepend present timestep to prediction to be able to draw the predictions
    # in piece-wise linear trajectories, each with constant transparency
    full_y_pred = torch.cat([x[[-1]], y_pred], dim=0)
    for i in range(num_obj):
        for j in range(1, num_steps):
            ax.plot(full_traj[[j-1, j], i, 0], full_traj[[j-1, j], i, 1], '-',
                    color=colors[i], alpha=0.1+0.9*j/num_steps)

        for j in range(num_gt_steps-1, num_steps):
            ax.plot(full_traj[j, i, 0], full_traj[j, i, 1], 'o', color=colors[i],
                    alpha=1.0,
                    markersize=3+2*color_range[j-num_gt_steps+1],
                    markeredgecolor='black' if j == (num_gt_steps-1) else None)

        for j in range(1, num_pred_steps+1):
            ax.plot(full_y_pred[[j-1, j], i, 0], full_y_pred[[j-1, j], i, 1],
                    '-', color=pred_colors[i], alpha=0.5)
            ax.plot(full_y_pred[j, i, 0], full_y_pred[j, i, 1], 'o',
                    color=pred_colors[i], alpha=0.5, markersize=3+2*color_range[j-1])


def set_axis(ax, full_traj, idx, ndim=2):
    if ndim == 2:
        ax.axis('equal')
    else:
        ax.axis('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    max_range = full_traj[idx][..., :ndim].reshape(-1, ndim).max(0)[0]
    min_range = full_traj[idx][..., :ndim].reshape(-1, ndim).min(0)[0]
    full_range = max_range - min_range
    multiplier = 0.0 if ndim == 2 else 0.1
    ax.set_xlim(min_range[0] - multiplier * full_range[0], max_range[0] + multiplier * full_range[0])
    ax.set_ylim(min_range[1] - multiplier * full_range[1], max_range[1] + multiplier * full_range[1])
    if ndim == 3:
        ax.set_zticks([])
        ax.set_zlim(min_range[2] - multiplier * full_range[2],
                    max_range[2] + multiplier * full_range[2])


def multicolumn_plot_trajectories(x, y, y_pred, indices, errors, ndim=2,
                                  separate_rows=True, separate_columns=True,
                                  latex=False, png=False, show_errors=True):
    """
    x shape: B x T_gt x N x D
    y shape: B x T_pr x N x D
    y_pred shape: list [B x T_pr x N x D]
    indices shape: M, all elements in [0, N)
    """
    colors = ['firebrick', 'forestgreen', 'dodgerblue', 'mediumvioletred', 'darkturquoise']
    # pred_colors = ['lightsalmon', 'lightgreen', 'lightskyblue', 'palevioletred', 'lightskyblue']

    num_columns = len(y_pred) + 1
    num_rows = len(indices)

    num_obj = x.size(2)
    num_gt_steps = x.size(1)
    num_pred_steps = y.size(1)
    full_traj = torch.cat([x, y], dim=1)
    num_steps = full_traj.size(1)

    alpha_range = 0.1 + 0.9 * np.arange(0, num_steps) / (num_steps-1)
    marker_range = 1.5 + 1.5 * np.arange(0, num_pred_steps+1) / (num_pred_steps)

    # Prepend present timestep to prediction to be able to draw the predictions
    # in piece-wise linear trajectories, each with constant transparency
    full_y_pred = [torch.cat([x, yp], dim=1) for yp in y_pred]

    figs = (np.empty((num_rows, num_columns), dtype=object)
            if separate_rows and separate_columns
            else [] if separate_rows else plt.figure())

    for i, idx in enumerate(indices):
        if separate_rows and separate_columns:
            figs[i, 0] = plt.figure()
            # figs[i, 0] = plt.figure(constrained_layout=True)
            ax = (figs[i, 0].add_subplot(111, projection='3d') if ndim == 3
                  else figs[i, 0].add_subplot(111))
        elif separate_rows:
            figs.append(plt.figure())
            ax = figs[-1].add_subplot(1, num_columns, 1, projection='3d') if ndim == 3 else figs[-1].add_subplot(1, num_columns, 1)
        else:
            serial_idx = i * num_columns + 1
            ax = (figs.add_subplot(num_rows, num_columns, serial_idx, projection='3d')
                  if ndim == 3 else figs.add_subplot(num_rows, num_columns, serial_idx))

        set_axis(ax, full_traj, idx, ndim)
        if latex:
            if ndim == 2:
                ax.set_xlabel(r'$\phantom{0.0}$')
            else:
                ax.text2D(0.5, -0.05, r'$\phantom{0.0}$', transform=ax.transAxes)
        if show_errors and png and ndim == 2:
            ax.set_xlabel(' ')
        for obj in range(num_obj):
            # Column 0 - Groundtruth
            for t in range(1, num_steps):
                ax.plot(
                    *(full_traj[idx, [t-1, t], obj, :ndim].T), '-',
                    color=colors[obj], alpha=alpha_range[t])

            for t in range(num_gt_steps-1, num_steps):
                ax.plot(
                    *(full_traj[idx, t, obj, :ndim]), 'o',
                    color=colors[obj], alpha=alpha_range[t] if t != (num_gt_steps-1) else 1.0,
                    markersize=marker_range[t-num_gt_steps+1],
                    markeredgecolor='black' if t == (num_gt_steps-1) else None)

        # Column 1+ - Predictions
        for model_idx in range(len(y_pred)):
            if separate_rows and separate_columns:
                figs[i, model_idx+1] = plt.figure()
                # figs[i, model_idx+1] = plt.figure(constrained_layout=True)
                ax = (figs[i, model_idx+1].add_subplot(111, projection='3d')
                      if ndim == 3 else figs[i, model_idx+1].add_subplot(111))
            elif separate_rows:
                serial_idx = model_idx + 2
                ax = (figs[-1].add_subplot(1, num_columns, serial_idx, projection='3d')
                      if ndim == 3 else figs[-1].add_subplot(1, num_columns, serial_idx))
            else:
                serial_idx = i * num_columns + model_idx + 2
                ax = (figs.add_subplot(num_rows, num_columns, serial_idx, projection='3d')
                      if ndim == 3 else figs.add_subplot(num_rows, num_columns, serial_idx))

            set_axis(ax, full_traj, idx, ndim)
            if show_errors:
                if ndim == 2:
                    ax.set_xlabel(f'{errors[model_idx][idx].item():.3f}')
                else:
                    ax.text2D(0.5, -0.05, f'{errors[model_idx][idx].item():.3f}',
                              transform=ax.transAxes, fontsize=11)

            for obj in range(num_obj):
                for t in range(1, num_steps):
                    ax.plot(
                        *(full_y_pred[model_idx][idx, [t-1, t], obj, :ndim].T),
                        '-', color=colors[obj], alpha=alpha_range[t])
                for t in range(num_gt_steps-1, num_steps):
                    ax.plot(
                        *(full_y_pred[model_idx][idx, t, obj, :ndim]), 'o',
                        color=colors[obj], alpha=alpha_range[t] if t != (num_gt_steps-1) else 1.0,
                        markersize=marker_range[t-num_gt_steps+1],
                        markeredgecolor='black' if t == (num_gt_steps-1) else None)

                # Re-plot groundtruth in high transparency
                for t in range(1, num_steps):
                    ax.plot(
                        *(full_traj[idx, [t-1, t], obj, :ndim].T), '-',
                        color=colors[obj], alpha=0.1)

    if separate_rows and separate_columns:
        for i in range(figs.shape[0]):
            for j in range(figs.shape[1]):
                figs[i, j].tight_layout()
    elif separate_rows:
        for fig in figs:
            fig.tight_layout()
    else:
        figs.tight_layout()

    return figs


def main(args):
    model_map = {
        0: 'gt',
        1: 'locs',
        2: 'dnri',
        3: 'nri',
        4: 'egnn',
    }
    dataset = args.dataset
    num_dims = 2 if args.dataset == 'synth' else 3
    if dataset == 'synth':
        work_dir = [
            'results/synth/locs_release/seed_1',
            'results/synth/dnri_release/seed_1',
            'results/synth/nri_release/seed_1',
            'results/synth/egnn_release/seed_1',
        ]
    else:
        work_dir = [
            'results/charged_3d_5_unboxed/locs_release/seed_1',
            'results/charged_3d_5_unboxed/dnri_release/seed_1',
            'results/charged_3d_5_unboxed/nri_release/seed_1',
            'results/charged_3d_5_unboxed/egnn_release/seed_1',
        ]
    file_prefix = 'inter_' if args.interactive else ''
    in_file = os.path.join(work_dir[0], f'{file_prefix}all_inputs_{args.in_steps}_{args.pred_steps}.npy')
    inp = torch.from_numpy(np.concatenate(np.load(in_file)))
    out = []
    errors = []
    for model in work_dir:
        out_file = os.path.join(model, f'{file_prefix}all_outputs_{args.in_steps}_{args.pred_steps}.npy')
        out.append(torch.from_numpy(np.concatenate(np.load(out_file))))
        error_file = os.path.join(model, f'{file_prefix}all_errors_{args.in_steps}_{args.pred_steps}.npy')
        errors.append(torch.from_numpy(np.load(error_file)).mean(-1))
        print(model, error_file, errors[-1].shape)

    indices = [0, 1, 2, 3, 4, 5]
    if args.latex or args.png:
        os.makedirs('visualizations', exist_ok=True)

    if args.latex:
        figs = multicolumn_plot_trajectories(
            inp[:, :args.in_steps],
            inp[:, args.in_steps:args.in_steps+args.pred_steps], out, indices,
            errors, separate_rows=True, separate_columns=True,
            ndim=num_dims, latex=True)
        for i, idx in enumerate(indices):
            for j in range(figs.shape[1]):
                file_name = f'{file_prefix}{dataset}_qualitative_results_{args.in_steps}_{args.pred_steps}_{idx}_{model_map[j]}.pgf'
                figs[i, j].savefig(
                    os.path.join('visualizations', file_name),
                    bbox_inches='tight')
    elif args.png:
        figs = multicolumn_plot_trajectories(
            inp[:, :args.in_steps],
            inp[:, args.in_steps:args.in_steps+args.pred_steps], out, indices,
            errors, separate_rows=True, separate_columns=True,
            ndim=num_dims, latex=False, png=True, show_errors=args.print_errors)
        for i, idx in enumerate(indices):
            for j in range(figs.shape[1]):
                file_name = f'{file_prefix}{dataset}_qualitative_results_{args.in_steps}_{args.pred_steps}_{idx}_{model_map[j]}.png'
                figs[i, j].savefig(
                    os.path.join('visualizations', file_name),
                    bbox_inches='tight')
    else:
        figs = multicolumn_plot_trajectories(
            inp[:, :args.in_steps],
            inp[:, args.in_steps:args.in_steps+args.pred_steps], out, indices,
            errors, separate_rows=False, separate_columns=False,
            ndim=num_dims)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--latex', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--print_errors', action='store_true')
    parser.add_argument('--in_steps', type=int, default=10)
    parser.add_argument('--pred_steps', type=int, default=40)
    parser.add_argument('--dataset', type=str, choices=['charged', 'synth'])

    args = parser.parse_args()
    if args.latex:
        mpl_latex_params = {
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            'pgf.preamble': '\n'.join(
                [r"\usepackage{amsmath}", r"\usepackage{amssymb}",
                 r"\usepackage{inputenc}"]),
            'font.size': 11,
            'legend.fontsize': 11,
        }

        mpl.use("pgf")
        mpl.rcParams.update(mpl_latex_params)
    elif args.png:
        mpl_latex_params = {
            # 'font.family': 'serif',
            # 'font.size': 20,
            # 'legend.fontsize': 12,
        }

        mpl.rcParams.update(mpl_latex_params)
        mpl.use("TkAgg")
    else:
        mpl.use("TkAgg")

    main(args)
