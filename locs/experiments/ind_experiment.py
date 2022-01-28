import os

import numpy as np

from locs.utils.flags import build_flags
import locs.models.model_builder as model_builder
from locs.datasets.ind_data import IndData, ind_collate_fn
import locs.training.train_dynamicvars as train
import locs.training.train_utils as train_utils
import locs.training.evaluate as evaluate
import locs.utils.misc as misc


if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--error_out_name', default='val_prediction_errors.npy')
    parser.add_argument('--train_data_len', type=int, default=-1)
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--expand_train', action='store_true')
    parser.add_argument('--final_test', action='store_true')
    parser.add_argument('--vel_norm_norm', action='store_true')
    parser.add_argument('--report_error_norm', action='store_true')
    parser.add_argument('--test_short_sequences', action='store_true')
    parser.add_argument('--present_gnn', action='store_true')
    parser.add_argument('--isotropic', action='store_true')
    parser.add_argument('--pos_representation', choices=['cart', 'polar'], default='polar')
    parser.add_argument('--strict', type=int, default=-1)

    args = parser.parse_args()
    params = vars(args)

    misc.seed(args.seed)

    params['input_size'] = 4
    params['nll_loss_type'] = 'gaussian'
    params['dynamic_vars'] = True
    params['collate_fn'] = ind_collate_fn
    train_data = IndData(args.data_path, 'train', params)
    val_data = IndData(args.data_path, 'valid', params)

    model = model_builder.build_model(params)
    if args.mode == 'train':
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer, val_writer)
    elif args.mode == 'eval':
        if args.final_test:
            test_data = IndData(args.data_path, 'test', params)
            test_mse, test_pos_mse, test_vel_mse, counts = evaluate.eval_forward_prediction_dynamicvars_unnormalized(model, test_data, params)
        else:
            test_mse, test_pos_mse, test_vel_mse, counts = evaluate.eval_forward_prediction_dynamicvars_unnormalized(model, val_data, params)

        file_prefix = "norm_" if args.report_error_norm else ""
        error_file_name = file_prefix + args.error_out_name

        path = os.path.join(args.working_dir, error_file_name)
        np.save(path, test_mse.cpu().numpy())
        path = os.path.join(args.working_dir, 'counts_' + error_file_name)
        np.save(path, counts.cpu().numpy())

        mid_pred_step = int(len(test_mse) / 2)
        test_mse_1 = test_mse[0].item()
        test_mse_mid = test_mse[mid_pred_step].item()
        test_mse_final = test_mse[-1].item()
        if args.final_test:
            print("TEST FORWARD PRED RESULTS:")
        else:
            print("VAL FORWARD PRED RESULTS:")
        print("\t1 STEP:  ", test_mse_1, counts[0].item())
        print(f"\t{mid_pred_step+1} STEP: ", test_mse_mid, counts[mid_pred_step].item())
        print(f"\t{len(test_mse)} STEP: ", test_mse_final, counts[-1].item())

        pos_path = os.path.join(args.working_dir, 'pos_' + error_file_name)
        np.save(pos_path, test_pos_mse.cpu().numpy())
        vel_path = os.path.join(args.working_dir, 'vel_' + error_file_name)
        np.save(vel_path, test_vel_mse.cpu().numpy())

        print("POSITION FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_pos_mse[0].item())
        print(f"\t{mid_pred_step+1} STEP: ", test_pos_mse[mid_pred_step].item())
        print(f"\t{len(test_mse)} STEP: ", test_pos_mse[-1].item())

        print("VELOCITY FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_vel_mse[0].item())
        print(f"\t{mid_pred_step+1} STEP: ", test_vel_mse[mid_pred_step].item())
        print(f"\t{len(test_mse)} STEP: ", test_vel_mse[-1].item())
