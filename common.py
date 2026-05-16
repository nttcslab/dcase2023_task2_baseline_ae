import yaml
import itertools
import argparse

########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

def param_to_args_list(params):
    params = list(itertools.chain.from_iterable(zip(params.keys(), params.values())))
    args_list = []
    for param in params:
        if type(param) is list:
            args_list.extend([str(p) for p in param])
        else:
            args_list.append(str(param))
    return args_list

########################################################################
# argparse setting
########################################################################
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def float_or_None(v):
    if v.lower() in ["none", "null"]:
        return None
    return float(v)

def str_or_None(v):
    if v is None:
        return None
    if isinstance(v, str) and v.lower() in ["none", "null", ""]:
        return None
    return v

def get_argparse():
    parser = argparse.ArgumentParser(
            description='Main function to call training for different AutoEncoders')
    parser.add_argument('--model', type=str, default='DCASE2023T2-AE', metavar='N',
                        help='train model name')
    parser.add_argument('--score', type=str, default="MSE", choices=["MSE"])
    parser.add_argument('--seed', type=int, default=39876401, metavar='S',
                        help='random seed (default: 39876401)')
    
    parser.add_argument('--use_cuda', type=str2bool, default=True,
                        help='enables CUDA training')
    parser.add_argument('--gpu_id',type=int, nargs='*', default=[0,],
                        help='Specify GPU id')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--decision_threshold', type=float, default=0.9)
    parser.add_argument(
        '--threshold_mode',
        type=str,
        default='gamma',
        choices=['gamma', 'validation_percentile', 'mean_std', 'auto_sweep'],
        help='Threshold calibration mode: gamma (legacy), validation_percentile, mean_std, or auto_sweep.',
    )
    parser.add_argument(
        '--percentile_value',
        type=float,
        default=95.0,
        help='Percentile value for validation_percentile mode.',
    )
    parser.add_argument(
        '--std_multiplier',
        type=float,
        default=2.0,
        help='k multiplier for mean+std threshold mode: threshold = mean + k*std.',
    )
    parser.add_argument(
        '--threshold_percentile_candidates',
        type=float,
        nargs='*',
        default=[90.0, 95.0, 97.0, 99.0],
        help='Percentile candidates used in auto_sweep mode.',
    )
    parser.add_argument(
        '--threshold_std_candidates',
        type=float,
        nargs='*',
        default=[1.5, 2.0, 2.5, 3.0],
        help='Mean+std k candidates used in auto_sweep mode.',
    )
    parser.add_argument(
        '--threshold_selection_metric',
        type=str,
        default='f1_mean',
        choices=['f1_mean', 'f1_source', 'f1_target'],
        help='Metric used to select the best candidate in auto_sweep mode.',
    )
    parser.add_argument(
        '--source_model_export_dir',
        type=str_or_None,
        default=None,
        help='Optional export_dir to load an existing trained model from while writing outputs to a new export_dir.',
    )
    parser.add_argument('--max_fpr', type=float, default=0.1)

    # feature
    parser.add_argument('--n_mels',type=int, default=128, 
                        help='Length of the melfilter bank')
    parser.add_argument('--frames',type=int, default=5, 
                        help='Number of frames in a feature vector')
    parser.add_argument('--frame_hop_length',type=int, default=1, 
                        help='number of frames between successive feature')
    parser.add_argument('--use_global_norm', type=str2bool, default=False,
                        help='Use global training-set normalization')
    parser.add_argument('--use_temporal_stack', type=str2bool, default=False,
                        help='Use temporal context frame stacking')
    parser.add_argument('--temporal_context', type=int, default=5,
                        help='Temporal context size when stacking is enabled')
    parser.add_argument('--n_fft',type=int, default=1024, 
                        help='length of the FFT window')
    parser.add_argument('--hop_length',type=int, default=512, 
                        help='number of samples between successive frames')
    parser.add_argument('--power', type=float, default=2.0)
    parser.add_argument('--fmin', type=float, default=0.0)
    parser.add_argument('--fmax', type=float_or_None, default=None)
    parser.add_argument('--win_length', type=float_or_None, default=None)

    # fit
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', '--learning_rate', default=0.03, type=float,
                        help='learning rate (default: 0.03)')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping patience (default: 10)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='minimum validation loss improvement for early stopping')
    parser.add_argument('--shuffle', type=str, default="full",
                        help='shuffle type (full , simple)')
    parser.add_argument('--validation_split', type=float, default=0.1)

    # dataset
    parser.add_argument('--dataset_directory', type=str, default='data',
                        help='Where to parent dataset dir')
    parser.add_argument('--dataset', type=str, default='DCASE2023T2ToyCar', metavar='N',
                        help='dataset to use')
    parser.add_argument('-d', '--dev', action='store_true',
                        help='Use Development dataset')
    parser.add_argument('-e', '--eval', action='store_true',
                        help='Use Evaluation dataset')
    parser.add_argument('--use_ids', type=int, nargs='*', default=[],
                    help='Machine ID to be treated as nml data')
    parser.add_argument('--is_auto_download', type=str2bool, default=False,
                        help="Download dataset if not exist")
    parser.add_argument("--mono", type=str2bool, default=True,
                        help="Load input audio as monaural. DCASE2020T2-2025T2 are monaural, DCASE2026T2 is stereo")

    # save data
    parser.add_argument('--result_directory', type=str, default='results/', metavar='N',
                        help='Where to store images')
    parser.add_argument('--export_dir',type=str, default='', 
                        help='Name of the directory to be generated under the Result directory.')
    parser.add_argument('-tag','--model_name_suffix',type=str, default='', 
                        help='Add a word to file name')

    # resume learning
    parser.add_argument('--restart',action='store_true', 
                        help='Resume learning with checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default="",
                        help="Using checkpoint file path. default: this checkpoint")
    parser.add_argument('--train_only', action='store_true', default=False,
                        help='Run train only')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='Run test only')
    
    return parser