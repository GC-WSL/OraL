import argparse
import os

def get_args_Bay(seed):
    print('---------------------------func: get_args_Bay---------------------------')
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Bay.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='Bay',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file', default='./data/num_idx_Bay.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    path_NL = './result_NL'
    isExists = os.path.exists(path_NL)
    if not isExists:
        os.makedirs(path_NL)
    else:
        print('There is ', path_NL)
    parser.add_argument('--save_model_path',
                        default= path + '/Bay_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',
                        default= path + '/Bay_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_model_path_NL',default=path_NL + '/' + str(seed),
                        type=str, help='path filename of the trained model')

    parser.add_argument('--sess', default='name', type=str)
    parser.add_argument('--double', default=False, type=bool)
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--tune', default=False, type=bool)

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--phases', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--meta_epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--epochs', default=[5,50], type=list, metavar='N', help='number of total epochs to run')
    parser.add_argument('--HBCD_epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float) 
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args

def get_args_USA(seed):
    print('---------------------------func: get_args_USA---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/USA.mat', # './data/USA.mat', '../Dropdomain/DDdata_USA.mat'
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='USA',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file',
                        default='./data/num_idx_usa.mat',
                        type=str, help='path filename of the trained model')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
        
    path_NL = './result_NL'
    isExists = os.path.exists(path_NL)
    if not isExists:
        os.makedirs(path_NL)
    else:
        print('There is ', path_NL)
        
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    parser.add_argument('--save_model_path',
                        default= path + '/Bay_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',
                        default= path + '/Bay_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_model_path_NL',default=path_NL + '/' + str(seed),
                        type=str, help='path filename of the trained model')

    parser.add_argument('--sess', default='name', type=str)
    parser.add_argument('--double', default=False, type=bool)
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--tune', default=False, type=bool)

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--phases', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--meta_epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--epochs', default=[5, 50], type=list, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float) 
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)

    return args

def get_args_Barbara(seed):
    print('---------------------------func: get_args_Barbara---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Barbara.mat',# './data/Barbara.mat', '../Dropdomain/DDdata_Barbara.mat'
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='Barbara',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    path_NL = './result_NL'
    isExists = os.path.exists(path_NL)
    if not isExists:
        os.makedirs(path_NL)
    else:
        print('There is ', path_NL)
        
    parser.add_argument('--save_model_path',
                            default= path + '/Bay_'+str(seed)+'.pkl',
                            type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',
                        default= path + '/Bay_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_model_path_NL',default=path_NL + '/' + str(seed),
                        type=str, help='path filename of the trained model')
    parser.add_argument('--sess', default='name', type=str)
    parser.add_argument('--double', default=False, type=bool)
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--tune', default=False, type=bool)
    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--phases', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--meta_epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--epochs', default=[5, 20], type=list, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float) #5e-4
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args
