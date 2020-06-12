import argparse
from preprocess import Data_process
from train import Train


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--L_hgn', type=int, default=5, help='number of low layers')
parser.add_argument('--T_hgn', type=int, default=3, help='number of low layers')
parser.add_argument('--L_mkr', type=int, default=1, help='number of low layers')
parser.add_argument('--H_mkr', type=int, default=1, help='number of high layers')
parser.add_argument('--lr_rs', type=float, default=1, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-5, help='learning rate of KGE task')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')       
parser.add_argument('--neg_samples', type=int, default=3, help='number of low layers')
parser.add_argument('--sets_of_neg_samples', type=int, default=50, help='number of low layers')
parser.add_argument('--n_filters', type=int, default=3, help='number of low layers')
parser.add_argument('--filter_height', type=int, default=2, help='number of low layers')

if __name__ == '__main__':
    args = parser.parse_args()
    data_process = Data_process(args)
    param_,data_,test_ = data_process.run()
    Train(param_,data_,test_,data_process).run()
