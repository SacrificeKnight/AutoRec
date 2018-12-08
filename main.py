import tensorflow as tf
import argparse
from data import getrating
from modle import AutoRec

parser = argparse.ArgumentParser(description='I-AutoRec')
#定义网络超参数
parser.add_argument('--batch_size',type=int,default=128,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--train_epoch',type=int,default=1500,help='train epoch')
parser.add_argument('--display_step',type=int,default=10,help='display step')
parser.add_argument('--n_hidden',type=int,default=500,help='hidden number')
parser.add_argument('--lambda_value',type=float,default=0.1,help='lambda value')
args = parser.parse_args()

path = 'ml-1m/ratings.csv'
num_user = 6040
num_movie = 3952

with tf.Session() as sess:
    data, num_train_data, num_test_data = getrating(path, num_user, num_movie)
    AutoRec = AutoRec(data, args, sess, num_user, num_movie, num_train_data, num_test_data)
    AutoRec.train()
