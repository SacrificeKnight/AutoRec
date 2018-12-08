import tensorflow as tf
import tensorlayer as tl
import numpy as np


class AutoRec():
    def __init__(self, data, args, sess, num_user, num_movie,
                 num_train_data,
                 num_test_data):
        self.data = data
        self.num_train_data = num_train_data
        self.num_test_data = num_test_data

        self.sess = sess
        self.args = args

        self.num_user = num_user
        self.num_movie = num_movie

        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.train_epoch = args.train_epoch
        self.display_step = args.display_step
        self.n_hidden = args.n_hidden
        self.lambda_value = args.lambda_value

        self.total_batch = int(np.ceil(self.num_user / self.batch_size))

        # 模型输入
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.num_movie], name='input')
        self.input_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.num_movie], name='input_mask')

        # 模型参数
        # self.V = tf.Variable(tf.random_normal([self.num_movie, self.n_hidden]))
        # self.W = tf.Variable(tf.random_normal([self.n_hidden, self.num_movie]))

        # self.mu = tf.Variable(tf.random_normal([self.n_hidden]))
        # self.b = tf.Variable(tf.random_normal([self.num_movie]))

        self.V = tf.get_variable(name="V", initializer=tf.truncated_normal(shape=[self.num_movie, self.n_hidden],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        self.W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[self.n_hidden, self.num_movie],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        self.mu = tf.get_variable(name="mu", initializer=tf.zeros(shape=self.n_hidden),dtype=tf.float32)
        self.b = tf.get_variable(name="b", initializer=tf.zeros(shape=self.num_movie), dtype=tf.float32)

        # 自编码器
        self.encoder = tf.nn.sigmoid(tf.add(tf.matmul(self.input, self.V), self.mu))
        pred_r = tf.nn.sigmoid(tf.add(tf.matmul(self.encoder, self.W), self.b))
        self.pred_r = tf.identity(pred_r)

        # 定义cost函数以及优化器
        pre_rec_cost = tf.multiply((self.input - self.pred_r) , self.input_mask)
        rec_cost = tf.square(self.l2_norm(pre_rec_cost))
        pre_reg_cost = tf.square(self.l2_norm(self.W)) + tf.square(self.l2_norm((self.V)))
        reg_cost = self.lambda_value * 0.5 * pre_reg_cost

        self.cost = rec_cost + reg_cost
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

    def train(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.train_epoch):
            rand = np.random.permutation(self.num_user)
            total_cost = 0
            for i in range(self.total_batch):
                if i == self.total_batch-1:
                    k = rand[i*self.batch_size:]
                else:
                    k = rand[i*self.batch_size:(i+1)*self.batch_size]
                # print(self.data['train']['mask'][k,:])
                _,cost = self.sess.run([self.optimizer, self.cost],
                                       feed_dict={self.input: self.data['train']['r'][k, :],
                                                  self.input_mask: self.data['train']['mask'][k, :]})
                total_cost += cost
            if epoch % self.display_step == 0:
                print('Epoch: ' + str(epoch) + '| Total cost: ' + str(total_cost))
            self.test(epoch)
        print('finish!!!')

    def test(self,iter):
        Cost, Decoder = self.sess.run([self.cost, self.pred_r],
                                    feed_dict = {self.input:self.data['test']['r'],
                                                 self.input_mask: self.data['test']['mask']})
        if iter % self.display_step==0:
            Estimated_R = Decoder.clip(min=1,max=5)
            unseen_user_test_list = list(self.data['test']['user'] - self.data['train']['user'])
            unseen_movie_test_list = list(self.data['test']['movie'] - self.data['train']['movie'])

            # print(Estimated_R)

            for user in unseen_user_test_list:
                for movie in unseen_movie_test_list:
                    if self.data['test']['mask'][user, movie] == 1:
                        Estimated_R[user, movie] = 3

            pre_numrator = np.multiply((Estimated_R - self.data['test']['r']), self.data['test']['mask'])
            # print(pre_numrator)
            numrator = np.sum(np.square(pre_numrator))
            # print('numrator: ' + str(numrator))
            denominator = self.num_test_data
            RMSE = np.sqrt(numrator / float(denominator))
            print('RMSE : ' + str(RMSE))

    def l2_norm(self,tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))