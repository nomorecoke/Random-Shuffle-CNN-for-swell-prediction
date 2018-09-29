import tensorflow as tf
import numpy as np
import os

from constants import RESULT_PATH, TF_SEED, NP_SEED
import util
from visualizer import plot_metrics

np.random.seed(NP_SEED)
tf.set_random_seed(TF_SEED)

def to_one_hot(y_sparse):
    # y_sparse : shape of (-1, 1)
    # return : shape of(-1, 3)
    if y_sparse.dtype != int:
        y_sparse = y_sparse.astype(int)
    return np.eye(3)[y_sparse.ravel()]

def to_sparse(y_one_hot):
    # y_sparse : shape of (-1, 3)
    # return : shape of(-1, 1)
    y_sparse = np.argmax(y_one_hot, axis=-1)
    return y_sparse

class NN:
    def __init__(self, session: tf.Session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name):

        self.name = name
        self.n_class = n_class
        self.dropout_keep_prob = dropout_keep_prob
        self.n_lookback = n_lookback
        self.n_feature = n_feature


        self.metrics = {'train': {'loss': []},
                        'val': {'loss': [], 'score_seq': []},
                        'test': {'loss': [], 'score_seq': []}}
        self.predicts = {'val':[], 'test':[], 'problem':[],
                         'val_softmax':[],'test_softmax':[],'problem_softmax':[]}

        self.sess = session

        with tf.name_scope(name):
            self.x = tf.placeholder(tf.float32, [None, n_lookback, n_feature], name='x')
            self.y = tf.placeholder(tf.float32, [None, n_class], name='y_onehot')
            self.w = tf.placeholder(tf.float32, [None, ], name='w')

            self.is_training = tf.placeholder(tf.bool)
            self.tf_dropout_keep_prob = tf.placeholder(tf.float32)
            global_step = tf.Variable(0, trainable=False)
            decaying_lr = tf.train.exponential_decay(lr, global_step, 100, lr_decay)

            x = self.x
            output = self.hidden_layers(x)  # (-1, 3)
            
            self.y_softmax = tf.nn.softmax(output)
            self.y_pred = tf.expand_dims(tf.argmax(output, axis=-1), 1)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=output))
            # self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=output, weights=self.w))

            # self.optimizer = tf.train.AdamOptimizer(decaying_lr)
            self.optimizer = tf.contrib.opt.NadamOptimizer(decaying_lr)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

            self.saver = tf.train.Saver()

    def hidden_layers(self, x):
        """
        x: shape=[-1, n_lookback, n_feature]
        :return: logit tensor (shape=[-1, n_class])
        """
        raise NotImplementedError("YOU MUST IMPLEMENT hidden_layers() FUNCTION")
        pass

    @staticmethod
    def swell_noise_layer(input_layer, std):
            """
            마지막 feature인 swell_t-1에 대해 가우시안 노이즈 추가.
            가우시안 노이즈 추가한 후 [0.0~2.0]으로 클리핑한 후 원래 input에 적용한다.
            :param input_layer: a tensor shape of [batch_size, n_lookback, n_feature]
            :param std: stddev of noise
            :return:
            """
            noise_shape = tf.shape(input_layer)[0], tf.shape(input_layer)[1], 1
            noise = tf.truncated_normal(shape=noise_shape, mean=0.0, stddev=std, dtype=tf.float32)
            noised_swell = input_layer[:, :, -1:] + noise
            noised_swell = tf.clip_by_value(noised_swell, 0.0, 2.0)
            input_layer = tf.concat([input_layer[:, :, :-1], noised_swell], axis=-1)
            return input_layer

    @staticmethod
    def add_noise_layer(input_layer, std):
        """
        마지막 feature(swell_t-1)를 제외한 나머지 feature 들에 대해
        가우시안 노이즈 추가한다.
        :param input_layer: a tensor shape of [batch_size, n_lookback, n_feature]
        :param std: stddev of noise
        :return:
        """
        noise_shape = tf.shape(input_layer)[0], tf.shape(input_layer)[1], tf.shape(input_layer)[2] - 1
        noise = tf.truncated_normal(shape=noise_shape, mean=0.0, stddev=std, dtype=tf.float32)
        noised_input = input_layer[:, :, :-1] + noise
        input_layer = tf.concat([noised_input, input_layer[:, :, -1:]], axis=-1)
        return input_layer

    def run_batch(self, x, y, w, batch_size, is_training):

        if batch_size == -1:
            batch_size = len(x)

        total_loss = 0
        y_preds = []

        total_steps = len(x) // batch_size
        last_batch_size = len(x) - total_steps * batch_size
        if last_batch_size != 0:
            print('last batch is size of {}'.format(last_batch_size))
            total_steps += 1

        if is_training:
            for j in range(total_steps):
                x_batch = x[j * batch_size: (j + 1) * batch_size]
                y_batch = y[j * batch_size: (j + 1) * batch_size]
                w_batch = w[j * batch_size: (j + 1) * batch_size]
                _, loss, y_pred = self.sess.run([self.train_op, self.loss, self.y_pred],
                                                   feed_dict={
                                                       self.x: x_batch,
                                                       self.y: y_batch,
                                                       self.w: w_batch,
                                                       self.tf_dropout_keep_prob: self.dropout_keep_prob,
                                                       self.is_training: True})
                total_loss += loss
                y_preds.append(y_pred)

        else:
            for j in range(total_steps):
                x_batch = x[j * batch_size: (j + 1) * batch_size]
                y_batch = y[j * batch_size: (j + 1) * batch_size]
                w_batch = w[j * batch_size: (j + 1) * batch_size]
                loss, y_pred = self.sess.run([self.loss, self.y_pred],
                                                     feed_dict={
                                                         self.x: x_batch,
                                                         self.y: y_batch,
                                                         self.w: w_batch,
                                                         self.tf_dropout_keep_prob: 1.,
                                                         self.is_training: False})
                total_loss += loss
                y_preds.append(y_pred)

        y_preds = np.concatenate(y_preds)
        return total_loss/total_steps, y_preds

    def train(self, ds, BATCH_SIZE, EPOCH, feature_shuffle=False, train_all_data=False, verbose=True):


        # feature 순서 랜덤 셔플. 단, 마지막 feature는 swell_t-1으로 고정
        if feature_shuffle:
            p = np.random.permutation(ds['train']['x'].shape[2]-1)
            ds['train']['x'][:, :, :len(p)] = ds['train']['x'][:,:,p]
            ds['val']['x'][:, :, :len(p)] = ds['val']['x'][:,:,p]
            ds['test']['x'][:, :, :len(p)] = ds['test']['x'][:,:,p]
            ds['problem']['x'][:, :, :len(p)] = ds['problem']['x'][:,:,p]

        # one hot
        ds['train']['y_onehot'] = to_one_hot(ds['train']['y'])
        ds['val']['y_onehot'] = to_one_hot(ds['val']['y'])
        ds['test']['y_onehot'] = to_one_hot(ds['test']['y'])

        # class weight
        for d0 in ['train', 'val', 'test']:
            # w0 = (ds[d0]['y'] == 0).sum() / len(ds[d0]['y'])
            # w1 = (ds[d0]['y'] == 1).sum() / len(ds[d0]['y'])
            # w2 = (ds[d0]['y'] == 2).sum() / len(ds[d0]['y'])
            # ds[d0]['w'] = np.array([1 / w0, 1 / w1, 1 / w2])[ds[d0]['y']]
            # ds[d0]['w'] = np.array([1 / 0.45, 1 / 0.36, 1 / 0.19])[ds[d0]['y']]
            diff_samples = ds[d0]['x'][:, -1, -1] != ds[d0]['y'][:, -1]
            ds[d0]['w'] = np.array([1, 2])[diff_samples.astype(int)]

        for i in range(EPOCH):

            print('[NAME: {}, EPOCH: {}]'.format(self.name, i))
            # Train
            train_loss, _ = self.run_batch(ds['train']['x'], ds['train']['y_onehot'], ds['train']['w'], BATCH_SIZE, is_training=True)

            # Validation
            if verbose:
                print('predict ONE Validation')
            val_loss, val_pred_one = self.run_batch(ds['val']['x'], ds['val']['y_onehot'], ds['val']['w'], BATCH_SIZE, is_training=train_all_data)
            val_acc_one, val_score_one, val_max_score = util.calc_metric(ds['val']['y'].ravel(),
                                                          val_pred_one.round().astype(int).ravel(), self.n_class, verbose)
            
            if verbose:
                print('predict SEQ Validation')
            val_pred_seq, val_softmax_seq = self.predict_sequence(ds['val']['x'])
            val_pred_seq = val_pred_seq.round().astype(int).ravel()
            val_acc_seq, val_score_seq, val_max_score = util.calc_metric(ds['val']['y'].ravel(),
                                                          val_pred_seq, self.n_class, verbose)

            # Test
            if verbose:
                print('predict ONE Test')
            test_loss, test_pred_one = self.run_batch(ds['test']['x'], ds['test']['y_onehot'], ds['test']['w'], BATCH_SIZE, is_training=train_all_data)
            test_acc_one, test_score_one, test_max_score = util.calc_metric(ds['test']['y'].ravel(),
                                                            test_pred_one.ravel().round().astype(int), self.n_class, verbose)
            
            if verbose:
                print('predict SEQ Test')
            test_pred_seq, test_softmax_seq = self.predict_sequence(ds['test']['x'])
            test_pred_seq = test_pred_seq.round().astype(int).ravel()
            test_acc_seq, test_score_seq, test_max_score = util.calc_metric(ds['test']['y'].ravel(),
                                                            test_pred_seq, self.n_class, verbose)

            print("[SUMMARY]\n(Loss) train: {:.5} val: {:.5} test: {:.5}".format(train_loss, val_loss, test_loss))
            print("val_acc_seq : {:.5} val_score_seq : {:.5} (max: {:.5})".format(val_acc_seq, val_score_seq, val_max_score))
            print("test_acc_seq: {:.5} test_score_seq: {:.5} (max: {:.5})\n".format(test_acc_seq, test_score_seq, test_max_score))

            # append current epoch's metrics
            self.metrics['train']['loss'].append(train_loss)
            self.metrics['val']['loss'].append(val_loss)
            self.metrics['test']['loss'].append(test_loss)
            self.metrics['val']['score_seq'].append(val_score_seq)
            self.metrics['test']['score_seq'].append(test_score_seq)

            if verbose:
                plot_metrics(**self.metrics)

            # predict Problem
            problem_pred, problem_softmax= self.predict_sequence(ds['problem']['x'])
            problem_pred = problem_pred.astype(int).ravel()

            # append current epoch's predictions
            self.predicts['val'].append(val_pred_seq)
            self.predicts['test'].append(test_pred_seq)
            self.predicts['problem'].append(problem_pred)
            self.predicts['val_softmax'].append(val_softmax_seq)
            self.predicts['test_softmax'].append(test_softmax_seq)
            self.predicts['problem_softmax'].append(problem_softmax)
        # END for i in range(EPOCH):

        return self.predicts

    def restore(self, path):
        self.saver.restore(self.sess, path)

    def predict_one(self, x):
        """
        :param x: shape=[N_LOOKBACK, N_FEATURE]
        :return: y
        """
        x = x[np.newaxis,]
        assert len(x[np.isnan(x)]) == 0
        y_pred = self.sess.run(self.y_pred, feed_dict={self.x: x,
                                                       self.tf_dropout_keep_prob: 1.,
                                                       self.is_training: False})
        return y_pred

    def predict_batch(self, x_list):
        y_pred, y_softmax = self.sess.run([self.y_pred, self.y_softmax], 
                                                    feed_dict={self.x: x_list,
                                                    self.tf_dropout_keep_prob: 1.,
                                                    self.is_training: False})
        return y_pred, y_softmax

    def predict_sequence(self, x_list):
        """
        test data와 같은 데이터로 예측할 때 사용하는 함수
        * 테스트 데이터같은 경우 연속된 24시간을 예측해야 하기 때문에, swell_t-1 데이터 중간에 nan이 들어있다.
        이를 NN을 이용해 한 시간 단위로 예측한 후 swell_t-1데이터를 채워나가며 예측을 진행한다.
        :param x_list: shape=[-1, N_LOOKBACK, N_FEATURE]
        :return:  list of y
        :param x_list:
        :return:
        """
        x_day_list = x_list.reshape(-1, 24, self.n_lookback, self.n_feature).copy()

        y_preds = np.zeros((x_list.shape[0] // 24, 24))
        y_softmaxs = np.zeros((x_list.shape[0] // 24, 24, 3))

        for i in range(24):
            x_batch = x_day_list[:, i, :, :]
            y_pred_batch, y_softmax_batch = self.predict_batch(x_batch)
            y_preds[:, i:i + 1] = y_pred_batch
            y_softmaxs[:, i, :] = y_softmax_batch
            for j in range(1, min(self.n_lookback + 1, 24 - i)):
                x_day_list[:, i + j, -j, -1] = y_pred_batch.ravel()

        y_preds = y_preds.ravel()
        y_softmaxs = y_softmaxs.reshape(y_softmaxs.shape[0]*y_softmaxs.shape[1], 3)

        return y_preds, y_softmaxs


class Dense(NN):

    def __init__(self, session: tf.Session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name):
        super().__init__(session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name)

    def hidden_layers(self, x):

        x = tf.cond(self.is_training, lambda: self.swell_noise_layer(x, 2.0), lambda: x)
        x = tf.cond(self.is_training, lambda: self.add_noise_layer(x, 0.1), lambda: x)
        X = tf.reshape(x, (-1, x.shape[1] * x.shape[2]))
        print(X)
        layer1 = tf.contrib.layers.fully_connected(X, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        # layer1 = tf.layers.batch_normalization(layer1, training=self.is_training, renorm=True, momentum=0.999)
        layer1 = tf.layers.dropout(layer1, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        layer2 = tf.contrib.layers.fully_connected(layer1, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        # layer2 = tf.layers.batch_normalization(layer2, training=self.is_training, renorm=True, momentum=0.999)
        layer2 = tf.layers.dropout(layer2, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        layer3 = tf.contrib.layers.fully_connected(layer2, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        # layer3 = tf.layers.batch_normalization(layer3, training=self.is_training, renorm=True, momentum=0.999)
        layer3 = tf.layers.dropout(layer3, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        layer4 = tf.contrib.layers.fully_connected(layer3, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        # layer4 = tf.layers.batch_normalization(layer4, training=self.is_training, renorm=True, momentum=0.999)
        layer4 = tf.layers.dropout(layer4, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        layer5 = tf.contrib.layers.fully_connected(layer4, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        # layer5 = tf.layers.batch_normalization(layer5, training=self.is_training, renorm=True, momentum=0.999)
        layer5 = tf.layers.dropout(layer5, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        output = tf.contrib.layers.fully_connected(layer5, 3, activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        return output


class RAND_CNN(NN):
    def __init__(self, session: tf.Session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name):
        super().__init__(session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name)
        
    def hidden_layers(self, x):
         #x: [None, n_lookback, n_feature]   
        print("INPUT SHAPE",x.shape)
        x = tf.cond(self.is_training, lambda: self.swell_noise_layer(x, 2.0), lambda: x)
        x = tf.cond(self.is_training, lambda: self.add_noise_layer(x, 0.1), lambda: x)

        X = tf.reshape(x, [-1, self.n_lookback, self.n_feature, 1])
        conv1 = tf.layers.conv2d(X, filters=64,kernel_size=[1,  3], strides=[1, 1],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.relu)
        #conv1 = tf.layers.max_pooling2d(conv1, [1, 2], [1, 1])
        #print(conv1)
        # conv1 = tf.nn.dropout(conv1, self.dropout_keep_prob)
        conv1 = tf.layers.dropout(conv1, rate=0.5,
                                   #noise_shape=[self.batch_size, *conv1.shape[1:]],
                                   training=self.is_training)
        print(conv1)
        conv2 = tf.layers.conv2d(conv1, filters=64,kernel_size=[conv1.shape[1], 4], strides=[1, 1],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.relu)
        # conv2 = tf.nn.dropout(conv2, self.dropout_keep_prob)
        conv2 = tf.layers.dropout(conv2, rate=0.5,
                                   #noise_shape=[self.batch_size, *conv2.shape[1:]],
                                   training=self.is_training)
        print(conv2)
        
#         conv3 = tf.layers.conv2d(conv2, filters=64,kernel_size=[conv2.shape[1], 4], strides=[1, 1],
#                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                  activation=tf.nn.relu)
#         # conv2 = tf.nn.dropout(conv2, self.dropout_keep_prob)
#         conv3 = tf.layers.dropout(conv3, rate=0.5,
#                                    #noise_shape=[self.batch_size, *conv2.shape[1:]],
#                                    training=self.is_training)
#         print(conv3)


        conv2 = tf.contrib.layers.flatten(conv2)
        """
        fc1 = tf.contrib.layers.fully_connected(conv2, 1024, activation_fn=tf.nn.relu,    
                                  weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        fc1 = tf.layers.dropout(fc1, rate=1 - self.dropout_keep_prob,
                                   training=self.is_training, seed=TF_SEED)
                                   """
        output = tf.contrib.layers.fully_connected(conv2, 3, activation_fn=None)

        return output


class RNN(NN):

    def __init__(self, session: tf.Session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name):
        super().__init__(session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name)

    def hidden_layers(self, x):

        x = tf.cond(self.is_training, lambda: self.swell_noise_layer(x, 2.0), lambda: x)
        x = tf.cond(self.is_training, lambda: self.add_noise_layer(x, 0.1), lambda: x)
 

        fw_cell = []
        cell_1 = tf.contrib.rnn.NASCell(num_units=512) #, state_is_tuple=True)
        cell_1 = tf.contrib.rnn.DropoutWrapper(cell_1, output_keep_prob=self.tf_dropout_keep_prob)
        fw_cell.append(cell_1)
        
        cell_2 = tf.contrib.rnn.NASCell(num_units=512) #, state_is_tuple=True)
        cell_2 = tf.contrib.rnn.DropoutWrapper(cell_2, output_keep_prob=self.tf_dropout_keep_prob)
        fw_cell.append(cell_2)
        fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cell)

        bw_cell = []
        cell_1 = tf.contrib.rnn.NASCell(num_units=256) #, state_is_tuple=True)
        cell_1 = tf.contrib.rnn.DropoutWrapper(cell_1, output_keep_prob=self.tf_dropout_keep_prob)
        bw_cell.append(cell_1)
        
        cell_2 = tf.contrib.rnn.NASCell(num_units=256) #, state_is_tuple=True)
        cell_2 = tf.contrib.rnn.DropoutWrapper(cell_2, output_keep_prob=self.tf_dropout_keep_prob)
        bw_cell.append(cell_2)
        bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cell)

        rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x , dtype=tf.float32)
        rnn_outputs = tf.concat([rnn_outputs[0][:, -1], rnn_outputs[1][:, -1]], axis=-1)

#         rnn_outputs, states = tf.nn.dynamic_rnn(fw_cell, rnn_input, dtype=tf.float32)
#         # rnn_outputs = rnn_outputs[:, -1, :]
#         rnn_outputs = tf.reshape(rnn_outputs, (-1, np.prod(rnn_outputs.shape[1:])))
#         print(rnn_outputs)
#         rnn_outputs = tf.layers.batch_normalization(rnn_outputs, training=self.is_training)
        rnn_outputs = tf.layers.dropout(rnn_outputs, rate=self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        print(rnn_outputs)
        
        fc_input = tf.layers.flatten(rnn_outputs)
        fc1 = tf.contrib.layers.fully_connected(fc_input, 1024, activation_fn=tf.nn.relu)
        # fc1 = tf.layers.batch_normalization(fc1, training=self.is_training)
        fc1 = tf.layers.dropout(fc1, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        output = tf.contrib.layers.fully_connected(fc1, 3, activation_fn=None)

        return output
    
    
class CNN_to_RNN(NN):

    def __init__(self, session: tf.Session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay,
                 feature_size_list, name):
        self.feature_size_list = feature_size_list

        super().__init__(session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name)

    def hidden_layers(self, x):
        # x: [None, n_lookback, n_feature]

        def gaussian_noise_layer(input_layer, std):
            """
            마지막 feature인 swell_t-1에 대해 가우시안 노이즈 추가.
            가우시안 노이즈 추가한 후 [0.0~2.0]으로 클리핑한 후 원래 input에 적용한다.
            :param input_layer: a tensor shape of [batch_size, n_lookback, n_feature]
            :param std: stddev of noise
            :return:
            """
            noise_shape = tf.shape(input_layer)[0], tf.shape(input_layer)[1], 1
            noise = tf.truncated_normal(shape=noise_shape, mean=0.0, stddev=std, dtype=tf.float32)
            noised_swell = input_layer[:, :, -1:] + noise
            noised_swell = tf.clip_by_value(noised_swell, 0.0, 2.0)
            input_layer = tf.concat([input_layer[:, :, :-1], noised_swell], axis=-1)
            return input_layer

        noised_x = tf.cond(self.is_training, lambda: gaussian_noise_layer(x, 2.0), lambda: x)

        splited_x_list = tf.split(noised_x, self.feature_size_list+[1], axis=-1)  # 마지막 원소는 swell_t-1
        swell_t_1 = splited_x_list[-1]
        splited_x_list = splited_x_list[:-1]
        print(splited_x_list)

        conv1_list = []
        for splited_x in splited_x_list:
            conv1_filters = 32
            conv1_input = tf.expand_dims(tf.concat([splited_x, swell_t_1], -1), -1)
            conv1 = tf.layers.conv2d(conv1_input,
                                     filters=conv1_filters,
                                     kernel_size=[1, conv1_input.shape[2]],
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
            # conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
            conv1 = tf.layers.dropout(conv1, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)
            conv1_list.append(conv1)  # [-1, n_lookback, 1, conv1_filters]

        print(conv1_list)
        conv1_output = tf.concat(conv1_list, axis=2)  # [-1, n_lookback, n_lacal, conv1_filters]
        print(conv1_output)
        rnn_input = tf.reshape(conv1_output, (-1, conv1_output.shape[1],
                                              conv1_output.shape[2] * conv1_output.shape[3]))

        fw_cell = []
        cell_1 = tf.contrib.rnn.NASCell(num_units=512) #, state_is_tuple=True)
        cell_1 = tf.contrib.rnn.DropoutWrapper(cell_1, output_keep_prob=self.tf_dropout_keep_prob)
        fw_cell.append(cell_1)
        
        cell_2 = tf.contrib.rnn.NASCell(num_units=512) #, state_is_tuple=True)
        cell_2 = tf.contrib.rnn.DropoutWrapper(cell_2, output_keep_prob=self.tf_dropout_keep_prob)
        fw_cell.append(cell_2)
        fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cell)

#         bw_cell = []
#         cell_1 = tf.contrib.rnn.NASCell(num_units=256) #, state_is_tuple=True)
#         cell_1 = tf.contrib.rnn.DropoutWrapper(cell_1, output_keep_prob=self.tf_dropout_keep_prob)
#         bw_cell.append(cell_1)
        
#         cell_2 = tf.contrib.rnn.NASCell(num_units=256) #, state_is_tuple=True)
#         cell_2 = tf.contrib.rnn.DropoutWrapper(cell_2, output_keep_prob=self.tf_dropout_keep_prob)
#         bw_cell.append(cell_2)
#         bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cell)

#         rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, rnn_input, dtype=tf.float32)
#         rnn_outputs = tf.concat([rnn_outputs[0][:, -1], rnn_outputs[1][:, -1]], axis=-1)

        rnn_outputs, states = tf.nn.dynamic_rnn(fw_cell, rnn_input, dtype=tf.float32)
        # rnn_outputs = rnn_outputs[:, -1, :]
        rnn_outputs = tf.reshape(rnn_outputs, (-1, np.prod(rnn_outputs.shape[1:])))
        print(rnn_outputs)
#         rnn_outputs = tf.layers.batch_normalization(rnn_outputs, training=self.is_training)
        rnn_outputs = tf.layers.dropout(rnn_outputs, rate=self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        print(rnn_outputs)
        
        fc_input = tf.layers.flatten(rnn_outputs)
        fc1 = tf.contrib.layers.fully_connected(fc_input, 1024, activation_fn=tf.nn.relu)
        # fc1 = tf.layers.batch_normalization(fc1, training=self.is_training)
        fc1 = tf.layers.dropout(fc1, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        output = tf.contrib.layers.fully_connected(fc1, 3, activation_fn=None)

        return output
