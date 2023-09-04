import tensorflow.compat.v1 as tf
import numpy as np
from collections import namedtuple
import math

from Model_GATTA import GATTA_Net
from Dataset import Dataset

GATTAModel = namedtuple('GATTAModel', 'X Y train_op loss_op accuracy para lp np')

class Clients_GATTA:
    def __init__(self, input_shape, num_classes, learning_rate, clients_num, local_dist, local_dist_test, me):
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(1)

        self.sess = tf.Session(graph=self.graph)
        self.Merge = me

        # Call the create function to build the computational graph of GATNN
        net = GATTA_Net(input_shape, num_classes, learning_rate, self.graph, self.Merge)
        self.model = GATTAModel(*net)
        self.learning_rate=learning_rate

        # initialize
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        self.dataset = Dataset(tf.keras.datasets.cifar10.load_data,
                               local_dist, local_dist_test,
                        split=clients_num)


    def run_train(self, num, local_old, neig_old):
        with self.graph.as_default():
            # num = batch size
            batch_x, batch_y = self.dataset.train_eval.next_batch(num)
            feed_dict = {
                self.model.X: batch_x,
                self.model.Y: batch_y,
                self.model.lp: local_old,
                self.model.np: neig_old,
            }
        return self.sess.run(self.model.loss_op, feed_dict=feed_dict)

    def run_test(self, num, local_old, neig_old):
        with self.graph.as_default():
            batch_x, batch_y = self.dataset.test_eval.next_batch(num)
            feed_dict = {
                self.model.X: batch_x,
                self.model.Y: batch_y,
                self.model.lp: local_old,
                self.model.np: neig_old,
            }
        return self.sess.run(self.model.loss_op, feed_dict=feed_dict)


    def train_epoch(self, cid, local_old, neig_old, batch_size=128):
        """
            Train one client with its own data for one epoch
            cid: Client id
        """
        dataset = self.dataset.train[cid]
        # batch_size = max(dataset.size//10, 1)
        with self.graph.as_default():
            for step in range(math.ceil(dataset.size / batch_size)):
                if step == math.ceil(dataset.size / batch_size) - 1 and dataset.size % batch_size != 0:
                    batch_x, batch_y = dataset.next_batch(dataset.size % batch_size)
                else:
                    batch_x, batch_y = dataset.next_batch(batch_size)
                feed_dict = {
                    self.model.X: batch_x,
                    self.model.Y: batch_y,
                    self.model.lp: local_old,
                    self.model.np: neig_old
                }
                self.sess.run(self.model.train_op, feed_dict=feed_dict)

    def get_client_vars(self):
        """ Return all of the variables list """
        with self.graph.as_default():
            client_vars = self.sess.run(tf.trainable_variables())
        return client_vars

    def set_global_vars(self, global_vars):
        """ Assign all of the variables with global vars """
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, global_vars):
                variable.load(value, self.sess)

    def get_agnostic_vars(self):
        """ Return the model-agnostic variables list """
        with self.graph.as_default():
            Name_set = [v.name for v in tf.trainable_variables()]
            Name_G = ['fc1_ns', 'GAT1', 'fc2_ns', 'GAT2', 'lamb', 'lamb2']
            Name_Gat = [s for s in Name_set if any(xs in s for xs in Name_G)]
            Name_agn = [ele for ele in Name_set if ele not in Name_Gat]
            Index_ag = [Name_set.index(s) for s in Name_agn]# parameters for model agnostic
            var_agn = self.sess.run(Name_agn)
        return var_agn, Index_ag

    def get_ns_paras(self, local_old, neig_old):
        """ Return the updated local parameters"""
        with self.graph.as_default():
            feed_dict = {
                self.model.lp: local_old,
                self.model.np: neig_old
            }
            par = self.sess.run(self.model.para, feed_dict=feed_dict)
        return par


    def initial_ns(self, shape):
        ini_ns=np.random.normal(0.0, 1.0, shape)
        return ini_ns

    def save_model(self, cid):
        with self.graph.as_default():
            saver = tf.train.Saver()
            if self.Merge==1:
                save_path = saver.save(self.sess, './Model_save/GATTA_model%d' % cid)
            elif self.Merge==2:
                save_path = saver.save(self.sess, './Model_save/GATTA_model%d' % cid)
            print("GATTA Model saved to : ", save_path)

    def restore_model(self, cid):
        with self.graph.as_default():
            saver = tf.train.Saver()
            if self.Merge == 1:
                saver.restore(self.sess, './Model_save/GATTA_model%d' % cid)
            elif self.Merge == 2:
                saver.restore(self.sess, './Model_save/GATTA_model%d' % cid)
            print("GATTA Model loaded %d" % cid)


