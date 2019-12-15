import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

def _cumulativesum(ip, iplength):
    bottom_triangle_ones = tf.constant(
        np.tril(np.ones((iplength, iplength))),
        dtype=tf.float32)
    return tf.reshape(
            tf.matmul(bottom_triangle_ones,
                      tf.reshape(ip, [iplength, 1])),
            [iplength])


def _reverse_cumulativesum(ip, iplength):
    top_triangle_ones = tf.constant(
        np.triu(np.ones((iplength, iplength))),
        dtype=tf.float32)
    return tf.reshape(
            tf.matmul(top_triangle_ones,
                      tf.reshape(ip, [iplength, 1])),
            [iplength])


class RNN(object):

    def __init__(self, embedding_count, embedding_dimension, hidden_dimension,
                 seq_len, begin_token,
                 learningrate=0.01, feedbackreward_gamma=0.9):
        self.embedding_count = embedding_count
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.seq_len = seq_len
        self.begin_token = tf.constant(begin_token, dtype=tf.int32)
        self.learningrate = tf.Variable(float(learningrate), trainable=False)
        self.feedbackreward_gamma = feedbackreward_gamma
        self.gen_parameters = []
        self.dis_parameters = []

        self.estimated_reward = tf.Variable(tf.zeros([self.seq_len]))

        with tf.variable_scope('generator'):
            self.gen_embedding = tf.Variable(self.initialise_matrix([self.embedding_count, self.embedding_dimension]))
            self.gen_parameters.append(self.gen_embedding)
            self.gen_rnn_cell = self.define_rnn_cell(self.gen_parameters)  # hidden_tml to hidden_t generator mapping
            self.gen_output_cell = self.define_output_cell(self.gen_parameters, self.gen_embedding)  # hidden_t to op_t output token logits mapping

        with tf.variable_scope('discriminator'):
            self.discriminator_embedding = tf.Variable(self.initialise_matrix([self.embedding_count, self.embedding_dimension]))
            self.dis_parameters.append(self.discriminator_embedding)
            self.dis_rnn_cell = self.define_rnn_cell(self.dis_parameters)  # hidden_tml to hidden_t discriminator mapping
            self.discriminator_classification_cell = self.define_classification_cell(self.dis_parameters)  # hidden_t to class prediction logit mapping
            self.dis_h0 = tf.Variable(self.initialise_vector([self.hidden_dimension]))
            self.dis_parameters.append(self.dis_h0)

        self.h0 = tf.placeholder(tf.float32, shape=[self.hidden_dimension])  # inception random vectorisation for generator
        self.ip = tf.placeholder(tf.int32, shape=[self.seq_len])  # index sequence of real data, excluding begin token
        self.random_sample = tf.placeholder(tf.float32, shape=[self.seq_len])  

        # generator for initial random selection
        generator_op = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len,
                                             dynamic_size=False, infer_shape=True)
        generator_ip = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.seq_len,
                                             dynamic_size=False, infer_shape=True)
        random_sample = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.seq_len)
        random_sample = random_sample.unstack(self.random_sample)
        def _generator_recurrent(i, ip_t, hidden_tml, generator_op, generator_ip):
            hidden_t = self.gen_rnn_cell(ip_t, hidden_tml)
            op_t = self.gen_output_cell(hidden_t)
            token_sample = random_sample.read(i)
            o_cumulativesum = _cumulativesum(op_t, self.embedding_count)  
            following_token = tf.to_int32(tf.reduce_min(tf.where(token_sample < o_cumulativesum)))  
            ip_tp1 = tf.gather(self.gen_embedding, following_token)
            generator_op = generator_op.write(i, tf.gather(op_t, following_token)) 
            generator_ip = generator_ip.write(i, following_token)  
            return i + 1, ip_tp1, hidden_t, generator_op, generator_ip

        _, _, _, self.generator_op, self.generator_ip = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.seq_len,
            body=_generator_recurrent,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.gather(self.gen_embedding, self.begin_token),
                       self.h0, generator_op, generator_ip))

        # discriminator on synthesized and real data
        discriminator_generated_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.seq_len,
            dynamic_size=False, infer_shape=True)
        discriminator_authentic_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.seq_len,
            dynamic_size=False, infer_shape=True)

        self.generator_ip = self.generator_ip.stack()
        embedding_gen_x = tf.gather(self.discriminator_embedding, self.generator_ip)
        ta_embedding_gen_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.seq_len)
        ta_embedding_gen_x = ta_embedding_gen_x.unstack(embedding_gen_x)

        embedding_real_x = tf.gather(self.discriminator_embedding, self.ip)
        ta_embedding_real_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.seq_len)
        ta_embedding_real_x = ta_embedding_real_x.unstack(embedding_real_x)

        def _dis_recurrent(i, dis_input, hidden_tml, dis_prediction):
            ip_t = dis_input.read(i)
            hidden_t = self.dis_rnn_cell(ip_t, hidden_tml)
            disoutput_t = self.discriminator_classification_cell(hidden_t)
            dis_prediction = dis_prediction.write(i, disoutput_t)
            return i + 1, dis_input, hidden_t, dis_prediction

        _, _, _, self.discriminator_generated_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.seq_len,
            body=_dis_recurrent,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       ta_embedding_gen_x,
                       self.dis_h0,
                       discriminator_generated_predictions))
        self.discriminator_generated_predictions = tf.reshape(
                self.discriminator_generated_predictions.stack(),
                [self.seq_len])

        _, _, _, self.discriminator_authentic_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.seq_len,
            body=_dis_recurrent,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       ta_embedding_real_x,
                       self.dis_h0,
                       discriminator_authentic_predictions))
        self.discriminator_authentic_predictions = tf.reshape(
                self.discriminator_authentic_predictions.stack(),
                [self.seq_len])

        # generator supervised pretraining
        generator_prediction = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.seq_len,
            dynamic_size=False, infer_shape=True)

        embedding_ip = tf.gather(self.gen_embedding, self.ip)
        ta_embedding_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.seq_len)
        ta_embedding_x = ta_embedding_x.unstack(embedding_ip)

        def _pretraining_recurrent(i, ip_t, hidden_tml, generator_prediction):
            hidden_t = self.gen_rnn_cell(ip_t, hidden_tml)
            op_t = self.gen_output_cell(hidden_t)
            generator_prediction = generator_prediction.write(i, op_t)
            ip_tp1 = ta_embedding_x.read(i)
            return i + 1, ip_tp1, hidden_t, generator_prediction

        _, _, _, self.generator_prediction = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.seq_len,
            body=_pretraining_recurrent,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.gather(self.gen_embedding, self.begin_token),
                       self.h0, generator_prediction))

        self.generator_prediction = tf.reshape(
                self.generator_prediction.stack(),
                [self.seq_len, self.embedding_count])

        # discriminator loss calculatation
        self.dis_generated_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.discriminator_generated_predictions, labels=tf.zeros([self.seq_len])))
        self.dis_authentic_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.discriminator_authentic_predictions, labels=tf.ones([self.seq_len])))

        # generator feedbackreward and loss calculatation
        decayloss = tf.exp(tf.log(self.feedbackreward_gamma) * tf.to_float(tf.range(self.seq_len)))
        feedbackreward = _reverse_cumulativesum(decayloss * tf.sigmoid(self.discriminator_generated_predictions),
                                    self.seq_len)
        normalized_feedbackreward = \
            feedbackreward / _reverse_cumulativesum(decayloss, self.seq_len) - self.estimated_reward

        self.feedbackreward_loss = tf.reduce_mean(normalized_feedbackreward ** 2)
        self.generator_loss = \
            -tf.reduce_mean(tf.log(self.generator_op.stack()) * normalized_feedbackreward)

        # pretraining loss
        self.pretraining_loss = \
            (-tf.reduce_sum(
                tf.one_hot(tf.to_int64(self.ip),
                           self.embedding_count, 1.0, 0.0) * tf.log(self.generator_prediction))
             / self.seq_len)

        # training updation
        dis_optimise = self.discriminator_optimiser(self.learningrate)
        gen_optimise = self.generator_optimiser(self.learningrate)
        pretrain_optimise = self.generator_optimiser(self.learningrate)
        feedbackreward_optimise = tf.train.GradientDescentOptimizer(self.learningrate)

        self.dis_gen_gradient = tf.gradients(self.dis_generated_loss, self.dis_parameters)
        self.dis_authentic_gradient = tf.gradients(self.dis_authentic_loss, self.dis_parameters)
        self.dis_gen_updates = dis_optimise.apply_gradients(zip(self.dis_gen_gradient, self.dis_parameters))
        self.dis_authentic_updates = dis_optimise.apply_gradients(zip(self.dis_authentic_gradient, self.dis_parameters))

        self.feedbackreward_gradient = tf.gradients(self.feedbackreward_loss, [self.estimated_reward])
        self.feedbackreward_updates = feedbackreward_optimise.apply_gradients(zip(self.feedbackreward_gradient, [self.estimated_reward]))

        self.generator_gradient = tf.gradients(self.generator_loss, self.gen_parameters)
        self.generator_updates = gen_optimise.apply_gradients(zip(self.generator_gradient, self.gen_parameters))

        self.pretraining_gradient = tf.gradients(self.pretraining_loss, self.gen_parameters)
        self.pretraining_updates = pretrain_optimise.apply_gradients(zip(self.pretraining_gradient, self.gen_parameters))

    def generate(self, session):
        output = session.run(
                [self.generator_ip],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dimension),
                           self.random_sample: np.random.random(self.seq_len)})
        return output[0]

    def training_generator_step(self, session):
        output = session.run(
                [self.generator_updates, self.feedbackreward_updates, self.generator_loss,
                 self.estimated_reward, self.generator_ip],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dimension),
                           self.random_sample: np.random.random(self.seq_len)})
        return output

    def training_dis_generator_step(self, session):
        output = session.run(
                [self.dis_gen_updates, self.dis_generated_loss],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dimension),
                           self.random_sample: np.random.random(self.seq_len)})
        return output

    def training_dis_authentic_step(self, session, ip):
        output = session.run([self.dis_authentic_updates, self.dis_authentic_loss],
                              feed_dict={self.ip: ip})
        return output

    def pretraining_step(self, session, ip):
        output = session.run([self.pretraining_updates, self.pretraining_loss, self.generator_prediction],
                              feed_dict={self.ip: ip,
                                         self.h0: np.random.normal(size=self.hidden_dimension)})
        return output

    def initialise_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def initialise_vector(self, shape):
        return tf.zeros(shape)

    def define_rnn_cell(self, params):
        self.weight_rec = tf.Variable(self.initialise_matrix([self.hidden_dimension, self.embedding_dimension]))
        params.append(self.weight_rec)
        def singleunit(ip_t, hidden_tml):
            return hidden_tml + tf.reshape(tf.matmul(self.weight_rec, tf.reshape(ip_t, [self.embedding_dimension, 1])), [self.hidden_dimension])
        return singleunit

    def define_output_cell(self, params, embeddings):
        self.Weight_out = tf.Variable(self.initialise_matrix([self.embedding_dimension, self.hidden_dimension]))
        self.bias_out1 = tf.Variable(self.initialise_vector([self.embedding_dimension, 1]))
        self.bias_out2 = tf.Variable(self.initialise_vector([self.embedding_count, 1]))
        params.extend([self.Weight_out, self.bias_out1, self.bias_out2])
        def singleunit(hidden_t):
            logits = tf.reshape(
                    self.bias_out2 +
                    tf.matmul(embeddings,
                              tf.tanh(self.bias_out1 +
                                      tf.matmul(self.Weight_out, tf.reshape(hidden_t, [self.hidden_dimension, 1])))),
                    [1, self.embedding_count])
            return tf.reshape(tf.nn.softmax(logits), [self.embedding_count])
        return singleunit

    def define_classification_cell(self, params):
        self.weight_class = tf.Variable(self.initialise_matrix([1, self.hidden_dimension]))
        self.bias_class = tf.Variable(self.initialise_vector([1]))
        params.extend([self.weight_class, self.bias_class])
        def singleunit(hidden_t):
            return self.bias_class + tf.matmul(self.weight_class, tf.reshape(hidden_t, [self.hidden_dimension, 1]))
        return singleunit

    def discriminator_optimiser(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)

    def generator_optimiser(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)

class GRU(RNN):

    def define_rnn_cell(self, params):
        self.weight_rx = tf.Variable(self.initialise_matrix([self.hidden_dimension, self.embedding_dimension]))
        self.weight_zx = tf.Variable(self.initialise_matrix([self.hidden_dimension, self.embedding_dimension]))
        self.weight_hx = tf.Variable(self.initialise_matrix([self.hidden_dimension, self.embedding_dimension]))
        self.update_rh = tf.Variable(self.initialise_matrix([self.hidden_dimension, self.hidden_dimension]))
        self.unit_zh = tf.Variable(self.initialise_matrix([self.hidden_dimension, self.hidden_dimension]))
        self.unit_hh = tf.Variable(self.initialise_matrix([self.hidden_dimension, self.hidden_dimension]))
        params.extend([
            self.weight_rx, self.weight_zx, self.weight_hx,
            self.update_rh, self.unit_zh, self.unit_hh])

        def singleunit(ip_t, hidden_tml):
            ip_t = tf.reshape(ip_t, [self.embedding_dimension, 1])
            hidden_tml = tf.reshape(hidden_tml, [self.hidden_dimension, 1])
            reset_vector = tf.sigmoid(tf.matmul(self.weight_rx, ip_t) + tf.matmul(self.update_rh, hidden_tml))
            update_gate_vector = tf.sigmoid(tf.matmul(self.weight_zx, ip_t) + tf.matmul(self.unit_zh, hidden_tml))
            output_vector_tilde = tf.tanh(tf.matmul(self.weight_hx, ip_t) + tf.matmul(self.unit_hh, reset_vector * hidden_tml))
            hidden_t = (1 - update_gate_vector) * hidden_tml + update_gate_vector * output_vector_tilde
            return tf.reshape(hidden_t, [self.hidden_dimension])

        return singleunit
