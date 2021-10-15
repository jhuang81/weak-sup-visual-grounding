import tensorflow as tf

class Encoder:
    """LSTM Encoder."""

    def __init__(self, rnn_hidden_size, num_rnn_steps):
        self.rnn_hidden_size = rnn_hidden_size
        self.num_rnn_steps = num_rnn_steps
        self.rnn = tf.nn.rnn_cell.LSTMCell(self.rnn_hidden_size)

    def encode(self, phrases, lstm_mask):
        # phrases  16kb * 10 * 300
        # lstm_mask 16kb * 10
        batch_size = tf.shape(phrases)[0]
        with tf.variable_scope('encoder') as scope:
            output_embed = tf.zeros([batch_size, self.rnn_hidden_size])  # 16kb * 300
            state = self.rnn.zero_state(batch_size, tf.float32)
            with tf.variable_scope('rnn') as scope_rnn:
                for step in range(self.num_rnn_steps):
                    if step > 0:
                        scope_rnn.reuse_variables()
                    hidden, state = self.rnn(phrases[:, step, :], state, scope=scope_rnn)  # 16kb * 300
                    # Max pool all the steps/tokens.
                    # If mask is True for ith example in the batch, update the max hidden value.
                    if step == 0:
                        output_embed = output_embed + hidden
                    else:
                        output_embed = tf.where(lstm_mask[:, step], tf.maximum(output_embed, hidden), output_embed)
        return output_embed
