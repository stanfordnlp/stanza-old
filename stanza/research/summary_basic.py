import tensorflow as tf

__author__ = 'kelvinguu'

class TensorBoardLogger(object):
    """Log scalars to event files that can then be read by TensorBoard.

    This object keeps its own TF Graph, and creates a Variable on the fly
    for every metric you want to log.

    This can be easily extended to log other kinds of summary events.

    @wmonroe has a version that doesn't rely so heavily on the TF library.
    See summary.py
    """

    def __init__(self, log_dir):
        self.g = tf.Graph()
        self.summaries = {}
        self.sess = tf.Session(graph=self.g)
        self.summ_writer = tf.train.SummaryWriter(log_dir, flush_secs=5)

    def log_proto(self, proto, step_num):
        """Log a Summary protobuf to the event file.

        :param proto:  a Summary protobuf
        :param step_num: the iteration number at which this value was logged
        """
        self.summ_writer.add_summary(proto, step_num)
        return proto

    def log(self, key, val, step_num):
        """Directly log a scalar value to the event file.

        :param string key: a name for the value
        :param val: a float
        :param step_num: the iteration number at which this value was logged
        """
        try:
            ph, summ = self.summaries[key]
        except KeyError:
            # if we haven't defined a variable for this key, define one
            with self.g.as_default():
                ph = tf.placeholder(tf.float32, (), name=key)  # scalar
                summ = tf.scalar_summary(key, ph)
            self.summaries[key] = (ph, summ)

        summary_str = self.sess.run(summ, {ph: val})
        self.summ_writer.add_summary(summary_str, step_num)
        return val
