import tensorflow as tf
from dnc.dnc import *
from recurrent_controller import *

graph = tf.Graph()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

graph_nodes = None
nodes = {}

with graph.as_default():
    with tf.Session(graph=graph, config=config) as session:

        ncomputer = DNCDuo(
            MemRNNController,
            2048,
            512,
            100,
            256,
            256,
            4,
            1,
            testing=False,
            output_feedback=True
        )

        # tf.train.Saver(tf.trainable_variables()).restore(session, os.path.join('checkpoint', 'model.ckpt'))
        graph_nodes = graph.as_graph_def().node

        for node in graph_nodes:
            nodes[node.name] = {
                'in': node.input,
                'out': []
            }

        for node in graph_nodes:
            for innode in node.input:
                try:
                    nodes[innode]['out'].append(innode)
                except KeyError:
                    print(innode, " Not found")
