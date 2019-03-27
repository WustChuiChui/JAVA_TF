import tensorflow as tf

x = tf.placeholder(tf.float32, [1, None], name="x")
y = tf.placeholder(tf.float32, [1, None], name="y")
z = tf.add(x, y, name="z")
s = tf.reduce_sum(z, name="s")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print([n.name for n in sess.graph.as_graph_def().node])
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph_def,
                    output_node_names=["z", "s"])
    with open('model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

