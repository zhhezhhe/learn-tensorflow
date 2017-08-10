OLD_CHECKPOINT_FILE = "/home/store-1-img/zhenghe/chuangxin_data/model_100000/train/model.ckpt-1742043"
NEW_CHECKPOINT_FILE = "/home/store-1-img/zhenghe/chuangxin_data/model_100000/train/model.ckpt-1742043"

import tensorflow as tf
vars_to_rename = {
    "lstm/basic_lstm_cell/weights": "lstm/basic_lstm_cell/kernel",
    "lstm/basic_lstm_cell/biases": "lstm/basic_lstm_cell/bias",
}
new_checkpoint_vars = {}
reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
for old_name in reader.get_variable_to_shape_map():
  if old_name in vars_to_rename:
    new_name = vars_to_rename[old_name]
  else:
    new_name = old_name
  new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
  sess.run(init)
  saver.save(sess, NEW_CHECKPOINT_FILE)
