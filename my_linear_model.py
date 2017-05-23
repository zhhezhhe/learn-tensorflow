import pandas as pd
import numpy as np
import tensorflow as tf
sample_num = 10000
np.random.seed([3,1415])
data_emedding = np.random.random([sample_num,128])
data_emedding = data_emedding.astype(np.float32)
headers = [ "feature_" + str(i) for i in range(128)]
label = np.random.choice((0,1), (sample_num,))
_x_df = pd.DataFrame(data=data_emedding, columns=headers)
_y_df = pd.DataFrame(dict(label=label))
trainBegin, trainEnd = 0, int(0.8*sample_num)
testBegin, testEnd = int(0.8*sample_num), sample_num
x_df_train = _x_df.iloc[trainBegin:trainEnd, :]
x_df_test = _x_df.iloc[testBegin:testEnd, :]
y_df_train = _y_df.iloc[trainBegin:trainEnd, :]
y_df_test = _y_df.iloc[testBegin:testEnd, :]

wide_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_df_train)

m = tf.contrib.learn.LinearClassifier(feature_columns=wide_columns)
m.fit(x_df_train, y_df_train, batch_size=5, steps=200)

results = m.evaluate(x_df_test, y_df_test, batch_size=5, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))