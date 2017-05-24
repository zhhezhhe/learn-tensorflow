import pandas as pd
import numpy as np
import tensorflow as tf

def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}
    # feature_cols = continuous_cols
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

def my_linear_classfier1():
    headers, x_df_train, x_df_test, y_df_train, y_df_test, data_emedding, label = random_input()
    wide_columns = tf.contrib.learn.infer_real_valued_columns_from_input(data_emedding)
    # model_dir = "/media/zh/E/test/Linear_model/"
    m = tf.contrib.learn.LinearClassifier(feature_columns=wide_columns,
                                          optimizer=tf.train.FtrlOptimizer(
                                                      learning_rate=0.1,
                                                      l1_regularization_strength=1.0,
                                                      l2_regularization_strength=1.0)
                                                  # model_dir=model_dir
                                          )
    m.fit(data_emedding, label, batch_size=5, steps=200)
    results = m.evaluate(data_emedding, label, batch_size=5, steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

def random_input():
    sample_num = 10000
    np.random.seed([3, 1415])
    data_emedding = np.random.random([sample_num, 128])
    data_emedding = data_emedding.astype(np.float32)
    headers = ["feature_" + str(i) for i in range(128)]
    label = np.random.choice((0, 1), ([sample_num,]))
    _x_df = pd.DataFrame(data=data_emedding, columns=headers)
    _y_df = pd.DataFrame(dict(label=label))
    trainBegin, trainEnd = 0, int(0.8 * sample_num)
    testBegin, testEnd = int(0.8 * sample_num), sample_num
    x_df_train = _x_df.iloc[trainBegin:trainEnd, :]
    x_df_test = _x_df.iloc[testBegin:testEnd, :]
    y_df_train = _y_df.iloc[trainBegin:trainEnd, :]
    y_df_test = _y_df.iloc[testBegin:testEnd, :]
    return headers, x_df_train, x_df_test, y_df_train, y_df_test, data_emedding, label

def embedding_input():
    return


def main():
    # my_linear_classfier1()
    feature_columns = []
    for i in COLUMNS:
        feature_columns.append(tf.contrib.layers.real_valued_column(i))
    # model_dir = "/media/zh/E/test/Linear_model/"
    m = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns,
        optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=1.0,
            l2_regularization_strength=1.0)
        # model_dir=model_dir
                                          )
    m.fit(input_fn=train_input_fn, steps=200)
    results = m.evaluate(input_fn=eval_input_fn, steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
if __name__ == "__main__":
    COLUMNS,  x_df_train, x_df_test, y_df_train, y_df_test, data_emedding, label = random_input()
    df_train = x_df_train
    df_test = x_df_test
    LABEL_COLUMN = "label"
    df_train[LABEL_COLUMN] = y_df_train.astype(int)
    df_test[LABEL_COLUMN] = y_df_test.astype(int)
    CATEGORICAL_COLUMNS = []
    CONTINUOUS_COLUMNS = COLUMNS
    main()