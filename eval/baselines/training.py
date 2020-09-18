import eval.baselines.preprocessing as preprocessing
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import xgboost as xgb
from eval.baselines.data_generator import DataGenerator


def train(log, indices, mapping_activities, max_seq_length, predictor, fold, name_of_activity):
    """
    Train model.
    :param name_of_activity:
    :param fold:
    :param predictor:
    :param max_seq_length:
    :param mapping_activities:
    :param log:
    :param indices:
    :return:
    """

    sequences = preprocessing.sequences_from_log_and_indices(log, indices)


    # learning model
    if predictor == "BiLSTM":


        ids = preprocessing.create_index_from_sequences(sequences)
        data_ids, validation_ids, _, _ = preprocessing.split_train_test(ids)

        training_generator = DataGenerator(preprocessing, data_ids, sequences, mapping_activities, max_seq_length, name_of_activity)
        validation_generator = DataGenerator(preprocessing, validation_ids, sequences, mapping_activities, max_seq_length, name_of_activity)

        num_epochs = 100

        # input layer
        main_input = tf.keras.layers.Input(shape=(max_seq_length, len(mapping_activities.keys())), name='main_input')

        # hidden layer
        b1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(100, use_bias=True,
                                 implementation=1,
                                 activation="tanh",
                                 kernel_initializer='glorot_uniform',
                                 return_sequences=False, dropout=0.2))(main_input)

        # output layer
        out_output = tf.keras.layers.Dense(2, activation='softmax', name='out_output',
                                           kernel_initializer='glorot_uniform')(b1)

        optimizer = tf.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                              schedule_decay=0.004, clipvalue=3)

        model = tf.keras.models.Model(inputs=[main_input], outputs=[out_output])
        model.compile(loss={'out_output': 'categorical_crossentropy'}, optimizer=optimizer)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('../../model/%s_%s.h5' % (predictor, fold),
                                                              monitor='val_loss',
                                                              verbose=0,
                                                              save_best_only=True,
                                                              save_weights_only=False,
                                                              mode='auto')
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                          mode='auto',
                                                          min_delta=0.0001, cooldown=0, min_lr=0)
        model.summary()

        model.fit(training_generator,  verbose=1, validation_data=validation_generator,  # validation_split=0.10,  {'out_output': label_tensor}
                  callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=128, epochs=num_epochs)

    elif predictor == "RF":

        data_tensor = preprocessing.data_tensor_from_sequences(sequences, mapping_activities, max_seq_length, name_of_activity)
        label_tensor = preprocessing.label_tensor_from_sequences(sequences)

        data_tensor = data_tensor.reshape(-1, max_seq_length * len(mapping_activities.keys()))
        model = RandomForestClassifier(n_jobs=-1,  # use all processors
                                       random_state=0,
                                       n_estimators=100,  # default value
                                       criterion="gini",  # default value
                                       max_depth=None,  # default value
                                       min_samples_split=2,  # default value
                                       min_samples_leaf=1,  # default value
                                       min_weight_fraction_leaf=0.0,  # default value
                                       max_features="auto",  # default value
                                       max_leaf_nodes=None,  # default value
                                       min_impurity_decrease=0.0,  # default value
                                       bootstrap=True,  # default value
                                       oob_score=False,  # default value
                                       warm_start=False,  # default value
                                       class_weight=None)  # default value

        labels = preprocessing.label_sequence_from_tensor(label_tensor)
        model.fit(data_tensor, labels)
        dump(model, '../../model/%s_%s.joblib' % (predictor, fold))

    elif predictor == "XG":

        data_tensor = preprocessing.data_tensor_from_sequences(sequences, mapping_activities, max_seq_length, name_of_activity)
        label_tensor = preprocessing.label_tensor_from_sequences(sequences)

        labels = preprocessing.label_sequence_from_tensor(label_tensor)
        d_train = xgb.DMatrix(data_tensor.reshape(-1, max_seq_length * len(mapping_activities.keys())), label=labels)

        param = {'random_state': 0,
                 'booster': 'gbtree',
                 'eta': 0.3,  # default value; learning rate
                 'gamma': 0,  # default value; min_split_loss
                 'max_depth': 6}  # default value
        steps = 100

        model = xgb.train(param, d_train, steps)

        dump(model, '../../model/%s_%s.joblib' % (predictor, fold))
