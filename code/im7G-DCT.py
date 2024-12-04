# 引入库文件
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.layers import Input, Dropout, Dense, Activation, Concatenate, Add, Conv2D, AveragePooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, multiply, Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from keras.models import Model, load_model,Sequential
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer,Embedding,Bidirectional,GRU,MultiHeadAttention,LayerNormalization
from keras import initializers
from sklearn.model_selection import KFold
# 忽略提醒
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")




def read_fasta(fasta_file_name):
    seqs = []
    seqs_num = 0
    file = open(fasta_file_name)

    for line in file.readlines():
        if line.strip() == '':
            continue

        if line.startswith('>'):
            seqs_num = seqs_num + 1
            continue
        else:
            seq = line.strip()

            result1 = 'N' in seq
            result2 = 'n' in seq
            if result1 == False and result2 == False:
                seqs.append(seq)
    return seqs


# one-hot
def to_one_hot(seqs):
    base_dict = {
        'a': 0, 'c': 1, 'g': 2, 't': 3,
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
    }

    one_hot_4_seqs = []
    for seq in seqs:

        one_hot_matrix = np.zeros([4, len(seq)], dtype=float)
        index = 0
        for seq_base in seq:
            one_hot_matrix[base_dict[seq_base], index] = 1
            index = index + 1

        one_hot_4_seqs.append(one_hot_matrix)
    return one_hot_4_seqs


# NCP+ND
def to_properties_density_code(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([4, len(seq)], dtype=float)
        A_num = 0
        C_num = 0
        G_num = 0
        T_num = 0
        All_num = 0
        for seq_base in seq:
            if seq_base == "A":
                All_num += 1
                A_num += 1
                Density = A_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "C":
                All_num += 1
                C_num += 1
                Density = C_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "G":
                All_num += 1
                G_num += 1
                Density = G_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "T":
                All_num += 1
                T_num += 1
                Density = T_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]

        properties_code.append(properties_matrix)
    return properties_code


def show_performance(y_true, y_pred):
    # 定义tp, fp, tn, fn初始值
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    # 计算敏感性Sn
    Sn = TP / (TP + FN + 1e-06)
    # 计算特异性Sp
    Sp = TN / (FP + TN + 1e-06)
    # 计算Acc值
    Acc = (TP + TN) / len(y_true)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Sn, Sp, Acc, MCC

def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))


def conv_factory(x, filters, dropout_rate, weight_decay=1e-4):
    x = Activation('elu')(x)
    x = Conv2D(filters=filters,
               kernel_size=(3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    return x


def transition(x, filters, dropout_rate, weight_decay=1e-4):
    x = Activation('elu')(x)
    x = Conv2D(filters=filters,
               kernel_size=(1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    return x


def denseblock(x, layers, filters, growth_rate, dropout_rate=None, weight_decay=1e-4):
    list_feature_map = [x]
    for i in range(layers):
        x = conv_factory(x, growth_rate,
                         dropout_rate, weight_decay)

        list_feature_map.append(x)
        x = Concatenate(axis=-1)(list_feature_map)
        filters = filters + growth_rate
    return x, filters


def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(
        input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(
        input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])



def spatial_attention(input_feature):
    kernel_size = 7

    channel = input_feature.shape[-1]
    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    # assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    # assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    # assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])


# Cbam
def cbam_block(cbam_feature, ratio=8):
    channel_feature = channel_attention(cbam_feature, ratio)
    spatial_feature = spatial_attention(channel_feature)
    return spatial_feature


class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        # Multi-Head Attention层
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        # Feed Forward层
        self.dense_proj = Sequential(
            [Dense(dense_dim, activation="relu"),
             Dense(embed_dim),]
        )
        # Add&Norm层1
        self.layernorm_1 = LayerNormalization()
        # Add&Norm层2
        self.layernorm_2 = LayerNormalization()

    def call(self, inputs):
        # 首先经过Multi-Head Attention层
        attention_output = self.attention(inputs, inputs)
        # 残差连接+层规范化
        proj_input = self.layernorm_1(inputs + attention_output)
        # 前馈层
        proj_output = self.dense_proj(proj_input)
        # 残差连接+层规范化
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


def build_model(windows=501, denseblocks=4, layers=3, filters=96,
                growth_rate=32, dropout_rate=0.2, weight_decay=1e-4):
    input_1 = Input(shape=(8, windows, 1))


    x_1 = input_1

    # Add denseblock
    for i in range(denseblocks - 1):
        # Add denseblock
        x_1, filters_1 = denseblock(x_1, layers=layers,
                                    filters=filters, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        x_1 = BatchNormalization(axis=-1)(x_1)

        x_1 = cbam_block(x_1)


        # Add transition
        x_1 = transition(x_1, filters=filters_1,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last denseblock
    # Add denseblock

    x_1, filters_1 = denseblock(x_1, layers=layers,
                                filters=filters, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)

    x_1 = BatchNormalization(axis=-1)(x_1)



    x_2 = K.squeeze(input_1, -1)

    x_2 = TransformerEncoder(501, 2, 128)(x_2)


    # 展平成向量
    x_1 = Flatten()(x_1)
    x_2 = Flatten()(x_2)

    # 添加全连接层进行预测

    x_1 = Dense(units=240, activation="sigmoid", use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay))(x_1)
    x_1 = Dropout(0.5)(x_1)

    x_2 = Dense(units=240, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x_2)
    x_2 = Dropout(0.5)(x_2)

    x = Concatenate()([x_1, x_2])

    # relu sigmoid
    x = Dense(units=40, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    x = Dropout(0.5)(x)

    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    inputs = [input_1]
    outputs = [x]

    model = Model(inputs=inputs, outputs=outputs, name="enhancer")

    optimizer = Adam(learning_rate=1e-4, epsilon=1e-8)


    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':

    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility

    train_seqs = np.array(read_fasta('../dataset/train.fasta'))

    print("train_seqs:",train_seqs.shape)
    train_onehot = np.array(to_one_hot(train_seqs)).astype(np.float32)
    train_properties_code = np.array(to_properties_density_code(train_seqs)).astype(np.float32)

    train = np.concatenate((train_onehot, train_properties_code), axis=1)

    train_label = np.array([1] * 3292 + [0] * 3292).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)

    valid_seqs = np.array(read_fasta('../dataset/valid.fasta'))

    valid_onehot = np.array(to_one_hot(valid_seqs)).astype(np.float32)
    valid_properties_code = np.array(to_properties_density_code(valid_seqs)).astype(np.float32)

    valid = np.concatenate((valid_onehot, valid_properties_code), axis=1)

    valid_label = np.array([1] * 1097 + [0] * 1097).astype(np.float32)
    valid_label = to_categorical(valid_label, num_classes=2)



    test_seqs = np.array(read_fasta('../dataset/test.fasta'))

    test_onehot = np.array(to_one_hot(test_seqs)).astype(np.float32)
    test_properties_code = np.array(to_properties_density_code(test_seqs)).astype(np.float32)

    test = np.concatenate((test_onehot, test_properties_code), axis=1)

    test_label = np.array([1] * 1097 + [0] * 1097).astype(np.float32)
    test_label = to_categorical(test_label, num_classes=2)


    model = build_model()

    BATCH_SIZE = 30
    EPOCHS = 300

    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)


    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    all_performance = []
    for fold_count, (train_index, val_index) in enumerate(k_fold.split(train)):
        print('*' * 30 + ' fold ' + str(fold_count + 1) + ' ' + '*' * 30)
        tra, val = train[train_index], train[val_index]
        tra_label, val_label = train_label[train_index], train_label[val_index]


        model.fit(x=tra, y=tra_label, validation_data=(val, val_label), epochs=EPOCHS,
                   batch_size=BATCH_SIZE, shuffle=True,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=30, mode='auto')],
                   verbose=1)

        model.save('../models/model_' + str(fold_count + 1) + '.h5')

        del model

        model = load_model('../models/model_' + str(fold_count + 1) + '.h5', custom_objects={'TransformerEncoder': TransformerEncoder})


        valid_score = model.predict(valid)


        Sn, Sp, Acc, MCC = show_performance(valid_label[:, 1], valid_score[:, 1])
        AUC = roc_auc_score(valid_label[:, 1], valid_score[:, 1])
        print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))

        performance = [Sn, Sp, Acc, MCC, AUC]
        all_performance.append(performance)

        '''Mapping the ROC'''
        fpr, tpr, thresholds = roc_curve(valid_label[:, 1], valid_score[:, 1], pos_label=1)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, label='ROC fold {} (AUC={:.4f})'.format(str(fold_count + 1), AUC))


    fold_count += 1
    all_performance = np.array(all_performance)
    print('5 fold result:', all_performance)
    performance_mean = performance_mean(all_performance)

    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(np.array(all_performance)[:, 4])
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.4f)' % (mean_auc), lw=2, alpha=.8)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/5fold_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()

    model.fit(x=train, y=train_label, validation_data=(valid, valid_label), epochs=EPOCHS,
                      batch_size=BATCH_SIZE, shuffle=True,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=30, mode='auto')],
                      verbose=1)

    model.save('../models/model_test.h5')

    del model


    model = load_model('../models/model_test.h5', custom_objects={'TransformerEncoder': TransformerEncoder})


    test_score = model.predict(test)


    Sn, Sp, Acc, MCC = show_performance(test_label[:, 1], test_score[:, 1])
    AUC = roc_auc_score(test_label[:, 1], test_score[:, 1])

    print('-----------------------------------------------test---------------------------------------')
    print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))

    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    test_fpr, test_tpr, thresholds = roc_curve(test_label[:,1], test_score[:,1], pos_label=1)

    plt.plot(test_fpr, test_tpr, color='b', label=r'test ROC (AUC=%0.4f)' % (AUC), lw=2, alpha=.8)

    plt.title('ROC Curve OF')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/test_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()




