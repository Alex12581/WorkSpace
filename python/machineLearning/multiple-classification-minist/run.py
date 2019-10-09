# -*- "coding: utf-8" -*-

'''
用mnist数据集做多分类，用于测试Recall和F1指标的计算代码
'''

import keras
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_data(path="./mnist.pkl"):
    with open("mnist.pkl", "rb") as f:
        data = pickle._Unpickler(f)
        data.encoding='latin1'
        train, val, test = data.load()
    # train is a tuple, train[0] is x_train, train[1] is y_train
    return train[0], train[1], val[0], val[1], test[0], test[1]

def evaluation(origin, predict):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for ori, pre in zip(origin, predict):
        if ori == 1:
            if pre == 1:
                TP += 1
            elif pre == 0:
                FP += 1
        elif ori == 0:
            if pre == 1:
                FN += 1
            elif pre == 0:
                TN += 1
    # precision, recall, f1
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    if TP+FP == 0:
        precision = 0
    else:
        precision = TP/(TP+FP)  # 在所有实际为1中，预测为1的比例
    recall = TP/(TP+FN)  # 在所有预测为1中，实际为1的比例
    if precision+recall == 0:
        f1 = 0
    else:
        f1 = (2*precision*recall)/(precision+recall)

    return [TP, TN, FP, FN], accuracy, precision, recall, f1

def train():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    # onehot encode
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_val = keras.utils.to_categorical(y_val, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    print("x_train:{}, y_train:{}, x_val:{}, y_val:{}, x_test:{}, y_test:{}".format(
        x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape))

    # model
    model = keras.Sequential([
        keras.layers.Dense(256, input_shape=(784,), activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=30, verbose=2).history

    model.save("model.h5")
    with open("history-adam-30epochs.pkl", "wb") as f:
        pickle.dump(history, f, -1)

def plot():
    with open("history-adam-30epochs.pkl", "rb") as f:
        history = pickle.load(f)
    # 可视化
    plt.figure()
    plt.subplot("211")
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.subplot("212")
    plt.plot(history["acc"], label="acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend()
    plt.savefig("history_figure.png")

def predict():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    model = keras.models.load_model("model.h5")
    result = model.predict(x_test)
    result = np.argmax(result, axis=1)
    _, accuracy, precision, recall, f1 = evaluation(y_test, result)
    print("accuracy:{}, precision:{}, recall:{}, f1:{}".format(accuracy, precision, recall, f1))

if __name__ == "__main__":
    # train()
    # plot()
    # predict()
