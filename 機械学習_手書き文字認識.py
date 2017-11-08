# 0. 必要なモジュールを読み込む
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# MNISTデータの取得
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 1. 学習したいモデルを記述する
# 入力変数と出力変数のプレースホルダを生成
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# モデルパラメータ(入力層:784ノード, 隠れ層:100ノード, 出力層:10ノード)
W1 = tf.Variable(tf.truncated_normal([784, 100]))
b1 = tf.Variable(tf.zeros([100]))
W2 = tf.Variable(tf.truncated_normal([100, 10]))
b2 = tf.Variable(tf.zeros([10]))

# モデル式
h = tf.sigmoid(tf.matmul(x, W1) + b1) # 入力層->隠れ層
u = tf.matmul(h, W2) + b2 # 隠れ層->出力層 (ロジット)
y = tf.nn.softmax(u) # 隠れ層->出力層 (ソフトマックス後)

# 2. 学習やテストに必要な関数を定義する
# 誤差関数(loss)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(u, y_))

# 最適化手段(最急降下法)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 正答率(学習には用いない)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 3. 実際に学習処理を実行する
# (1) セッションを準備し，変数を初期化
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# (2) バッチ型確率的勾配降下法でパラメータ更新
for i in range(10000):
    # 訓練データから100サンプルをランダムに取得
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 学習
    _, l = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
    if (i + 1) % 1000 == 0:
     print("step=%3d, loss=%.2f" % (i + 1, l))

# 4. テスト用データに対して予測し，性能を確認
# (1) テスト用データを1000サンプル取得
new_x = mnist.test.images[0:1000]
new_y_ = mnist.test.labels[0:1000]

# (2) 予測と性能評価
accuracy, new_y = sess.run([acc, y], feed_dict={x:new_x , y_:new_y_ })
print("Accuracy (for test data): %6.2f%%" % accuracy)
print("True Label:", np.argmax(new_y_[0:15,], 1))
print("Est Label:", np.argmax(new_y[0:15, ], 1))

# 5. 後片付け
# セッションを閉じる
sess.close()

