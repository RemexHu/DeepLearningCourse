import tensorflow as tf
#from imageio import imread
from scipy.misc import imread
import csv
import cifar_utils
import numpy as np




def predict(X_test,model_name):
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph('model/%s'%model_name)
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        graph = tf.get_default_graph()
        #tf_model,tf_input,tf_output = load_graph(graph)
        #idx = 0
        tf_input = graph.get_operations()[0].name+':0'
        print("tf_input:",tf_input)
        print(graph.get_operations())
        x = graph.get_tensor_by_name(tf_input)
        y = graph.get_operation_by_name('evaluate/ArgMax').outputs[0]

        # Make prediciton
        y_out = sess.run(y, feed_dict={x: X_test})
    return y_out


X_test = []
for pic in range(1, 5001):
    pic_idx = str(pic)
    arr = imread('/home/runchen/Downloads/test/0%s.png'%(pic_idx))
    X_test.append(arr[::,::])


X_test = np.asarray(X_test)
X_test.flatten()


X = []

for i in range(10):
    X.append(X_test[i * 500: (i + 1) * 500])


for X_test_batch in X:

    mean_image = np.mean(X_test_batch, axis=0)
    X_test_batch = X_test_batch.astype(np.float32) - mean_image.astype(np.float32)
    X_test_batch /= np.std(X_test_batch, axis=0)

    X_test_batch = X_test_batch.reshape([-1,32,32,3])


tf.reset_default_graph()



def batch_test(X_test, batch_size = 500):
    result = set()
    for i in range(10):
        result.add(predict(X_test[i * batch_size: (i + 1) * batch_size, :, :, :], "lenet_1519359862.meta"))
    return result

kaggle_result = []

for X_test_batch in X:
    kaggle_result.append(predict(X_test_batch, "runchennet_1519449834.meta"))


#kaggle_result = batch_test(X_test, 500)

test_sheet = {0: 'airplane', 1:'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
       7: 'horse', 8: 'ship', 9: 'truck'}


with open('predicted.csv','w',newline='') as csvfile:
    fieldnames = ['Id','label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for index_0, result in enumerate(kaggle_result):
        for index_1,label in enumerate(result):
            filename = index_0 * 500 + index_1
            writer.writerow({'Id': filename, 'label': test_sheet[label]})




