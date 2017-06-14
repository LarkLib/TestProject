# -*- coding: UTF-8 -*-
'''Test tensorflow of python api'''

import os
import platform
import functools as funtools

import tensorflow as tf
import numpy as np
import matplotlib.cm as cm  # install pillow to support jpg format image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d  # incuded in matplotlib
import matplotlib.animation as animation
import psutil
import cv2  # install opencv-python
# http://www.lfd.uci.edu/~gohlke/pythonlibs/
import scipy.signal as signal
import scipy.spatial as sptial
import sklearn.datasets as skdatasets
import sklearn.preprocessing as skpreprocessing
import sklearn.model_selection as skmodelselection
import urllib3
import pandas as pd
import quandl
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_something():
    # https://docs.python.org/3.5/library/os.path.html
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, '..', 'test')
    print(os.getcwd())

    # First, load the image
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    filename = os.path.join(dir_path, "MarshOrchid.jpg")
    print(filename)
    # Print out its shape
    image = mpimg.imread(filename)
    height, width, depth = image.shape
    print(image.shape)
    # plt.imshow(image)
    # plt.show()

    x = tf.Variable(image, name='x')
    model = tf.global_variables_initializer()

    with tf.Session() as session:
        x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
        session.run(model)
        result = session.run(x)
        print(result.shape)
        # plt.imshow(result)
        # plt.show()

    with tf.Session() as session:
        x = tf.transpose(x, perm=[1, 0, 2])
        session.run(model)
        result = session.run(x)
        # plt.imshow(result)
        # plt.show()

    del x
    x = tf.placeholder("float", None)
    y = x * 2
    with tf.Session() as session:
        result = session.run(y, feed_dict={x: [1, 2, 3, 4, 5, 6]})
        print(y)

    raw_image_data = mpimg.imread(filename)
    image = tf.placeholder("uint8", [None, None, 3])
    slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1])
    with tf.Session() as session:
        result = session.run(slice, feed_dict={image: raw_image_data})
        print(result.shape)
        # plt.imshow(result)
        # plt.show()


def test_image():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, '..', 'test')
    # First, load the image
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    filename = os.path.join(dir_path, "MarshOrchid.jpg")
    image = mpimg.imread(filename)
    plt.imshow(image)
    plt.show()
    image_cv2 = cv2.imread(filename)
    plt.imshow(image_cv2)
    plt.show()
    plt.imshow(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    plt.show()
    image_s = image[:, :, 0]
    plt.imshow(image_s)
    plt.colorbar()
    plt.show()
    plt.imshow(image_s, cmap="hot")
    plt.show()
    plt.imshow(image_s)
    plt.set_cmap("nipy_spectral")
    plt.show()
    plt.hist(image_s.flatten(), bins=256, range=(0.0, 1.0), fc="k", ec="k")
    plt.show()
    # plt.imshow(image_s)
    plt.clim(0.0, 0.7)
    plt.show()


def get_system_info():
    print(platform.uname())
    print(platform.platform(), platform.processor(), platform.node(), platform.version())
    mem_info = psutil.virtual_memory()
    print("{} Kb".format(mem_info.available))


def test_interactive_tensorFlow():
    session = tf.InteractiveSession()
    x = tf.constant(list(range(10)))
    print(x.eval())
    print("{} Kb".format(psutil.virtual_memory().available))
    X = tf.constant(np.eye(10))
    Y = tf.constant(np.random.randn(10, 30))
    print("{} Kb".format(psutil.virtual_memory().available))
    Z = tf.matmul(X, Y)
    Z.eval()
    print("{} Kb".format(psutil.virtual_memory().available))
    session.close()


def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))
        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor / 2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    # Create a big "samples" dataset
    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')
    return centroids, samples


def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    # Plot out the different clusters
    # Choose a different colour for each cluster for each cluster
    colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
    for i, centroid in enumerate(centroids):
        # Grab just the samples fpr the given cluster and plot them out with a new colour
        samples = all_samples[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster]
        plt.scatter(samples[:, 0], samples[:, 1], c=colour[i])
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()


def test_clusters():
    n_features = 2
    n_clusters = 3
    n_samples_per_cluster = 500
    seed = 700
    embiggen_factor = 70

    np.random.seed(seed)

    centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

    model = tf.global_variables_initializer()
    with tf.Session() as session:
        sample_values = session.run(samples)
        centroid_values = session.run(centroids)
    plot_clusters(sample_values, centroid_values, n_samples_per_cluster)


def choose_random_centroids(samples, n_clusters):
    # Step 0: Initialisation: Select `n_clusters` number of random points
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples))
    begin = [0, ]
    size = [n_clusters, ]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids


def assign_to_nearest(samples, centroids):
    # Finds the nearest centroid for each sample

    # START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(
        tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    # END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    nearest_indices = mins
    return nearest_indices


def update_centroids(samples, nearest_indices, n_clusters):
    # Updates the centroid to be the mean of all samples associated with it.
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    return new_centroids


def test_k_means():
    # n_features = 2
    # n_clusters = 3
    # n_samples_per_cluster = 500
    # seed = 700
    # embiggen_factor = 70
    #
    # centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
    # initial_centroids = choose_random_centroids(samples, n_clusters)
    #
    # model = tf.global_variables_initializer()
    # with tf.Session() as session:
    #     sample_values = session.run(samples)
    #     updated_centroid_value = session.run(initial_centroids)
    #
    # plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)
    n_features = 2
    n_clusters = 3
    n_samples_per_cluster = 500
    seed = 700
    embiggen_factor = 70

    data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
    initial_centroids = choose_random_centroids(samples, n_clusters)
    nearest_indices = assign_to_nearest(samples, initial_centroids)
    updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

    model = tf.global_variables_initializer()
    with tf.Session() as session:
        sample_values = session.run(samples)
        updated_centroid_value = session.run(updated_centroids)
        print(updated_centroid_value)

    plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)


def test_gradient_descent():
    # x and y are placeholders for our training data
    x = tf.placeholder("float")
    y = tf.placeholder("float")
    # w is the variable storing our values. It is initialised with starting "guesses"
    # w[0] is the "a" in our equation, w[1] is the "b"
    w = tf.Variable([1.0, 2.0], name="w")
    # Our model of y = a*x + b
    y_model = tf.multiply(x, w[0]) + w[1]

    # Our error is defined as the square of the differences
    error = tf.square(y - y_model)
    # The Gradient Descent Optimizer does the heavy lifting
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

    # Normal TensorFlow - initialize values, create a session and run the model
    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        for i in range(1000):
            x_value = np.random.rand()
            y_value = x_value * 2 + 6
            session.run(train_op, feed_dict={x: x_value, y: y_value})

        w_value = session.run(w)
        print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
        # errors = []
        # with tf.Session() as session:
        #     session.run(model)
        #     for i in range(1000):
        #         x_train = tf.random_normal((1,), mean=5, stddev=2.0)
        #         y_train = x_train * 2 + 6
        #         x_value, y_value = session.run([x_train, y_train])
        #         _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
        #         errors.append(error_value)
        #     w_value = session.run(w)
        #     print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
        # plt.plot([np.mean(errors[i - 50:i]) for i in range(len(errors))])
        # plt.show()
        # plt.savefig("errors.png")


def update_board(X):
    # Check out the details at: https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    # Compute number of neighbours,
    N = signal.convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    # Apply rules of the game
    X = (N == 3) | (X & (N == 2))
    return X


def test_custom_functions():
    shape = (50, 50)
    initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)
    # with tf.Session() as session:
    #     X = session.run(initial_board)
    fig = plt.figure()
    # plot = plt.imshow(X, cmap='Greys', interpolation='nearest')
    # plt.show()
    board = tf.placeholder(tf.int32, shape=shape, name='board')
    board_update = tf.py_func(update_board, [board], [tf.int32])
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        initial_board_values = session.run(initial_board)
        # update_board(initial_board_values)
        X = session.run(board_update, feed_dict={board: initial_board_values})[0]

        # def game_of_life(*args):
        #     A = session.run(board_update, feed_dict={board: X})[0]
        #     plot.set_array(A)
        #     return plot,
        #
        # ani = animation.FuncAnimation(fig, game_of_life, interval=100, blit=False)
        # plot = plt.imshow(X, cmap='Greys', interpolation='nearest')
        # plt.show()


def multiply_by_2(value):
    return value * 2


def test_py_func():
    sess = tf.InteractiveSession()
    input_pl = tf.placeholder(tf.int32, [])
    result_tensor = tf.py_func(multiply_by_2, [input_pl], [tf.int32])[0]
    result_value = sess.run([result_tensor], feed_dict={input_pl: 6})
    print(result_value)
    sess.close()


def test_distributed_computing():
    # run python create_worker.py 0 and python create_worker.py 1
    cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
    x = tf.constant(2)
    with tf.device("/job:local/task:1"):
        y2 = x - 66
    with tf.device("/job:local/task:0"):
        y1 = x + 300
        y = y1 + y2
    with tf.Session("grpc://localhost:2222") as sess:
        result = sess.run(y)
        print(result)

    x = tf.placeholder(tf.float32, 100)
    with tf.device("/job:local/task:1"):
        first_batch = tf.slice(x, [0], [50])
        mean1 = tf.reduce_mean(first_batch)
    with tf.device("/job:local/task:0"):
        second_batch = tf.slice(x, [50], [-1])
        mean2 = tf.reduce_mean(second_batch)
        mean = (mean1 + mean2) / 2
    with tf.Session("grpc://localhost:2222") as sess:
        result = sess.run(mean, feed_dict={x: np.random.random(100)})
        print(result)


def myfunction(x):
    return x + 5


def add(a, b):
    return a + b


def test_map_reduce():
    map_result = map(myfunction, [1, 2, 3])
    map_result_list = list(map_result)
    print(type(map_result_list), map_result_list)
    print(funtools.reduce(add, map_result_list))


def plot_basic_object(points):
    """Plots a basic object, assuming its convex and not too complex"""
    tri = sptial.Delaunay(points).convex_hull
    fig = plt.figure(figsize=(8, 8))
    ax = mplot3d.Axes3D(fig)  # fig.add_subplot(111, projection='3d')
    S = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                        triangles=tri,
                        shade=True, cmap=cm.Blues, lw=0.5)
    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)

    plt.show()


def create_cube(bottom_lower=(0, 0, 0), side_length=5):
    """Creates a cube starting from the given bottom-lower point (lowest x, y, z values)"""
    bottom_lower = np.array(bottom_lower)
    points = np.vstack([
        bottom_lower,
        bottom_lower + [0, side_length, 0],
        bottom_lower + [side_length, side_length, 0],
        bottom_lower + [side_length, 0, 0],
        bottom_lower + [0, 0, side_length],
        bottom_lower + [0, side_length, side_length],
        bottom_lower + [side_length, side_length, side_length],
        bottom_lower + [side_length, 0, side_length],
        bottom_lower,
    ])
    return points


def translate(points, amount):
    return tf.add(points, amount)


def rotate_around_z(points, theta):
    theta = float(theta)
    rotation_matrix = tf.stack([[tf.cos(theta), tf.sin(theta), 0],
                                [-tf.sin(theta), tf.cos(theta), 0],
                                [0, 0, 1]])
    return tf.matmul(tf.to_float(points), tf.to_float(rotation_matrix))


def test_3d():
    cube_1 = create_cube(side_length=2)
    plot_basic_object(cube_1)

    points = tf.constant(cube_1, dtype=tf.float32)
    # Update the values here to move the cube around.
    translation_amount = tf.constant([3, -3, 0], dtype=tf.float32)
    translate_op = translate(points, translation_amount)
    with tf.Session() as session:
        translated_cube = session.run(translate_op)
    plot_basic_object(translated_cube)
    # Rotating around the x - axis
    # [[1, 0, 0],
    #  [0, cos \theta, sin \theta],
    # [0, -sin \theta, cos \theta]]
    #
    #
    # Rotating around the y - axis
    # [[cos \theta, 0, -sin \theta],
    # [0, 1, 0],
    # [sin \theta, 0, cos \theta]]
    #
    # Rotating around the z - axis
    # [[cos \theta, sin \theta, 0],
    # [-sin \theta, cos \theta, 0],
    # [0, 0, 1]]
    with tf.Session() as session:
        result = session.run(rotate_around_z(cube_1, 75))
    plot_basic_object(result)


def test_visualisation():
    with tf.name_scope("MyOperationGroup"):
        with tf.name_scope("Scope_A"):
            a = tf.add(1, 2, name="Add_these_numbers")
            b = tf.multiply(a, 3)
        with tf.name_scope("Scope_B"):
            c = tf.add(4, 5, name="And_These_ones")
            d = tf.multiply(c, 6, name="Multiply_these_numbers")

    with tf.name_scope("Scope_C"):
        e = tf.multiply(4, 5, name="B_add")
        f = tf.div(c, 6, name="B_mul")
    g = tf.add(b, d)
    h = tf.multiply(g, f)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("output", sess.graph)
        print(sess.run(h))
        writer.close()


def test_readcsv():
    # todo: how to get current string encoding or convert any string to utf-8
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + "/olympics2016.csv"
    features = tf.placeholder(tf.int32, shape=[3], name='features')
    country = tf.placeholder(tf.string, name='country')
    total = tf.reduce_sum(features, name='total')
    printerop = tf.Print(total, [country, features, total], name='printer')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with open(filename) as inf:
            # Skip header
            next(inf)
            for line in inf:
                # Read data, using python, into our features
                code, country_name, gold, silver, bronze, total = line.strip().split(",")
                print(code, country_name, gold, silver, bronze, total)
                gold = int(gold)
                silver = int(silver)
                bronze = int(bronze)
                # Run the Print ob
                total = sess.run(printerop, feed_dict={features: [gold, silver, bronze], country: country_name})
                print(country_name, total)


def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[""], [""], [0], [0], [0], [0]]
    country, code, gold, silver, bronze, total = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.stack([gold, silver, bronze])
    return features, country


def test_tf_csv():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filenames = [dir_path + "/olympics2016.csv"]
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
    example, country = create_file_reader_ops(filename_queue)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        while True:
            try:
                example_data, country_name = sess.run([example, country])
                print(example_data, country_name)
            except tf.errors.OutOfRangeError:
                break


def test_linear_classifiers():
    X_values, y_flat = skdatasets.make_blobs(n_features=2, n_samples=800, centers=3, random_state=500)
    y = skpreprocessing.OneHotEncoder().fit_transform(y_flat.reshape(-1, 1)).todense()
    y = np.array(y)
    # Optional line: Sets a default figure size to be a bit larger.
    plt.rcParams['figure.figsize'] = (24, 10)
    # plt.scatter(X_values[:, 0], X_values[:, 1], c=y_flat, alpha=0.4, s=150)
    # plt.show()

    X_train, X_test, y_train, y_test, y_train_flat, y_test_flat = skmodelselection.train_test_split(X_values, y, y_flat)
    X_test += np.random.randn(*X_test.shape) * 1.5
    plt.plot(X_test[:, 0], X_test[:, 1], 'rx', markersize=20)
    plt.show()

    n_features = X_values.shape[1]
    n_classes = len(set(y_flat))
    weights_shape = (n_features, n_classes)
    W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model
    X = tf.placeholder(dtype=tf.float32)
    Y_true = tf.placeholder(dtype=tf.float32)
    bias_shape = (1, n_classes)
    b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))
    Y_pred = tf.matmul(X, W) + b
    loss_function = tf.losses.softmax_cross_entropy(Y_true, Y_pred)
    learner = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            result = sess.run(learner, {X: X_train, Y_true: y_train})
            if i % 100 == 0:
                print("Iteration {}:\tLoss={:.6f}".format(i, sess.run(loss_function, {X: X_test, Y_true: y_test})))
        y_pred = sess.run(Y_pred, {X: X_test})
        W_final, b_final = sess.run([W, b])
    predicted_y_values = np.argmax(y_pred, axis=1)
    predicted_y_values
    h = 1
    x_min, x_max = X_values[:, 0].min() - 2 * h, X_values[:, 0].max() + 2 * h
    y_min, y_max = X_values[:, 1].min() - 2 * h, X_values[:, 1].max() + 2 * h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_0, x_1 = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
    decision_points = np.c_[x_0.ravel(), x_1.ravel()]
    # We recreate our model in NumPy
    Z = np.argmax(decision_points @ W_final[[0, 1]] + b_final, axis=1)
    # Create a contour plot of the x_0 and x_1 values
    Z = Z.reshape(xx.shape)
    plt.contourf(x_0, x_1, Z, alpha=0.1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_flat, alpha=0.3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_y_values, marker='x', s=200)
    plt.xlim(x_0.min(), x_0.max())
    plt.ylim(x_1.min(), x_1.max())
    plt.show()


def test_reduce_mean():
    w = tf.Variable(tf.random_normal([748, 10], stddev=0.01, dtype=tf.float32, seed=789, name="w"))
    b = tf.Variable([10, 20, 30, 40, 50, 60], name="t")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf.reduce_mean(b)))
        print(sess.run(tf.reduce_mean(w)))


def test_argmax():
    a = [[0.9, 0.2, 0.3],
         [20, 2000, 3],
         [300, 600, 900]
         ]
    b = tf.Variable(a, name="b")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tf.arg_max(b, 1))  # 1:show the index of maximum value in each row of a
        print(result)
        result = sess.run(tf.arg_max(b, 0))  # 0:show the index of maximum value in each column of a
        print(result)


def test_linear_regression():
    trainX = np.linspace(-1, 1, 100)
    trainY = 3 * trainX + np.random.rand(*trainX.shape) * 0.33
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    w = tf.Variable(0.0, name="weights")
    y_model = tf.multiply(X, w)
    cost = tf.pow(Y - y_model, 2)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            for (x, y) in zip(trainX, trainY):
                sess.run(train_op, feed_dict={X: x, Y: y})
            print(sess.run(w))


def test_shape_function():
    a = tf.truncated_normal([16, 128, 128, 3])
    b = tf.reshape(a, [16 * 128, 128 * 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf.shape(a)), sess.run(tf.shape(b)))


def test_urllib3():
    http = urllib3.PoolManager()
    image_string = http.request('GET', 'http://cv-tricks.com/wp-content/uploads/2017/03/pexels-photo-362042.jpeg')
    print(image_string.status, image_string.data)
    image = tf.image.decode_jpeg(image_string.data, channels=3)
    print(image.shape)


def test_save_model():
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'my_test_model')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('my_test_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print(sess.run('w1:0'))


def test_contrib_learn():
    # Declare list of features. We only have one real-valued feature. There are many
    # other types of columns that are more complicated and useful.
    features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

    # An estimator is the front end to invoke training (fitting) and evaluation
    # (inference). There are many predefined types like linear regression,
    # logistic regression, linear classification, logistic classification, and
    # many neural network classifiers and regressors. The following code
    # provides an estimator that does linear regression.
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features, model_dir="./output")

    # TensorFlow provides many helper methods to read and set up data sets.
    # Here we use `numpy_input_fn`. We have to tell the function how many batches
    # of data (num_epochs) we want and how big each batch should be.
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4,
                                                  num_epochs=1000)

    # We can invoke 1000 training steps by invoking the `fit` method and passing the
    # training data set.
    estimator.fit(input_fn=input_fn, steps=1000)

    # Here we evaluate how well our model did. In a real example, we would want
    # to use a separate validation and testing data set to avoid overfitting.
    print(estimator.evaluate(input_fn=input_fn))


def test_converting_feature_data_to_ensors():
    feature_column_data = [1, 2.4, 0, 9.9, 3, 120]
    feature_tensor = tf.constant(feature_column_data)
    # For sparse, categorical data (data where the majority of values are 0),
    # you'll instead want to populate a SparseTensor
    sparse_tensor = tf.SparseTensor(indices=[[0, 1], [2, 4]], values=[6, 0.5], dense_shape=[3, 5])
    print(sparse_tensor)
    # This corresponds to the following dense tensor:
    # [
    #     [0, 6, 0, 0, 0]
    #     [0, 0, 0, 0, 0]
    #     [0, 0, 0, 0, 0.5]
    # ]


def test_boston_predict_MEDV():
    tf.logging.set_verbosity(tf.logging.INFO)
    COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
    FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
    LABEL = "medv"

    training_set = pd.read_csv("boston/boston_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("boston/boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("boston/boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10],
                                              model_dir="boston/boston_model")

    def input_fn(data_set):
        feature_cols = {k: tf.constant(data_set[k].values)
                        for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        return feature_cols, labels

    regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))
    y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
    # .predict() returns an iterator; convert to a list and print predictions
    predictions = list(itertools.islice(y, 6))
    print("Predictions: {}".format(str(predictions)))


print()
# test_something()
# test_image()
# get_system_info()
# test_interactive_tensorFlow()
# test_clusters()
# test_k_means()
# test_gradient_descent()
# test_py_func()
# test_custom_functions()  # can't run now
# test_distributed_computing()
# test_map_reduce()
# test_3d()
# test_visualisation()
# test_readcsv()
# test_tf_csv()
# test_linear_classifiers()
# test_reduce_mean()
# test_argmax()
# test_linear_regression()
# test_shape_function()
# df = quandl.get("WIKI/GOOGL")
# print(df.head())
# test_urllib3()
# test_save_model()
# test_contrib_learn()
# test_converting_feature_data_to_ensors()
test_boston_predict_MEDV()
