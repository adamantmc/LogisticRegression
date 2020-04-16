import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import json

# Data from: https://www.kaggle.com/c/dogs-vs-cats/data
def load_dataset(image_folder="images/", split_ratio=0.7):
    labels_dict = {"cat": 0, "dog": 1}

    vectors_pickle_file = "image_vectors.pickle"
    labels_pickle_file = "image_labels.pickle"
    image_paths_file = "image_paths.json"

    if os.path.exists(vectors_pickle_file) and os.path.exists(labels_pickle_file):
        print("Loading image vectors from {}".format(vectors_pickle_file))
        images = pickle.load(open(vectors_pickle_file, "rb"))
        print("Loading image labels from {}".format(labels_pickle_file))
        labels = pickle.load(open(labels_pickle_file, "rb"))
        print("Loading image paths from {}".format(image_paths_file))
        filenames = json.load(open(image_paths_file, "r"))
    else:
        from PIL import Image
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        from tensorflow.keras import Model

        resnet = ResNet50(weights='imagenet')
        feature_model = Model(inputs=resnet.input, outputs=resnet.get_layer(index=len(resnet.layers) - 2).output)

        files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                 os.path.isfile(os.path.join(image_folder, filename))]
        np.random.shuffle(files)

        labels = [labels_dict[os.path.basename(f).split(".")[0]] for f in files]
        images = None
        filenames = []

        step = 2000
        for i in range(0, len(files), step):
            print(i+1)
            file_batch = files[i:i + step]

            batch_images = []

            for filename in file_batch:
                filenames.append(filename)
                img = Image.open(filename).resize((224, 224), Image.ANTIALIAS)
                img = np.asarray(img, dtype="float64")
                batch_images.append(preprocess_input(img.copy()))

            image_features = feature_model.predict(np.asarray(batch_images))

            if images is None:
                images = image_features
            else:
                images = np.concatenate((images, image_features), axis=0)

        pickle.dump(images, open(vectors_pickle_file, "wb"))
        pickle.dump(labels, open(labels_pickle_file, "wb"))
        json.dump(filenames, open(image_paths_file, "w"))

    train_last_index = int(split_ratio * len(images))
    no_classes = len(labels_dict["cat"]) if type(labels_dict["cat"]) == list else 1

    train_x = np.asarray(images[:train_last_index])
    train_y = np.asarray(labels[:train_last_index]).reshape(-1, no_classes)
    train_filenames = filenames[:train_last_index]

    test_x = np.asarray(images[train_last_index:])
    test_y = np.asarray(labels[train_last_index:]).reshape(-1, no_classes)
    test_filenames = filenames[train_last_index:]

    return train_x, train_y, train_filenames, test_x, test_y, test_filenames

def plot_images_with_predictions(image_paths, predictions, savepath="image_plot.png"):
    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(5, 5),
                     axes_pad=0.5,
                     )
    images = []
    labels = []

    for path, pred in zip(image_paths, predictions):
        img = Image.open(path).resize((224, 224), Image.ANTIALIAS)
        images.append(img)

        if pred > 0.5:
            labels.append("Dog - {}".format(str(pred)[:4]))
        else:
            labels.append("Cat - {}".format(str(1-pred)[:4]))

    for ax, img, label in zip(grid, images, labels):
        ax.imshow(img)
        ax.set_title(label)

    plt.savefig(savepath)
