import numpy as np
import os
from logistic_regression import LogisticRegression, confusion_matrix
import pickle

# Data from: https://www.kaggle.com/c/dogs-vs-cats/data
def load_dataset(image_folder="images/", split_ratio=0.7):
    labels_dict = {"cat": 0, "dog": 1}

    vectors_pickle_file = "image_vectors.pickle"
    labels_pickle_file = "image_labels.pickle"
    
    if os.path.exists(vectors_pickle_file) and os.path.exists(labels_pickle_file):
        print("Loading image vectors from {}".format(vectors_pickle_file))
        images = pickle.load(open(vectors_pickle_file, "rb"))
        print("Loading image labels from {}".format(labels_pickle_file))
        labels = pickle.load(open(labels_pickle_file, "rb"))
    else:
        from PIL import Image
        from keras.applications.resnet50 import ResNet50, preprocess_input
        from keras import Model
        
        resnet = ResNet50(weights='imagenet')
        feature_model = Model(inputs=resnet.input, outputs=resnet.get_layer("flatten_1").output)  
        
        files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, filename))][:100]
        np.random.shuffle(files)
        
        labels = [labels_dict[os.path.basename(f).split(".")[0]] for f in files]
        images = None

        step = 2000
        for i in range(0, len(files), step):
            file_batch = files[i:i+step]

            batch_images = []

            for filename in file_batch:
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
    
    print(images.shape)

    train_last_index = int(split_ratio*len(images))

    train_x = np.asarray(images[:train_last_index])
    train_y = np.asarray(labels[:train_last_index]).reshape(-1, 1)

    test_x = np.asarray(images[train_last_index:])
    test_y = np.asarray(labels[train_last_index:]).reshape(-1, 1)

    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = load_dataset("images/")

print("X shape: {}".format(train_x.shape))
print("Y shape: {}".format(train_y.shape))

model = LogisticRegression(iterations=1000, learning_rate=0.01)
model.fit(train_x, train_y)
tp, tn, fp, fn = confusion_matrix(model.predict(test_x), test_y)
print("Test set accuracy: {}".format((tp + tn) / (tp + tn + fp + fn)))