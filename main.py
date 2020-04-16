import numpy as np
from logistic_regression import LogisticRegression, confusion_matrix
from utils import load_dataset, plot_images_with_predictions

train_x, train_y, train_paths, test_x, test_y, test_paths = load_dataset("images/")

print("X shape: {}".format(train_x.shape))
print("Y shape: {}".format(train_y.shape))

model = LogisticRegression(iterations=1000, learning_rate=0.01)
model.fit(train_x, train_y)
model.plot_performance()
tp, tn, fp, fn = confusion_matrix(model.predict(test_x), test_y)
print("Test set accuracy: {}".format((tp + tn) / (tp + tn + fp + fn)))

random_indexes = np.random.choice(len(test_paths), 25, replace=False)
random_image_preds = [p[0] for p in model.predict(test_x[random_indexes]).T]

plot_images_with_predictions([test_paths[i] for i in random_indexes], random_image_preds, "image_plot.png")
