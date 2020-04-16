import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1.0-1e-7)
    return -(np.sum((y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))) / y_pred.shape[1])

def confusion_matrix(y_pred, y_true):
    tp, tn, fp, fn = 0, 0, 0, 0
    
    y_pred = list(y_pred.reshape(-1, 1))
    y_true = list(y_true.reshape(-1, 1))
    
    for pred, true in zip(y_pred, y_true):
        if pred < 0.5:
            if true == 0:
                tn += 1
            else:
                fn += 1
        else:
            if true == 1:
                tp += 1
            else:
                fp += 1

    return tp, tn, fp, fn

class LogisticRegression(object):

    def __init__(self, learning_rate=0.001, activation_function=sigmoid, loss_function=entropy_loss, iterations=10):
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.iterations = iterations
        
    def fit(self, x, y, validation_set_ratio=0.10):
        x = x.T # Cast to (# features, # examples)
        y = y.T # Cast to (label, # examples)

        train_set_limit = x.shape[1] - int(validation_set_ratio*x.shape[1])

        val_x = x[:, train_set_limit:]
        val_y = y[:, train_set_limit:]

        x = x[:, :train_set_limit]
        y = y[:, :train_set_limit]

        self._w = np.random.randn(x.shape[0], 1) * 0.01
        self._b = 0
        
        self._val_loss = []
        self._train_loss = []

        self._val_acc = []
        self._train_acc = []

        # Number of examples
        m = x.shape[1]

        for iteration in range(self.iterations):
            # Propagate - calculate W and b gradients
            z = np.dot(self._w.T, x) + self._b
            a = self.activation_function(z)

            loss = entropy_loss(a, y)
            dz = a - y

            dw = np.dot(x, dz.T) / m
            db = np.sum(dz) / m

            if (iteration + 1) % 10 == 0:
                tp, tn, fp, fn = confusion_matrix(z, y)
                acc = (tp+tn)/(tp+tn+fp+fn)

                val_x_predictions = self.predict(val_x.T)
                tp, tn, fp, fn = confusion_matrix(val_x_predictions, val_y)                
                v_acc = (tp+tn)/(tp+tn+fp+fn)
                v_loss = entropy_loss(val_x_predictions, val_y)

                self._train_acc.append(acc)
                self._train_loss.append(loss)

                self._val_acc.append(v_acc)
                self._val_loss.append(v_loss)

                print("Iteration {} - Training loss: {} Validation loss: {}".format(iteration + 1, loss, v_loss))

            # Optimize
            self._w = self._w - self.learning_rate * dw
            self._b = self._b - self.learning_rate * db
            

    def plot_performance(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self._val_loss, label="Validation set loss")
        plt.plot(self._train_loss, label="Training set loss")

        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self._val_acc, label="Validation set accuracy")
        plt.plot(self._train_acc, label="Training set accuracy")

        plt.xlabel("Iterations (in hundreds)")

        plt.legend()

        plt.show()

    def predict(self, x):
        # Cast to (# features, # examples)
        x = x.T

        return self.activation_function(np.dot(self._w.T, x) + self._b)
    