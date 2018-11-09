import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	fig, axes = plt.subplots(1, 2)
	axes[0].plot(train_losses, 'r-', label='training loss')
	axes[0].plot(valid_losses, 'b-', label='validation loss')
	axes[0].legend(loc="upper left")
	axes[0].set_xlabel("epoch")
	axes[0].set_ylabel("Loss")
	
	axes[1].plot(train_accuracies, 'r-', label='training accuracy')
	axes[1].plot(valid_accuracies, 'b-', label='validation accuracy')
	axes[1].legend(loc="upper left")
	axes[1].set_xlabel("epoch")
	axes[1].set_ylabel("Accuracy")	
	fig.savefig('RNN_res_2.png')


def plot_confusion_matrix(results, class_names):
	def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		fmt = '.2f'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")

		plt.ylabel('True')
		plt.xlabel('Predicted')
		plt.tight_layout()

	y_true, y_pred = zip(*results)
	cnf_matrix = confusion_matrix(y_true, y_pred)
	np.set_printoptions(precision=2)

	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, title='Normalized Confusion Matrix')
	plt.savefig("RNN_confusion_matrix_2.png")