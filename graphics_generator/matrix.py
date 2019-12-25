import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples): #Oranges
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Predita')
    plt.tight_layout()


# Compute confusion matrix 1
cnf_matrix1 = np.array([[ 3, 1, 0, 1, 2, 6, 0, 0, 0, 0],
 [ 0, 21, 3, 0, 10, 17, 2, 0, 0, 0],
 [ 0, 2, 0, 0, 2, 4, 0, 0, 0, 0],
 [ 0, 3, 0, 5, 6, 16, 5, 0, 0, 0],
 [ 4, 11, 2, 2, 35, 51, 16, 0, 0, 0],
 [ 4, 18, 3, 4, 39, 60, 19, 3, 0, 0],
 [ 0, 9, 4, 2, 17, 36, 18, 2, 0, 0],
 [ 0, 3, 0, 0, 2, 5, 6, 1, 0, 0],
 [ 0, 1, 0, 0, 1, 1, 2, 0, 1, 0],
 [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

#Compute confusion matrix 2
cnf_matrix2 = np.array([[13,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [0, 46,  0,  0,  3,  3,  1,  0,  0,  0],
 [0,  0,  8,  0,  0,  0,  0,  0,  0,  0],
 [0,  0,  0, 32,  1,  2,  0,  0,  0,  0],
 [0,  0,  1,  0, 114,  4,  2,  0,  0,  0],
 [0,  0,  0,  0,  4, 145,  1,  0,  0,  0],
 [0,  1,  1,  1,  4,  4, 77,  0,  0,  0],
 [0,  0,  0,  0,  0,  0,  0, 17,  0,  0],
 [0,  0,  0,  0,  0,  0,  0,  0,  6,  0],
 [0,  0,  0,  0,  0,  0,  0,  0,  0,  1]])

# Compute confusion matrix 3
cnf_matrix3 = np.array([[13,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [0, 53,  0,  0,  0,  0,  0,  0,  0,  0],
 [0,  0,  8,  0,  0,  0,  0,  0,  0,  0],
 [0,  0,  0, 35,  0,  0,  0,  0,  0,  0],
 [1,  2,  1,  2, 95, 16,  4,  0,  0,  0],
 [3,  5,  0,  4, 22, 107,  9,  0,  0,  0],
 [0,  4,  0,  2, 10, 15, 57,  0,  0,  0],
 [0,  0,  0,  0,  0,  0,  0, 17,  0,  0],
 [0,  0,  0,  0,  0,  0,  0,  0,  6,  0],
 [0,  0,  0,  0,  0,  0,  0,  0,  0,  1]])


def plot_matrix(matrix, approach):
    class_names = ['0','1','2','3','5','8','13','20','40','89']
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(matrix, classes=class_names,
                          title='Abordagem %d\nMatriz de Confusão' % approach)

    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(matrix, classes=class_names, normalize=True,
    #                       title='Abordagem %d\nMatriz de Confusão Normalizada' % approach)

    plt.show()


plot_matrix(cnf_matrix1, 1)
plot_matrix(cnf_matrix2, 2)
plot_matrix(cnf_matrix3, 3)



