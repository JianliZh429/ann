import cv2
import numpy as np

with np.load('knn_data.npz') as data:
    train = data['train']
    train_labels = data['train_labels']

    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    sample = cv2.imread('data/sample.jpg')
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    cols, rows = sample.shape[:2]
    print(cols, rows)
    cells = [np.hsplit(row, 9) for row in np.vsplit(sample, 4)]

    samples = []
    for i in range(4):
        for j in range(9):
            cells[i][j] = cv2.resize(cells[i][j], (20, 20))
            x = np.reshape(cells[i][j], 400)
            samples.append(x.astype(np.float32))

    samples = np.array(samples)
    # for (i, s) in enumerate(samples):
    #     cv2.imshow('{}'.format(i), s.reshape(20, 20).astype(np.uint8))

    ret, result, neighbours, dist = knn.findNearest(samples, k=3)
    print('ret:{}'.format(ret))
    print('result:{}'.format(result))
    print('neighbours:{}'.format(neighbours))
    print('dist:{}'.format(dist))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
