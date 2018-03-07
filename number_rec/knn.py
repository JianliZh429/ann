import cv2
import numpy as np

train_img = cv2.imread('./data/0-9.jpg')

gray_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)

im, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
train_cells = []
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    train_cells.append(cv2.resize(gray_img[y:y + h, x:x + w], (30, 30)))

# for (i, t) in enumerate(train_cells):
#     cv2.imshow('{}'.format(i), t)
# cv2.waitKey(0)

training = np.array(train_cells).reshape(-1, 900).astype(np.float32)

k = np.arange(10)
train_labels = k[:, np.newaxis]
test_labels = train_labels.copy()

knn = cv2.ml.KNearest_create()
knn.train(training, cv2.ml.ROW_SAMPLE, train_labels)

test_image = cv2.imread('./data/numbers.jpg')
cols, rows = test_image.shape[:2]

test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(test_gray, 100, 255, cv2.THRESH_BINARY_INV)
im2, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
test_cells = []
for (i, cnt) in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    test_cells.append(cv2.resize(test_gray[y:y + h, x:x + w], (30, 30)))

testing = np.array(test_cells).reshape(-1, 900).astype(np.float32)

ret, result, neighbours, dist = knn.findNearest(testing, k=1)
print('ret:{}'.format(ret))
print('result:{}'.format(result))
print('neighbours:{}'.format(neighbours))
print('dist:{}'.format(dist))

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
# matches = result == test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct * 100.0 / result.size
# print(accuracy)
