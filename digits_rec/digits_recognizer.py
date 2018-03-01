import os

import cv2
import numpy as np

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}


def thresholding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
    return threshold


def _contains_contours_rect(bounding_rect1, bounding_rect2):
    x1, y1, w1, h1 = bounding_rect1[0]
    x2, y2, w2, h2 = bounding_rect2[0]
    return x1 < x2 < x1 + w1 and y1 < y2 < y1 + h1 and w1 > w2 and h1 > h2


def _remove_god_contours(sorted_cnts):
    length = len(sorted_cnts)
    if length <= 1:
        return sorted_cnts
    cnts = sorted_cnts[::-1]
    filtered = []
    smaller = [0] * length
    for i in range(0, length):
        j = i + 1
        if smaller[i]:
            continue
        while j < length:
            if not smaller[j]:
                if _contains_contours_rect(cnts[j], cnts[i]):
                    smaller[j] = 1
            j += 1
    for i in range(length):
        if not smaller[i]:
            filtered.append(cnts[i][1])
    return filtered


def filter_contours(cnts, min_rect=None, max_rect=None):
    first_filtered = []

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        legal = True
        if min_rect is not None and (w <= min_rect[0] or h <= min_rect[1]):
            legal = False
        if max_rect is not None and (w >= max_rect[0] or h >= max_rect[1]):
            legal = False
        if legal:
            first_filtered.append(((x, y, w, h), cnt))

    first_filtered = sorted(first_filtered, key=lambda a: (a[0][2], a[0][3], a[0][0], a[0][1]), reverse=True)
    return _remove_god_contours(first_filtered)


def detect(img):
    row, col, _ = img.shape
    threshold1 = thresholding(img)
    opening = cv2.morphologyEx(threshold1, cv2.MORPH_OPEN, np.ones((3, 3)))

    im2, cnts, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
    for (i, cnt) in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        if col * 0.5 > w > 30 and row * 0.8 > h > 40:
            digit = rec(img[y:y + h, x:x + w])
            digits += digit
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(img, str(*digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if len(digits) > 0:
        print(u"{}{}.{} \u00b0C".format(*digits))


def rec(roi):
    (roi_h, roi_w, _) = roi.shape
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # print(roi_h, roi_w)
    # if roi_h == 63:
    #     cv2.imshow('roi{}'.format(roi_h), roi)
    (d_w, d_h) = (int(roi_w * 0.3), int(roi_h * 0.15))
    d_hc = int(roi_h * 0.05)
    segments = [
        ((0, 0), (roi_w, d_h)),  # top
        ((0, 0), (d_w, roi_h // 2)),  # top-left
        ((roi_w - d_w, 0), (roi_w, roi_h // 2)),  # top-right
        ((0, (roi_h // 2) - d_hc), (roi_w, (roi_h // 2) + d_hc)),  # center
        ((0, roi_h // 2), (d_w, roi_h)),  # bottom-left
        ((roi_w - d_w, roi_h // 2), (roi_w, roi_h)),  # bottom-right
        ((0, roi_h - d_h), (roi_w, roi_h))  # bottom
    ]
    on = [0] * len(segments)
    digits = []
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        seg_roi = roi[yA:yB, xA:xB]
        # if roi_h == 63:
        #     cv2.imshow('{}_{}'.format(roi_h, i), seg_roi)
        total = cv2.countNonZero(seg_roi)
        area = (xB - xA) * (yB - yA)
        if total / float(area) > 0.35:
            on[i] = 1

        # lookup the digit and draw it on the image
    print(on)
    digit = DIGITS_LOOKUP[tuple(on)]
    print('digit:{}'.format(digit))
    digits.append(digit)
    return digits


def inference(image_path):
    im = cv2.imread(image_path)
    row, col, _ = im.shape
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 50, 200, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    _, cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered = filter_contours(cnts, min_rect=(20, 45), max_rect=(int(row * 0.8), int(col * 0.8)))
    for (i, f) in enumerate(filtered):
        x, y, w, h = cv2.boundingRect(f)
        detect(im[y:y + h, x:x + w])
        # cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.putText(im, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imshow('edge', im)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image = os.path.join(os.path.dirname(__file__), 'data/sample_00.jpg')
    inference(image)
