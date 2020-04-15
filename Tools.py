import copy

"""
Collection of tools handling bounding-boxes
"""

def NMS(orig_dets, iou_thresh):
    """
    :param dets: [[x, y, w, h], ..., [x, y, w, h]]
    :return: keeping det indices
    """

    dets = copy.deepcopy(orig_dets)
    dets = dets[dets[:,5].argsort()[::-1]]
    keep = [True]*len(dets)
    for i in range(0, len(dets)-1):
        for j in range(i+1, len(dets)):
            if IOU(dets[i], dets[j]) > iou_thresh:
                keep[j] = False
                
    return keep

def IOU(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y2 = min(bb1[1]+bb1[3], bb1[1]+bb2[3])

    intersection = max(0, x2-x1+1)*max(0, y2-y1+1)

    bb1_area = bb1[2]*bb1[3]
    bb2_area = bb2[2]*bb2[3]

    iou = intersection/float(bb1_area + bb2_area - intersection)

    return iou