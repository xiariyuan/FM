import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """NumPy fallback for cython_bbox.bbox_overlaps."""
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)
    overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float64)
    if boxes.size == 0 or query_boxes.size == 0:
        return overlaps

    box_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    query_area = np.maximum(0.0, query_boxes[:, 2] - query_boxes[:, 0]) * np.maximum(
        0.0, query_boxes[:, 3] - query_boxes[:, 1]
    )

    for i in range(boxes.shape[0]):
        xx1 = np.maximum(boxes[i, 0], query_boxes[:, 0])
        yy1 = np.maximum(boxes[i, 1], query_boxes[:, 1])
        xx2 = np.minimum(boxes[i, 2], query_boxes[:, 2])
        yy2 = np.minimum(boxes[i, 3], query_boxes[:, 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = box_area[i] + query_area - inter
        valid = union > 0.0
        overlaps[i, valid] = inter[valid] / union[valid]
    return overlaps
