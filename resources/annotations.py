import os
import numpy as np
from detectron2.structures import BoxMode


def create_COCO_annotations(root_path : str, dataset_path: str):

  annotations_path =  os.path.join(root_path, dataset_path, 'labels')
  annotations_files = os.listdir(annotations_path)

  dataset_dicts = []
  for idx, filename in enumerate(annotations_files):

    annotation_json = {}

    imagename = os.path.join(root_path, dataset_path, 'images', filename[:-4] + '.jpg')
    height, width = cv2.imread(imagename).shape[:2]

    annotation_json["file_name"] = imagename
    annotation_json["image_id"] = idx
    annotation_json["height"] = height
    annotation_json["width"] = width

    annotation_json["annotations"] = []

    annotations=np.loadtxt(os.path.join(annotations_path, filename))

    if len(annotations) == 0:
      pass
    if annotations.ndim == 1:
      annotations = annotations[:, np.newaxis]

    for row in annotations:
      if len(row) == 5:
        c_x, c_y, w_, h_ = row[1], row[2], row[3], row[4]

        bbox = [
          int((float(c_x) - (float(w_)/2)) * width),
          int((float(c_y) - (float(h_)/2)) * height),
          int(float(w_) * width),
          int(float(h_) * height),
        ]

        annotation_dict = {}
        annotation_dict["category_id"] = int(row[0])
        annotation_dict["bbox"] = bbox
        annotation_dict["bbox_mode"] = BoxMode.XYWH_ABS

        annotation_json["annotations"].append(annotation_dict)

    dataset_dicts.append(annotation_json)

  return dataset_dicts