# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .coco import COCODetection
from .data_augment import *
from .config import *
from .voc0712_multi_iou import VOCDetection2
from .visd import visdDetection
