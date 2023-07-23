import torch, torchvision
import torchvision.transforms as T
import scipy
import matplotlib.pyplot as plt
import pycocotools.coco as coco
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

# ---------------------------------------------------------------------
# Monitoring of the training
# ---------------------------------------------------------------------

from util.plot_utils import plot_logs
from pathlib import Path

log_directory = [Path('outputs50epochsbalanceaugmented/')]

# fields_of_interest = (
#     'loss',
#     'mAP',
#     )
#
# plot_logs(log_directory, fields_of_interest)
#
#
# fields_of_interest = (
#     'loss_ce',
#     'loss_bbox',
#     'loss_giou',
#     )
#
# plot_logs(log_directory, fields_of_interest)
#
#
# fields_of_interest = (
#     'class_error',
#     'cardinality_error_unscaled',
#     )
#
# plot_logs(log_directory, fields_of_interest)

# ---------------------------------------------------------------------
# Load the fine-tuned model
# ---------------------------------------------------------------------

model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=10)#cambia a seconda di num_classes

checkpoint = torch.load('outputs50epochsbalanceaugmented/checkpoint.pth',
                        map_location='cpu')

model.load_state_dict(checkpoint['model'],
                      strict=False)

model.eval();

# ---------------------------------------------------------------------
# Boilerplate functions to display fine-tuned results
# ---------------------------------------------------------------------
#
# finetuned_classes = [
#           'N/A', 'artefatto', 'batteri', 'emazia', 'eosinophil', 'epithelial', 'epithelial ciliated', 'lymphocyte', 'mast cell', 'metaplastic', 'muciparous', 'neutrophil'
#       ]
finetuned_classes = [
          'N/A', 'artefatto', 'emazia', 'eosinophil', 'epithelial', 'lymphocyte',
          'mast cell', 'metaplastic', 'muciparous', 'neutrophil'
      ]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# ---------------------------------------------------------------------

transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ---------------------------------------------------------------------

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# ---------------------------------------------------------------------

def filter_bboxes_from_outputs(outputs, threshold=0.7):
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    return probas_to_keep, bboxes_scaled


# ---------------------------------------------------------------------

def plot_finetuned_results(pil_img, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def run_worflow(my_image, my_model):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(my_image).unsqueeze(0)

    # propagate through the model
    outputs = my_model(img)

    for threshold in [0.9, 0.7, 0.5]:
        probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, threshold=threshold)

        plot_finetuned_results(my_image, probas_to_keep, bboxes_scaled)

# ---------------------------------------------------------------------

# from PIL import Image
#
# img_name = "E:/Users/Mauro/PycharmProjects/Rhino-Detr/rhino-cells/train2017/img-0004_png_jpg.rf.419e3d423af19fcecc5e722716f1af99.jpg"
# im = Image.open(img_name)
#
# run_worflow(im, model)

# ---------------------------------------------------------------------

from PIL import Image

img_name = 'E:/Users/Mauro/PycharmProjects/Rhino-Detr/rhino-cells/test/img-0106_png_jpg.rf.5f72fab4b491ee92895da88108b2799a.jpg'
im = Image.open(img_name)

run_worflow(im, model)