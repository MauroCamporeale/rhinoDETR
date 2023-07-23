import torch, torchvision
import torchvision.transforms as T
import scipy

# ---------------------------------------------------------------------

if __name__ == '__main__':

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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

    # ------------------------------------------------------------------------------

    def filter_bboxes_from_outputs(outputs, threshold=0.7):
        # keep only predictions with confidence above threshold
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        probas_to_keep = probas[keep]

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

        return probas_to_keep, bboxes_scaled

    # ---------------------------------------------------------------------

    # COCO classes
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    # ---------------------------------------------------------------------

    import matplotlib.pyplot as plt

    def plot_results(pil_img, prob=None, boxes=None):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = COLORS * 100
        if prob is not None and boxes is not None:
          for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
              ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3))
              cl = p.argmax()
              text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
              ax.text(xmin, ymin, text, fontsize=15,
                      bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()

    # # ---------------------------------------------------------------------
    # #Load an image for a demo
    # # ---------------------------------------------------------------------
    #
    # model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    # model.eval();
    #
    # # ---------------------------------------------------------------------
    #
    # from PIL import Image
    # import requests
    #
    # url = 'http://images.cocodataset.org/train2017/000000310645.jpg'
    # im = Image.open(requests.get(url, stream=True).raw)
    #
    # # ---------------------------------------------------------------------
    #
    # # mean-std normalize the input image (batch-size: 1)
    # img = transform(im).unsqueeze(0)
    #
    # # propagate through the model
    # outputs = model(img)
    #
    # # ---------------------------------------------------------------------
    #
    # for threshold in [0.9, 0.7, 0.0]:
    #     probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, threshold=threshold)
    #     plot_results(im, probas_to_keep, bboxes_scaled)

    # ---------------------------------------------------------------------
    #Clone my custom code of DETR

    '''%cd /content/
    
    !rm -rf detr
    !git clone https://github.com/woctezuma/detr.git
    
    %cd detr/
    
    !git checkout finetune'''
    # ---------------------------------------------------------------------
    #Load pre-trained weights
    # ---------------------------------------------------------------------
    # Get pretrained weights
    checkpoint = torch.hub.load_state_dict_from_url(
                url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                map_location='cpu',
                check_hash=True)

    # Remove class weights
    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]

    # Save
    torch.save(checkpoint, 'detr-r50_no-class-head.pth')

    # ---------------------------------------------------------------------
    #Prepare dataset for fine-tuning
    # ---------------------------------------------------------------------

    # Choose whether to start indexing categories with 0 or with 1.
    #
    # NB: convention in COCO dataset is such that the 1st class (person) has ID n°1.
    #
    # NB²: this is why we chose to set to 1 the default value of `first_class_index`
    # in `via2coco.convert()`.

    first_class_index = 1

    # ---------------------------------------------------------------------
    # Check the dataset
    # ---------------------------------------------------------------------

    import pycocotools.coco as coco
    from pycocotools.coco import COCO
    import numpy as np
    import skimage.io as io
    import matplotlib.pyplot as plt
    import pylab
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)

    dataDir='E:/Users/Mauro/PycharmProjects/Rhino-Detr/rhino-cells-balanced-augmented/'
    dataType='train2017'
    annFile='{}annotations/custom_train.json'.format(dataDir)

    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())

    nms=[cat['name'] for cat in cats]
    print('Categories: {}'.format(nms))

    nms = set([cat['supercategory'] for cat in cats])
    print('Super-categories: {}'.format(nms))

    # # load and display image
    # catIds = coco.getCatIds(catNms=['epithelial']); #'balloon'
    # imgIds = coco.getImgIds(catIds=catIds );
    #
    # img_id = imgIds[np.random.randint(0,len(imgIds))]
    # print('Image n°{}'.format(img_id))
    #
    # img = coco.loadImgs(img_id)[0]
    #
    # img_name = '%s/%s/%s'%(dataDir, dataType, img['file_name'])
    # print('Image name: {}'.format(img_name))
    #
    # I = io.imread(img_name)
    # plt.figure()
    # plt.imshow(I)
    #
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
    # anns = coco.loadAnns(annIds)
    #
    # # load and display instance annotations
    # plt.imshow(I)
    # coco.showAnns(anns, draw_bbox=False)
    #
    # plt.imshow(I)
    # coco.showAnns(anns, draw_bbox=True)

    # ---------------------------------------------------------------------
    # Fine-tuning
    # ---------------------------------------------------------------------

    assert(first_class_index in [0, 1])

    if first_class_index == 0:

      # There is one class, balloon, with ID n°0.

      num_classes = len(cats) - 1

      # finetuned_classes = [
      #     'artefatto', 'batteri', 'emazia', 'eosinophil', 'epithelial', 'epithelial ciliated', 'lymphocyte', 'mast cell', 'metaplastic', 'muciparous', 'neutrophil'
      # ]

      finetuned_classes = [
          'artefatto', 'emazia', 'eosinophil', 'epithelial', 'lymphocyte',
          'mast cell', 'metaplastic', 'muciparous', 'neutrophil'
      ]

      # The `no_object` class will be automatically reserved by DETR with ID equal
      # to `num_classes`, so ID n°1 here.

    else:

      # There is one class, balloon, with ID n°1.
      #
      # However, DETR assumes that indexing starts with 0, as in computer science,
      # so there is a dummy class with ID n°0.
      # Caveat: this dummy class is not the `no_object` class reserved by DETR.

      num_classes = len(cats)

      # finetuned_classes = [
      #     'N/A', 'artefatto', 'batteri', 'emazia', 'eosinophil', 'epithelial', 'epithelial ciliated', 'lymphocyte', 'mast cell', 'metaplastic', 'muciparous', 'neutrophil'
      # ]

      finetuned_classes = [
          'N/A', 'artefatto', 'emazia', 'eosinophil', 'epithelial', 'lymphocyte',
          'mast cell', 'metaplastic', 'muciparous', 'neutrophil'
      ]

      # The `no_object` class will be automatically reserved by DETR with ID equal
      # to `num_classes`, so ID n°2 here.

    print('First class index: {}'.format(first_class_index))
    print('Parameter num_classes: {}'.format(num_classes))
    print('Fine-tuned classes: {}'.format(finetuned_classes))

    from detr import main

    args=[
        "--dataset_file", "custom",
        "--coco_path", "E:/Users/Mauro/PycharmProjects/Rhino-Detr/rhino-cells-balanced-augmented/",
        "--output_dir", "outputs10epochsbalanceaugmented",
        "--resume", "detr-r50_no-class-head.pth",
        "--num_classes", "10", #cambia a seconda di num_classes
        "--epochs", "50"
    ]

    main.main(args)