from my_utils import calc_mAP
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

import torch

import warnings
warnings.filterwarnings('ignore')

# for good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# parameters
data_folder = '../../input/voc_2005/'
batch_size = 32
workers = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = '../model_checkpoints/checkpoint_ssd300_vgg11.pth.tar'

# load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# switch to eval mode
model.eval()

# load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # switch to eval mode
    model.eval()

    # lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    with torch.no_grad():
        # batches
        for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # forward prop.
            predicted_locs, predicted_scores = model(images)

            # detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200...
            # ...for fair comparision with the paper's results and other repos

            # store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)

        # calculate mAP
        APs, mAP, recall = calc_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)

    # print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)