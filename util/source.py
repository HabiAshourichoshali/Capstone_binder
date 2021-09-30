

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np


def plot_example(path,batch_size,image_number):
    import torch
    from torchvision import datasets, transforms, models
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])
    test_data = datasets.ImageFolder(path,
                                      transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size)
    for index, (images, labels) in enumerate(testloader):
        if index == 2:
            break

    import dill
    with open('model_combined_datasets.dill', 'rb') as f:
        model = dill.load(f)

    with torch.no_grad():
        logps = model.forward(images)

    ps = torch.exp(logps)

    fig = view_classify(images[image_number], ps[image_number])
    return fig


    
def plot_confusion_matrix():
    import dill
    with open('y_valid.dill', 'rb') as h:
        true_values = dill.load(h)

    with open('pred_valid.dill', 'rb') as g:
        prediction_values = dill.load(g)
    
    fig, ax = plt.subplots()
    cm = confusion_matrix(true_values, prediction_values)
    tn, fp, fn, tp = cm.ravel()
    print("TN:",tn, " FP:", fp, " FN:", fn, " TP:", tp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Mask", "No Mask"])
    disp.plot(cmap='GnBu', ax=ax)
    ax.set_title("Performance on validation dataset")
    fig.show()
    
    


def view_classify(img, ps):
    ''' Function for viewing an image and its predicted classes.
    '''
    ps = ps.data.numpy().squeeze()
    plt.tight_layout()
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 3), ncols=2)
    ax1.imshow(img.permute(1, 2, 0))
    ax1.axis('off')
    ax2.barh(np.arange(2), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(2))
    ax2.set_yticklabels(["face_mask", "face_no_mask"])
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    return fig