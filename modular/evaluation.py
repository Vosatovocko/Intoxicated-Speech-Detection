import data_setup, model_builder
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Arg:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def eval_model(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               accuracy_fn,
               device:torch.device):
    """ Evaluates a given model on a given dataset.
    Returns:
        (dict): Results of model making predictions on data_loader, Forwardpass only.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred.squeeze(dim=1), y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

            # Scale loss and acc
            loss /= len(dataloader)
            acc /= len(dataloader)
    return {"model_name": model.__class__.__name__, # Only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


def createConfusionMatrix(dataloader, model, classes):
    all_preds = [] # Save prediction
    all_trues = [] # Save ground truth
    model.eval()
    with torch.inference_mode():
        # iterate over data
        for X, y in tqdm(dataloader, desc="Making predictions"):
            y_logit = model(X) # Feed Network
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            # note: perform softmax on the "logits" dimension, not "batch" dimension
            # (in this case we have a batch size of 32)
            # so can perform on dim=1
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            all_preds.extend(y_pred.cpu().numpy())
            all_trues.extend(y.cpu().numpy())

    # confusion matrix
    cf_matrix = confusion_matrix(all_trues, all_preds)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=[0,1])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:None], index=[i for i in classes],
                    columns=[i for i in classes])
    #disp.plot(cmap=plt.cm.Blues)
    #return plt.show()
    # Create Heatmap
    plt.figure(figsize=(12, 7))
    return sn.heatmap(df_cm, annot=True).get_figure()


def load_tensorboard(dataloader, model, optimizer, loss_fn, batch_size, lr):
    # tensorboard file
    running_loss = 0.0
    true_positives_in_current_epoch = 0
    epochs = NUM_EPOCHS
    #lr= LEARNING_RATE

    writer = SummaryWriter("runs/binary_classification")
    for epoch in range(epochs): # loop over the dataset multiple times
        print('Epoch-{0} lr: {1}'.format(epoch+1, lr))
        for index_of_sample, (input_features, ground_truth_labels) in enumerate(dataloader):
            input_features, ground_truth_labels = input_features.to(device), ground_truth_labels.to(device)
            y_logit = model(input_features) # Forward

            loss = loss_fn(y_logit.squeeze(dim=1), ground_truth_labels) # calculate loss
            running_loss += loss.item() # save loss
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward() # backward loss
            optimizer.step() # optimize gradients

            pred_labels = torch.where(y_logit.squeeze(dim=1) < 0.5, 0, 1) # save prediction
            true_positives_in_current_batch = torch.sum(pred_labels == ground_truth_labels) # save accuracy
            true_positives_in_current_epoch += true_positives_in_current_batch

            if index_of_sample % 100 == 99:  # every 100 mini-batches...
                number_of_updates_so_far = epoch * len(dataloader) + index_of_sample # calculate steps
                number_of_samples_so_far_in_this_epoch = index_of_sample * batch_size # calculate batch, step in this epoch
                print("Training loss {:.3} Accuracy {:.3} Steps: {}".format(running_loss / number_of_samples_so_far_in_this_epoch, true_positives_in_current_epoch / number_of_samples_so_far_in_this_epoch, number_of_updates_so_far))
                # Save accuracy and loss to Tensorboard
                writer.add_scalar("Training loss by ", running_loss / number_of_samples_so_far_in_this_epoch, global_step=number_of_updates_so_far)
                writer.add_scalar("Training accuracy by ", true_positives_in_current_epoch / number_of_samples_so_far_in_this_epoch, global_step=number_of_updates_so_far)
                writer.add_scalar("othergroup/step", number_of_updates_so_far, global_step=number_of_updates_so_far)

        print("Accuracy: {}/{} ({:.3} %) Loss: {:.3}".format(true_positives_in_current_epoch, len(dataloader), 100. * true_positives_in_current_epoch / len(dataloader.dataset), running_loss / len(dataloader.dataset)))

        # Save confusion matrix to Tensorboard
        # writer.add_figure("Confusion matrix", confusion_matrix, epoch)
        running_loss = 0.0
        true_positives_in_current_epoch = 0

    print("Finished Training")


# New instance for classes.
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
# hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_UNITIS = 100
LEARNING_RATE = 0.001
INPUT_SIZE = 88
OUTPUT_SIZE = 1

train_json = "data/intoxicated-speech-detection/dataset/Functionals_train_30_withBak.json"
test_json = "data/intoxicated-speech-detection/dataset/Functionals_valid_30_withBak.json"

train_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_json=train_json,
    test_json=test_json,
    load_data_func=data_setup.load_data_binary,
    batch_size=BATCH_SIZE
    )

model = model_builder.SimpleNN(input_size=INPUT_SIZE,
                                hidden_units=HIDDEN_UNITIS,
                                output_size=OUTPUT_SIZE).to(device)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

"""
loaded_model = model_builder.SimpleNN(input_size=INPUT_SIZE,
                                hidden_units=HIDDEN_UNITIS,
                                output_size=OUTPUT_SIZE).to(device)
"""
# loaded_model.load_state_dict(torch.load(f="data/intoxicated-speech-detection/models/binary_model_SimpleNN.pth"))

#loaded_model = loaded_model.to(device)

"""loaded_model_results = eval_model(
    model=loaded_model,
    dataloader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)

"""

load_tensorboard(dataloader=train_dataloader,
                 model=model,
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 batch_size=BATCH_SIZE,
                 lr=LEARNING_RATE)

confusion_matrix=createConfusionMatrix(dataloader=test_dataloader,
                      model=model,
                      classes=("sober", "intoxicated"))
