import torch
from torch import nn
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)



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


class RMSELoss(nn.Module): # for regression
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

def eval_model(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               accuracy_fn,
               device:torch.device):
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


def make__random_predictions(model:torch.nn.Module, data: list, device: torch.device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            # Add an extra dimension and send sample to device
            sample = torch.unsqueeze(sample, dim=0).to(device)
            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)
            # Get prediction probability (logit -> prediction probability)
            # note: perform softmax on the "logits" dimension, not "batch" dimension
            # (in this case we have a batch size of 1)
            # so can perform on dim=0
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            # Get pred_prob of GPU for further calculations
            pred_probs.append(pred_prob.cpu())
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


def load_tensorboard(dataloader, model, optimizer, loss_fn, batch_size, lr):
    # tensorboard file
    running_loss = 0.0
    accuracy = 0
    epochs = NUM_EPOCHS
    #lr= LEARNING_RATE

    writer = SummaryWriter("runs/binary_classification")
    for epoch in range(epochs): # loop over the dataset multiple times
        print('Epoch-{0} lr: {1}'.format(epoch+1, lr))
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_logit = model(X) # Forward

            loss = loss_fn(y_logit.squeeze(dim=1), y) # calculate loss
            running_loss += loss.item() # save loss
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward() # backward loss
            optimizer.step() # optimize gradients

            pred_labels = torch.where(y_logit.squeeze(dim=1) < 0.5, 0, 1) # save prediction
            accuracy += torch.sum(pred_labels == y) # save accuracy

            if i % 100 == 99:  # every 100 mini-batches...
                steps = epoch * len(dataloader) + i # calculate steps
                batch = i * batch_size # calculate batch
                print("Training loss {:.3} Accuracy {:.3} Steps: {}".format(running_loss / batch, accuracy / batch, steps))
                # Save accuracy and loss to Tensorboard
                #writer.add_scalar("Training loss by steps", running_loss / batch, steps)
                #writer.add_scalar("Training accuracy by steps", accuracy / batch, steps)
                writer.add_scalar("training accuracy", accuracy*1000, epoch)
                writer.add_scalar("training loss", running_loss, epoch)
                writer.add_scalar("othergroup/epoch", epoch, epoch)

        print("Accuracy: {}/{} ({:.3} %) Loss: {:.3}".format(accuracy, len(dataloader), 100. * accuracy / len(dataloader.dataset), running_loss / len(dataloader.dataset)))

        # Save confusion matrix to Tensorboard
        writer.add_figure("Confusion matrix", confusion_matrix, epoch)
        running_loss = 0.0
        accuracy = 0

    print("Finished Training")
