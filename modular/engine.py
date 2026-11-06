import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module,
               dataloader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               mode: str,
               device: torch.device) -> Tuple[float,float]:
    """Trains a PyTorch model for a single epoch.
    Turns a target PyTorch model to training mode and then runs throgh
    all of the required training steps (forwardpass, loss, optimizer step)
    Returns a tuple of training loss and training accuracy metrics."""


    model.train()

    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (input_features, ground_truth_labels)in enumerate(dataloader):
        # Send data to target device
        input_features, ground_truth_labels = input_features.to(device), ground_truth_labels.to(device)
       # print(X.shape)

        y_logit = model(input_features)

        loss = loss_fn(y_logit.squeeze(dim=1), ground_truth_labels) # calculate loss
        train_loss += loss.item() # save loss
        optimizer.zero_grad() # zero the parameter gradients
        loss.backward() # backward loss
        optimizer.step() # optimize gradients

        # calculate and accumulate accuracy metric across all batches
        if mode == "multiclass":
            pred_labels = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
        elif mode == "binary":
            pred_labels = torch.where(y_logit.squeeze(dim=1) < 0.5, 0, 1)

        train_acc += (pred_labels == ground_truth_labels).sum().item()/len(y_logit)

    # Adjusrt metircs to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              mode: str,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.
    Turns a target PyTorch model to  "eval" mode and then performs a forward pass only on a testing dataset"""

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            #print(X.shape)

            # Forward pass
            test_pred_logits = model(X)

            #print(test_pred_logits)
            #print(y)

            # Calculate and accumulate loss
            loss = loss_fn(test_pred_logits.squeeze(dim=1), y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            if mode == "multiclass":
                test_pred_labels = test_pred_logits.argmax(dim=1)
            elif mode == "binary":
                test_pred_labels = torch.where(test_pred_logits.squeeze(dim=1) < 0.5, 0, 1)

            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          writer: SummaryWriter,
          mode: str,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a Pytorch model.
    Passes a target PyTorch models throgh train_step() and test_step() functions
    for a number of epochs, training and testing the model in the same epoch loop.

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
    """

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           mode=mode,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        mode=mode,
                                        device=device)



        writer.add_scalar("Training loss", train_loss, global_step=epoch)
        writer.add_scalar("Training accuracy", train_acc, global_step=epoch)
        writer.add_scalar("Test loss", test_loss, global_step=epoch)
        writer.add_scalar("Test accuracy", test_acc, global_step=epoch)

        tqdm.write(
            f"Epoch: {epoch+1} | "
            f"train_acc: {train_acc * 100:.2f}% | "
            f"train_loss: {train_loss:.3f} | "
            f"test_acc: {test_acc* 100:.2f}% | "
            f"test_loss: {test_loss:.3f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
