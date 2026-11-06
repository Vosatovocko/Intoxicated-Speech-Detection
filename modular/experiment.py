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
