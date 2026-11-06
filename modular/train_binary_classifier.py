import argparse
import torch
import data_setup, engine, model_builder
from config import BinaryConfig
from torch.utils.tensorboard import SummaryWriter
from utils import save_model


# SEED & target device
SEED = 42
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get some hyparameters.")
    parser.add_argument("--num_epochs", type=int, default=BinaryConfig.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BinaryConfig.BATCH_SIZE)
    parser.add_argument("--hidden_units", type=int, default=BinaryConfig.HIDDEN_UNITS)
    parser.add_argument("--learning_rate", type=float, default=BinaryConfig.LEARNING_RATE)
    parser.add_argument("--train_json", type=str, default=BinaryConfig.train_json)
    parser.add_argument("--test_json", type=str, default=BinaryConfig.test_json)
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "multiclass"])


    args = parser.parse_args()
    print(f"[INFO] Training a model for {args.num_epochs} epochs with batch size {args.batch_size} using {args.hidden_units} hidden units and a learning rate of {args.learning_rate}")
    print(f"[INFO] Training data file path: {args.train_json}")
    print(f"[INFO] Testing/validation data file path: {args.test_json}")

    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_json=args.train_json,
        test_json=args.test_json,
        load_data_func=data_setup.load_data_binary,
        batch_size=args.batch_size)

    model = model_builder.ImprovedNN(input_size=88,
                                    hidden_units=args.hidden_units,
                                    output_size=1).to(device)


    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate)
    writer = SummaryWriter("runs/binary_classification")
    engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=args.num_epochs,
             writer=writer,
             mode=args.mode,
             device=device)

    save_model(model=model,
                 target_dir="models",
                 model_name="binary_model_SimpleNN.pth")
