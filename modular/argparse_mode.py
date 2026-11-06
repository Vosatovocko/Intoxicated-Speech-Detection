import argparse
import torch
import data_setup, engine, model_builder
from torch.utils.tensorboard import SummaryWriter
from utils import save_model


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyparameters.")
# Choose which function want to run
FUNCTION_MAP = {"binary_classfication": data_setup.load_data_binary
                } # add functions in data_setup.
#parser.add_argument("command", choices=FUNCTION_MAP.keys())

parser.add_argument("--num_epochs", default=300, type=int,
                    help="the number of epochs to train for")
parser.add_argument("--batch_size", default=32, type=int,
                    help="number of samples per batch")
parser.add_argument("--hidden_units", default=100, type=int,
                    help="number of hidden units in hidden layers")
parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="learning rate to use for model")
parser.add_argument("--train_json", type=str, default="data/intoxicated-speech-detection/dataset/Functionals_train_30_withBak.json")
parser.add_argument("--test_json", type=str, default="data/intoxicated-speech-detection/dataset/Functionals_valid_30_withBak.json",
                    help="the file path to testing(validation) data")
parser.add_argument("--loss_fn", )


# Get out arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")

# Setup directories
train_json = args.train_json
test_json = args.test_json
print(f"[INFO] Training data file path: {train_json}")
print(f"[INFO] Testing/validation data file path: {test_json}")

# Setup load data function.
#select_func = FUNCTION_MAP[args.command]

# Setup target device
device = "cuda: " if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

#load_data_func = data_setup.load_data_binary

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_json=train_json,
    test_json=test_json,
    load_data_func=data_setup.load_data_binary,
    batch_size=BATCH_SIZE
    )

# Create model with help from model_builder.py
model = model_builder.SimpleNN(input_size=88,
                                hidden_units=HIDDEN_UNITS,
                                output_size=1).to(device)

# Set loss and optimizer
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)


writer = SummaryWriter("runs/binary_classification")
# Start training with engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             writer=writer)

# Confusion matrix


# Save the model with utils.py
save_model(model=model,
                 target_dir="models",
                 model_name="binary_model_SimpleNN.pth")

# Import and use the function in main script.
