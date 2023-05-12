"""
The code to be ultimately ran to execute training of our neural network.
The rough sketch is given in the "rough flow of work" part in the README.
"""

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from my_first_cnn import MyFirstCNN

def set_device():
  """
  Set the device. CUDA if available, CPU otherwise

  Args:
    None

  Returns:
    Nothing
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: For this notebook to perform best, "
        "if possible, in the menu under `Runtime` -> "
        "`Change runtime type.`  select `GPU` ")
  else:
    print("GPU is enabled in this notebook.")

  return device

#TODO MAKE INTO AN ACTUAL EXPLANATION OF THE CODE
# TAKES A MODEL, AN OPTIMIZER, THE TRAIN / TEST DATA, THE NUMBER OF EPOCHS, 
# BATCH SIZE AND LEARNING RATE TO TRAIN THE MODEL AND RETURN A TRAINED VERSION 
# COULD INCLUDE OPTIMIZERS AND LOSS FUNCTIONS AS VARIABLES IF I REALLY WANTED TO
def train(model : nn.Module, 
          train_data : Dataset, 
          test_data : Dataset,
          num_classes : int,
          num_epochs : int,
          batch_size : int, 
          learning_rate : float,
          device
          ):
    """
    Taking a model to be trained, 

    :param nn.Module model: _description_
    :param Dataset train_data: _description_
    :param Dataset test_data: _description_
    :param int num_classes: _description_
    :param int num_epochs: _description_
    :param int batch_size: _description_
    :param float learning_rate: _description_
    :param _type_ device: _description_
    :return _type_: _description_
    """
    
    # Initialization
    # first create dataloaders containing the data such that we can iterate on
    train_dloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dloader = DataLoader(dataset=test_data, batch_size=batch_size)
    
    # set the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    # store training / testing accuracy and loss over time
    train_acc, test_acc, train_loss, test_loss = [], [], [], []


    # Train loop per epoch
    for i in range(num_epochs):
        print("Epoch", i + 1, "!")
        # Train phase; initialization
        model.train()
        
        epoch_train_loss = 0
        correct_train_pred = 0

        it_train = iter(train_dloader)
        # Training loop per data batch
        # feed images as batch into the network to calculate predictions
        for _ in tqdm.tqdm(range(len(train_dloader))):
            images, labels = next(it_train)
            labels = labels.to(torch.int64)
            # make predictions and compute corresponding loss
            predictions = model(images)
            # print("\n pred", predictions, "; shape: ", predictions.shape)
            labelwise_probability = torch.zeros(len(labels), num_classes, dtype=predictions.dtype).to(device)
            # print("\n lwprob1", labelwise_probability, "; shape: ", labelwise_probability.shape)
            labelwise_probability[torch.arange(len(labels)), labels] = 1.0
            # print("\n lwprob2", labelwise_probability, "; shape: ", labelwise_probability.shape)
            loss = loss_fn(predictions, labelwise_probability)
            # print("\n loss", loss, "; shape: ", loss.shape)
            correct_train_pred += torch.sum(
                torch.argmax(predictions, dim=1) == labels
                ).item()
            # print("\n corr_preds ", correct_train_pred)
            epoch_train_loss += loss.item()
            # perform an optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Train end in epoch
        # track performance at epoch end
        train_acc.append(tr_acc_ep := correct_train_pred / len(train_data))
        train_loss.append(tr_los_ep := epoch_train_loss / len(train_data))
        print("training accuracy: ", tr_acc_ep, "; training loss: ", tr_los_ep)
        

        # Evaluate phase; initialization
        # evaluate and store performance to track extent of learning
        model.eval()

        epoch_test_loss = 0
        correct_test_pred = 0

        with torch.no_grad():
            # Evaluate loop per data batch
            for images, labels in test_dloader:
                labels = labels.to(torch.int64)
                predictions = model(images)            
                labelwise_probability = torch.zeros(len(labels), num_classes, dtype=predictions.dtype).to(device)
                labelwise_probability[torch.arange(len(labels)), labels] = 1.0
                loss = loss_fn(predictions, labelwise_probability)
                correct_test_pred += torch.sum(
                    torch.argmax(predictions, dim=1) == labels
                    ).item()
                epoch_test_loss += loss.item()
            
            
            # Test end in epoch
            # track performance at epoch end
            test_acc.append(te_acc_ep := correct_test_pred / len(test_data))
            test_loss.append(te_los_ep := epoch_test_loss / len(test_data))
            print("testing accuracy: ", te_acc_ep, "; testing loss: ", te_los_ep)
    
    # After all training / testing
    # nothing for now

    return model, train_acc, test_acc, train_loss, test_loss

if __name__ == "__main__":

    DEVICE = set_device()

    input_shape = (1, 28, 28)
    num_classes = 10
    path_to_source = "./kmnist"

    # preparing the data
    with np.load(path_to_source + "/" + "kmnist-train-imgs.npz") as data:
        train_imgs = torch.from_numpy(data["arr_0"]).to(torch.float32).to(DEVICE)
        train_imgs = train_imgs.unsqueeze(1) #add a dummy "channel" dimension
    with np.load(path_to_source + "/" + "kmnist-train-labels.npz") as data:
        train_labels = torch.from_numpy(data["arr_0"]).to(DEVICE)
    with np.load(path_to_source + "/" + "kmnist-test-imgs.npz") as data:
        test_imgs = torch.from_numpy(data["arr_0"]).to(torch.float32).to(DEVICE)
        test_imgs = test_imgs.unsqueeze(1) #add a dummy "channel" dimension
    with np.load(path_to_source + "/" + "kmnist-test-labels.npz") as data:
        test_labels = torch.from_numpy(data["arr_0"]).to(DEVICE)
    
    train_d : Dataset = TensorDataset(train_imgs, train_labels)
    test_d : Dataset = TensorDataset(test_imgs, test_labels)
    
    num_epochs = 60
    batch_size = 256
    l_r = 0.001

    hyperparameters = [
        {
            "num_epochs" : 5,
            "batch_size" : 512,
            "l_r" : 1e-3
        },
        {
            "num_epochs" : 5,
            "batch_size" : 512,
            "l_r" : 1e-4
        }
    ]

    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    
    for hparams in hyperparameters:
        # initialize a new model to train
        cnn = MyFirstCNN(input_shape=input_shape, output_size=num_classes).to(DEVICE)

        trained_model, train_acc, test_acc, train_loss, test_loss = train(
            model=cnn,
            train_data=train_d,
            test_data=test_d,
            num_classes=num_classes,
            num_epochs=hparams["num_epochs"],
            batch_size=hparams["batch_size"],
            learning_rate=hparams["l_r"],
            device=DEVICE
        )
        new_model_name = "./models/kmnist_model_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
        new_model_name += ("_num_epochs_" + str(hparams["num_epochs"]))  
        new_model_name += ("_batch_size_" + str(hparams["batch_size"]))
        new_model_name += ("_l_r_" + str(hparams["l_r"]))
        torch.save(
            trained_model.state_dict(), 
            new_model_name
        )
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    for i in range(len(train_accs)):
        train_acc_i = train_accs[i]
        test_acc_i = test_accs[i]
        train_loss_i = train_losses[i]
        test_loss_i = test_losses[i]

        plt.plot(range(len(train_acc_i)), train_acc_i, range(len(test_acc_i)), test_acc_i)
        plt.title("Train and test accuracy " + str(i))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()

        plt.plot(range(len(train_loss_i)), train_loss_i, range(len(test_loss_i)), test_loss_i)
        plt.title("Train and test loss " + str(i))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
