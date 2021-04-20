import os
import csv
import wandb
import torch
import random
import numpy as np
import util.io as uio

from tqdm import tqdm
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from core.model import SnippetPolicyNetwork
from config.spn_config import Config
import mkl
mkl.set_num_threads(3)
torch.set_num_threads(3)


def execute(config, training_data, training_label, testing_data, testing_label, 
            testing_length = None, testing_index = None,
            offset = 499, wandb=None):

    model = SnippetPolicyNetwork(hidden_size = config.hidden_size)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[20,40,60,80],
                                                     gamma=0.5,
                                                     last_epoch=-1)
    
    result_model = None
    
    for epoch in range(config.epoch_size):
        
        training_correct = 0
        
        model.train()
        model._ALPHA = 0.01
        # shuffle the training sample
        num_batch = int(len(training_data)/config.batch_size)
        random_sort = np.arange(training_data.shape[0])
        np.random.shuffle(random_sort)
        training_data = training_data[random_sort]
        training_label = training_label[random_sort]

        training_stopping_point_array = []

        count = 0

        for idx in tqdm(range(num_batch)):
            
            #batch snippet loading
            X = training_data[idx*config.batch_size : idx * config.batch_size+config.batch_size]
            
            #batch label loading
            input_label = torch.from_numpy(
                training_label[idx*config.batch_size:idx*config.batch_size+config.batch_size]
            ).cuda()
            
            # transfer the one-hot label to a certain label
            _, input_label = torch.max(input_label, dim=1)
            
            # feed into the model, return a classification result and a prediction time point
            predictions, t = model(X)
            _, predicted = torch.max(predictions.data, dim=1)
            training_stopping_point_array.append(t)

            # calculate loss
            loss, c, r, b, p = model.applyLoss( predictions, input_label, beta=config.beta )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #calculate correct samples
            training_correct += (predicted == input_label.squeeze()).sum().cpu().detach().numpy()
            
            if (idx % 10 == 0 and wandb is not None):
                wandb.log({'training/loss': loss.cpu().detach().numpy(),
                           'training/classification_loss': c.cpu().detach().numpy(),
                           'training/regression_loss': r.cpu().detach().numpy(),
                           'training/baseline_loss': b.cpu().detach().numpy(),
                           'training/time_penalty_loss': p.cpu().detach().numpy()
                           })
        
        #learning rate decay
        scheduler.step()
        
        training_accuracy = training_correct/len(training_label)
        
        if (wandb is not None):
            wandb.log({'epoch': epoch,
                       'training/stopping_point': np.array(training_stopping_point_array).mean(),
                       'training/accuracy': training_accuracy})

        print("epoch: ", epoch, 
              "loss: ", loss.cpu().detach().numpy(), 
              "accuracy: ", training_accuracy, 
              "haulting point:", np.array(training_stopping_point_array).mean())

        
        ''' Testing '''
        model.eval()
        model._ALPHA = 0.05
        testing_correct = 0
        testing_pred = []
        testing_true = []
        testing_stopping_point_array = []
        testing_earliness_array = []
        testing_length_array = []
        testing_batch_size = 1
        
        for idx, row in tqdm(enumerate(testing_data)):

            X = testing_data[idx:idx+testing_batch_size]
            
            testing_input_label = torch.from_numpy(testing_label[idx:idx+testing_batch_size]).cuda()
            
            testing_input_label_list = testing_label[idx:idx + testing_batch_size]
            
            _, testing_input_label = torch.max(testing_input_label, dim=1)
            
            predictions, t = model(X)

            '''
            Earliness calculation for a single sample
            1. use the stopping point to map the original snippet
            2. calculate the position of the stopping snippet in the original ecg
            '''
            
            testing_stopping_point_array.append(t[0])
            
            testing_stopping_point = t[0]
            
            testing_early_position = testing_index[idx][testing_stopping_point]+offset
                                                
            testing_earliness = testing_early_position / testing_length[idx]

            testing_earliness_array.append(testing_earliness)
            
            '''
            Correct sample calculation (Multi-class)
            1. check the classification result is in the label list
            2. return the correct sample
            '''
            
            _, predicted = torch.max(predictions.data, dim=1)

            for index, val in enumerate(predicted):
                
                testing_pred.append(val.cpu().detach().numpy())
                #classification results
                
                if(testing_input_label_list[0][val.cpu().detach().numpy()]):
                    testing_correct += 1
                    testing_true.append(val.cpu().detach().numpy())
                    #if correct return the true label
                else:
                    testing_true.append(testing_input_label[index].cpu().numpy())
                    #if incorrect return the true label
        
        testing_stopping_point_array = np.array(testing_stopping_point_array)

        testing_accuracy = testing_correct/len(testing_label)
        
        testing_avg_earliness = np.mean(testing_earliness_array)
        
        f1 = np.mean(f1_score(testing_true, testing_pred, average=None))
        
        recall = recall_score(testing_true, testing_pred, average='macro', zero_division=1)
        
        precision = precision_score(testing_true, testing_pred, average='macro')
        
        if (wandb is not None):

            wandb.log({'epoch': epoch,
                       'testing/accuracy': testing_accuracy,
                       'testing/haulting_point': np.mean(testing_stopping_point_array),
                       'testing/earliness': testing_avg_earliness,
                       'testing/f1': f1,
                       'testing/recall': recall,
                       'testing/precision': precision
                       })

        print("Testing accuracy: ", testing_accuracy, 
              "Earliness: ", testing_avg_earliness,
              "F1-score: ",f1,
              "Recall: ", recall,
              "Precision: ", precision)
        
        result_model = model
        
    return result_model

if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    num_fold = 10
    
    for fold in range(1,2):

        config = Config(message="schduler = [20,40,60,80]",
                        model_name="SPN-B0.0001",
                        hidden_size=256,
                        fold=fold, 
                        beta=0.0001, 
                        batch_size=32, 
                        epoch_size=100,
                        snippet_name = "christov_checkup.pickle")

        input_folder = os.path.join(config.root_dir,
                                    config.data_dir,
                                    config.dataset_name,
                                    config.snippet_dir,
                                    config.snippet_name)

        model_path = os.path.join(config.root_dir,
                                  config.output_dir,
                                  config.model_dir,
                                  config.model_name,
                                  config.dataset_name,
                                  str(config.fold))

        log_path = os.path.join(config.root_dir,
                                config.output_dir,
                                config.wandb_dir,
                                config.model_name,
                                config.dataset_name,
                                str(fold))
        
        uio.check_folder(model_path)

        uio.check_folder(log_path)

        X, Y, I, L = uio.load_snippet_data(input_folder)
        
        y = np.argmax(Y, axis=1)
        
        sss = StratifiedKFold(n_splits=num_fold, shuffle=False)

        sss.get_n_splits(X, y)
        
        for index, (train_index, test_index) in enumerate(sss.split(X, y)):
            if(index is (fold-1)):
                print("Runing Fold:",fold)
                break

        training_data, testing_data = X[train_index], X[test_index]

        training_label, testing_label = Y[train_index], Y[test_index]
        
        training_index, testing_index = I[train_index], I[test_index]
            
        training_length, testing_length = L[train_index], L[test_index]
       
        result_model = execute(config, 
                               training_data, 
                               training_label,
                               testing_data, 
                               testing_label, 
                               testing_index = testing_index,
                               testing_length = testing_length,
                               wandb=None)
        
        torch.save(result_model.state_dict(), model_path+"/model.pt")
