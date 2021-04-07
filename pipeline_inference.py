import os
import csv
import time
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import util.io as uio


# +
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import StratifiedKFold
from core.loss import FocalLoss
from core.model import SnippetPolicyNetwork
from config.spn_config import Config
#import mkl
#mkl.set_num_threads(3)
torch.set_num_threads(3)

from tqdm import tqdm
from biosppy.signals import ecg
from biosppy.signals import tools


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# +


#load raw data

config = Config(model_name="FIT-SPN-E0.01-B0.0001-(2144)", dataset_name = "ICBEB", segmenter = "christov")

raw_input_path = os.path.join(config.root_dir,
                              config.data_dir,
                              config.dataset_name) + "/"

raw_tmp_path = os.path.join(config.root_dir,
                            config.data_dir,
                            config.dataset_name,
                            config.tmp_dir)

raw_data, raw_Y, raw_labels, classes = uio.load_formmated_raw_data(raw_input_path, raw_tmp_path)

# +
num_fold = 10

for fold in range(1, num_fold+1):
    one_label_y = np.argmax(raw_Y, axis=1)

    raw_sss = StratifiedKFold(n_splits=num_fold, random_state=0)

    raw_sss.get_n_splits(raw_data, one_label_y)

    raw_train_index_list = []
    raw_test_index_list = []

    for index, (train_index, test_index) in enumerate(raw_sss.split(raw_data, one_label_y)):
        raw_train_index_list.append(train_index)

        raw_test_index_list.append(test_index)

    raw_test_index = raw_test_index_list[fold-1]

    raw_testing_data = raw_data[raw_test_index]

    raw_testing_label = raw_Y[raw_test_index]
    
    model_path = os.path.join(config.root_dir,
                          config.output_dir,
                          config.model_dir,
                          config.model_name,
                          config.dataset_name,
                          str(fold))

    model = SnippetPolicyNetwork(hidden_size = config.hidden_size)

    model._ALPHA = 0.05
    
    print(model_path)

    model.load_state_dict(torch.load(model_path+"/model-best.pt"))

    model.cuda()

    model.eval()
    
    count = 0

    run_time = 0

    avg_earliness = 0

    for idx, raw_testing_ecg in tqdm(enumerate(raw_testing_data)):

        start = time.time()

        peaks = ecg.christov_segmenter(signal=raw_testing_ecg[:, 0], sampling_rate = 500)[0]

        if(len(peaks)<=1):
            la_peaks = ecg.christov_segmenter(signal=raw_testing_ecg[peaks[0]+500:, 0],
                                               sampling_rate = 500)[0]
            peaks = [(x+500) for x in la_peaks]

        hb = ecg.extract_heartbeats(signal=raw_testing_ecg,
                                    rpeaks=peaks,
                                    sampling_rate=500,
                                    before=1,
                                    after=1)

        rpeak_list = hb[1]

        raw_testing_ecg_corresponding_label = raw_testing_label[idx]

        input_snippet = np.array([hb[0]])

        predictions, t = model(input_snippet)

        end = time.time()

        run_time += (end - start)

        _, predicted = torch.max(predictions.data, 1)

        pred_position = t[0]

        pred_time_point = rpeak_list[pred_position]+499  # add a offset

        earliness = pred_time_point/raw_testing_ecg.shape[0]

        avg_earliness += earliness

        pred_label = predicted.cpu().numpy()[0]

        ground_truth = np.argmax(raw_testing_ecg_corresponding_label)

        if (raw_testing_ecg_corresponding_label[pred_label]): count+=1

        '''
        print(" Raw testing ECG shape: ", raw_testing_ecg.shape,
              "\n Snippet shape: ", hb[0].shape, 
              "\n R-peaks", rpeak_list,
              "\n Prediction Snippet Position: ", pred_position,
              "\n Real Prediction Point: ", pred_time_point,   # add a offset
              "\n Prediction Earliness: ", earliness,
              "\n Predicted Label:", pred_label,
              "\n Ground Truth Label:", ground_truth,
              "\n Classified Result: ", pred_label == ground_truth,
              "\n Real Time Accuracy: ", count/ raw_testing_data.shape[0],
              "\n Inference Time: ",(end - start),
              "\n ###################################################")
        '''

    print(count/raw_testing_data.shape[0])

    print(avg_earliness/raw_testing_data.shape[0])

    print(run_time/raw_testing_data.shape[0])
    
    print("=================================================")
# -




