import os
import numpy as np
import util.io as uio
from tqdm import tqdm
from biosppy.signals import ecg
from biosppy.signals import tools
from config.spn_config import Config

def execute(config):

    input_path = os.path.join(config.root_dir,
                              config.data_dir,
                              config.dataset_name) + "/"

    tmp_path = os.path.join(config.root_dir,
                            config.data_dir,
                            config.dataset_name,
                            config.tmp_dir)

    output_path = os.path.join(config.root_dir,
                               config.data_dir,
                               config.dataset_name,
                               config.snippet_dir)

    data, Y, labels, classes = uio.load_formmated_raw_data(input_path, tmp_path)
    
    print("Input path:", input_path)
    print(" Data shape: ", data.shape,
          "\n Label shape: ", Y.shape,
          "\n Classes: ", classes)
    
    processed_data = []
    output_data = []
    output_label = []
    filtered_data = []
    raw_length = []
    hb_index = []
    output_length = []
    output_index = []
        
    for raw_row in tqdm(data):
        
        peaks = []
                
        if (config.segmenter is "christov"):
            peaks = ecg.christov_segmenter(signal=raw_row.T[0],
                                           sampling_rate = config.sampling_rate)[0]
            
            if(len(peaks)<=1):
                la_peaks = ecg.christov_segmenter(signal=raw_row[peaks[0]+500:, 0],
                                           sampling_rate = config.sampling_rate)[0]
                peaks = [(x+500) for x in la_peaks]
        elif (config.segmenter is "hamilton"):
            peaks = ecg.hamilton_segmenter(signal=raw_row.T[0],
                                           sampling_rate = config.sampling_rate)[0]
        else:
            peaks = ecg.gamboa_segmenter(signal=raw_row.T[0], 
                                         sampling_rate = config.sampling_rate)[0]
            
        hb = ecg.extract_heartbeats(signal=raw_row,
                                    rpeaks=peaks,
                                    sampling_rate=config.sampling_rate,
                                    before=1,
                                    after=1)
        
        raw_length.append(len(raw_row))
        hb_index.append(hb['rpeaks'])
        processed_data.append(hb[0])
             
    for idx, row in tqdm(enumerate(processed_data)):

        if(len(row) < 1):
            #check if incorrect segmentation
            print(idx, '->', row.shape)
        else:
            output_data.append(row)
            output_label.append(Y[idx])
            output_index.append(hb_index[idx])
            output_length.append(raw_length[idx])
    
    output_data = np.array(output_data)
    output_label = np.array(output_label)
    output_index = np.array(output_index)
    output_length = np.array(output_length)

    output_dict = {
        "data": output_data,
        "label": output_label,
        "index": output_index,
        "length": output_length
    }

    uio.check_folder(output_path)
    
    uio.save_pkfile(output_path+"/"+config.snippet_name, output_dict)

if __name__ == "__main__":

    config = Config(snippet_name="christov_checkup.pickle", dataset_name = "ICBEB", segmenter = "christov")

    execute(config)


