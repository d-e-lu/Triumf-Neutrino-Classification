import os
import h5py
import numpy as np

import utils

if __name__ == '__main__':
    config = utils.parse_args()
    with(open(config.input_file_list[0])) as f:
        files = f.readlines()
    files = [x.strip() for x in files]
    if len(files) == 0:
        raise ValueError("No files provided!!")
    print("Merging "+str(len(files))+" files")
    num_events = utils.count_events(files)
    
    h5_file = h5py.File(config.output_file[0], 'w')
    dset_labels = h5_file.create_dataset("labels",
                                   shape=(num_events,),
                                   dtype=np.int32)
    dset_PATHS = h5_file.create_dataset("root_files",
                                  shape=(num_events,),
                                  dtype=h5py.special_dtype(vlen=str))
    dset_IDX = h5_file.create_dataset("event_ids",
                                shape=(num_events,),
                                dtype=np.int32)
    dset_event_data = h5_file.create_dataset("event_data",
                                       shape=(num_events,),
                                       dtype=np.float32)
    dset_energies = h5_file.create_dataset("energies",
                                     shape=(num_events, 1),
                                     dtype=np.float32)
    dset_positions = h5_file.create_dataset("positions",
                                      shape=(num_events, 1, 3),
                                      dtype=np.float32)
    i = 0
    
    # 22 -> gamma, 11 -> electron, 13 -> muon
    # corresponds to labelling used in CNN with only barrel
    pid_to_label = {22:0, 11:1, 13:2}
    
    offset = 0

    for filename in files:
        data = np.load(filename)
        
        labels = list(map(pid_to_label.get, data['pid']))
        """ 
        offset_next = offset + event_data.shape[0]

        dset_event_data[offset:offset_next,:] = event_data
        dset_labels[offset:offset_next] = labels
        dset_energies[offset:offset_next,:] = energies
        dset_positions[offset:offset_next,:,:] = positions
        dset_PATHS[offset:offset_next] = PATHS
        dset_IDX[offset:offset_next] = IDX
        
        offset = offset_next
        """
    f.close()
    
