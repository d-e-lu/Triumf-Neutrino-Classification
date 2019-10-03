import os
import h5py
import argparse
import numpy as np

IMAGE_SHAPE = (40,40,38)
PMT_LABELS = "PMT label - Sheet3.csv"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merges numpy arrays; outputs hdf5 file")
    parser.add_argument("input_file_list",
                        type=str, nargs=1,
                        help="Path to input text file,\
                        each file on a different line.")
    parser.add_argument('output_file', type=str, nargs=1,
                        help="Path to output file.")  
    args = parser.parse_args()
    return args

def count_events(files):
    num_events = 0
    for f in files:
        data = np.load(f)
        num_events += data['event_id'].shape[0]
    return num_events

def GenMapping(csv_file):

    mPMT_to_index = {}
    with open(csv_file) as f:
        rows = f.readline().split(",")[1:]
        rows = [int(r.strip()) for r in rows]

        for line in f:
            line_split = line.split(",")
            col = int(line_split[0].strip())
            for row, value in zip(rows, line_split[1:]):
                value = value.strip()
                if value: # If the value is not empty
                    mPMT_to_index[int(value)] = [col, row]
    return mPMT_to_index

if __name__ == '__main__':
    
# -- Parse arguments
    config = parse_args()

    #read in the input file list
    with open(config.input_file_list[0]) as f:
        files = f.readlines()

    #remove whitespace 
    files = [x.strip() for x in files] 
     
    # -- Check that files were provided
    if len(files) == 0:
        raise ValueError("No files provided!!")
    print("Merging "+str(len(files))+" files")
    
    #print("Files are:")
    #print(files)
    
    
  # -- Start merging
    num_events = count_events(files)
    
    # Only use 32 bit precision
    dtype_events = np.dtype(np.float32)
    dtype_labels = np.dtype(np.int32)
    dtype_energies = np.dtype(np.float32)
    dtype_positions = np.dtype(np.float32)
    dtype_IDX = np.dtype(np.int32)
    dtype_PATHS = h5py.special_dtype(vlen=str)
    dtype_angles = np.dtype(np.float32)
    h5_file = h5py.File(config.output_file[0], 'w')
    dset_event_data = h5_file.create_dataset("event_data",
                                       shape=(num_events,)+IMAGE_SHAPE,
                                       dtype=dtype_events)
    dset_labels = h5_file.create_dataset("labels",
                                   shape=(num_events,),
                                   dtype=dtype_labels)
    dset_energies = h5_file.create_dataset("energies",
                                     shape=(num_events, 1),
                                     dtype=dtype_energies)
    dset_positions = h5_file.create_dataset("positions",
                                      shape=(num_events, 1, 3),
                                      dtype=dtype_positions)
    dset_IDX = h5_file.create_dataset("event_ids",
                                shape=(num_events,),
                                dtype=dtype_IDX)
    dset_PATHS = h5_file.create_dataset("root_files",
                                  shape=(num_events,),
                                  dtype=dtype_PATHS)
    dset_angles = h5_file.create_dataset("angles",
                                 shape=(num_events, 2),
                                 dtype=dtype_angles)
    
    # 22 -> gamma, 11 -> electron, 13 -> muon
    # corresponds to labelling used in CNN with only barrel
    #IWCDmPMT_4pi_full_tank_gamma_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_329.npz has an event with pid 11 though....
    #pid_to_label = {22:0, 11:1, 13:2}
    
    offset = 0
    
    mPMT_to_index = GenMapping(PMT_LABELS)
    for filename in files:
        data = np.load(filename)
        num_events_in_file = len(data['digi_hit_pmt'])
        x_data = np.zeros((num_events_in_file,)+IMAGE_SHAPE, 
                          dtype=dtype_events)
        digi_hit_pmt = data['digi_hit_pmt']
        digi_hit_charge = data['digi_hit_charge']
        digi_hit_time = data['digi_hit_time']
        for i in range(num_events_in_file):
            hit_pmts = digi_hit_pmt[i]
            charge = digi_hit_charge[i]
            time = digi_hit_time[i]
            for j in range(len(hit_pmts)):
                hit_mpmt = hit_pmts[j] // 19
                pmt_channel = hit_pmts[j] % 19
                index = mPMT_to_index[hit_mpmt]
                x_data[i, index[0], index[1], pmt_channel] = charge[j]
                x_data[i, index[0], index[1], pmt_channel + 19] = time[j]
        directions = data['direction']
        if 'IWCDmPMT_4pi_full_tank_gamma' in filename:
            labels = np.zeros(num_events_in_file)
            e_mass=0.51099895000
            energies=np.expand_dims(data['energy'],1)
            mag_momenta=np.sqrt(energies**2-e_mass**2)
            mag_momenta=np.expand_dims(mag_momenta,2)
            momenta=mag_momenta*directions
            momenta_sum=np.sum(momenta,axis=1)
            momenta_sum_adj=momenta_sum[:,[2,0,1]]
            momenta_sum_adj_mag=np.sqrt(np.sum(momenta_sum_adj**2,axis=1))
            directions_proper=momenta_sum_adj/np.expand_dims(momenta_sum_adj_mag,1)
            energies = np.sum(energies, axis=1).reshape(-1,1)
            positions = np.expand_dims(data['position'],1)[:,0,:].reshape(-1, 1,3)
        elif 'IWCDmPMT_4pi_full_tank_e-' in filename:
            labels = np.ones(num_events_in_file)
            energies = np.expand_dims(data['energy'],1)
            positions = np.expand_dims(data['position'],1)
            directions = directions.squeeze()
            directions_proper = directions[:,[2,0,1]]
        elif 'IWCDmPMT_4pi_full_tank_mu-' in filename:
            labels = np.full(num_events_in_file, 2)
            energies = np.expand_dims(data['energy'],1)
            positions = np.expand_dims(data['position'],1)
            directions = directions.squeeze()
            directions_proper = directions[:,[2,0,1]]
        else:
            raise ValueError("File {} is not electron, muon or gamma".format(filename))
        energies = energies.astype(dtype_energies)
        positions = positions.astype(dtype_positions)
        
        planar_mag=np.sqrt(np.sum(directions_proper[:,[0,1]]**2,axis=1))
        azimuthal=np.arctan2(directions_proper[:,1],directions_proper[:,0])
        polar=np.arctan2(directions_proper[:,2],planar_mag)
        azimuthal=azimuthal.reshape(-1,1)
        polar=polar.reshape(-1,1)
        angles=np.hstack((polar,azimuthal))
        
        PATHS = data['root_file'].astype(dtype_PATHS)
        IDX = data['event_id'].astype(dtype_IDX)
        
        offset_next = offset + num_events_in_file

        dset_event_data[offset:offset_next,:] = x_data
        dset_labels[offset:offset_next] = labels
        dset_energies[offset:offset_next,:] = energies
        dset_positions[offset:offset_next,:,:] = positions
        dset_PATHS[offset:offset_next] = PATHS
        dset_IDX[offset:offset_next] = IDX
        dset_angles[offset:offset_next,:] = angles
        
        offset = offset_next
        print("Finished file: {}".format(filename))
        
    print("Saving")
    f.close()
    print("Finished")
