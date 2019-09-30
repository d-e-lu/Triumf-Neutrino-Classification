import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merges numpy arrays; outputs hdf5 file")
    parser.add_argument("input_file_list",
                        type=str, nargs=1,
                        help="txt file with a list of files to merge")
    parser.add_argument('output_file', type=str, nargs=1,
                        help="where do we put the output")  
    args = parser.parse_args()
    return args

def count_events(files):
    num_events = 0
    for f in files:
        data = np.load(f)
        num_events += data['event_id'].shape[0]
    return num_events
