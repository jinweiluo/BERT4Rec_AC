import numpy as np


class DataReader(object):
    def __init__(self, data_path, batch_size, max_len):
        super(DataReader, self).__init__()
        self.data_dir = data_path
        self.batch_size = batch_size
        self.max_len = max_len

    def read_file(self, file_path):
        f = open(file_path, "r")
        line = f.readline()
        while line:
            parsed_line = line
            yield parsed_line
            line = f.readline()

    def get_samples(self):
        def wraper():
            sample_count = 0
            for split_samples in self.read_file(self.data_dir):
                if sample_count % self.batch_size == 0:
                    src_ids = []
                    pos_ids = []
                    input_mask = []
                    mask_pos = []
                    mask_label = []
                split_samples = split_samples.split(";") 
                tmp_ids = split_samples[1].split(',')
                src_ids.append([int(x) for x in tmp_ids])
                tmp_pos = split_samples[3].split(',')
                pos_ids.append([int(x) for x in tmp_pos])
                tmp_mask = split_samples[2].split(',')
                input_mask.append([[int(x)] for x in tmp_mask])
                tmp_mask_pos = split_samples[4].split(',')
                mask_pos = mask_pos + [[int(x)+(sample_count % self.batch_size)*self.max_len] for x in tmp_mask_pos]
                tmp_label = split_samples[5].split(',')
                mask_label = mask_label + [[int(x)] for x in tmp_label]
                sample_count += 1
                if sample_count % self.batch_size == 0:
                    src_ids = np.array(src_ids)
                    pos_ids = np.array(pos_ids)
                    input_mask = np.array(input_mask)
                    mask_pos = np.array(mask_pos)
                    mask_label = np.array(mask_label)
                    yield src_ids, pos_ids, input_mask,  mask_pos, mask_label

        return wraper
