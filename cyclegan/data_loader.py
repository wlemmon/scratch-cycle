import scipy
from glob import glob
import numpy as np
import soundfile as sf
import os
from functools import reduce
import scipy.signal

class DataLoader():
    def __init__(self, dataset_name, path_A, path_B, bitdepth = 8, duration=10000, rate=144000):
        self.dataset_name = dataset_name
        self.path_A = path_A
        self.path_B = path_B
        self.rate = rate
        self.duration = duration
        self.bitdepth = bitdepth

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/**' % (self.dataset_name, data_type))
        files_grabbed = [glob(os.path.join(path, e)) for e in ['*.pdf', '*.cpp']]
        print(files_grabbed)
        batch_images = np.random.choice(files_grabbed, size=batch_size)

        auds = []
        for sf_path in batch_images:
            aud, rate = self.sfread(sf_path)
            print(rate)
            if not is_testing:
                sf = scipy.misc.imresize(sf, self.sf_res)

                if np.random.random() > 0.5:
                    sf = np.fliplr(sf)
            else:
                sf = scipy.misc.imresize(sf, self.sf_res)
            auds.append(sf)

        sfs = np.array(sfs)/127.5 - 1.

        return sfs

    def load_batch(self, batch_size=1, is_testing=False):
        #data_type = "dev-clean" if not is_testing else "test-clean"
        base_path = './datasets/%s' % (self.dataset_name,)
        #print(os.path.join(base_path, self.path_A, "**/**", "*.flac"))
        files_A = reduce(lambda x, y: x+y, [glob(os.path.join(base_path, self.path_A, "**/**", e)) for e in ['*.flac', ]])
        files_B = reduce(lambda x, y: x+y, [glob(os.path.join(base_path, self.path_B, "**/**", e)) for e in ['*.flac', ]])
        self.n_batches = int(min(len(files_A), len(files_B)) / batch_size)
        total_samples = self.n_batches * batch_size
       
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(files_A, total_samples, replace=False)
        path_B = np.random.choice(files_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            sfs_A, sfs_B = [], []
            for sf_A, sf_B in zip(batch_A, batch_B):
                sf_A, sr_A = self.sfread(sf_A)
                sf_B, sr_B = self.sfread(sf_B)

                sf_A =  scipy.signal.resample(sf_A, int(float(len(sf_A)) / sr_A * self.rate ))
                sf_A = sf_A[self.duration]
                
                sf_B =  scipy.signal.resample(sf_B, int(float(len(sf_B)) / sr_B * self.rate ))
                sf_B = sf_B[self.duration]
                
                #sf_A = scipy.misc.imresize(sf_A, self.sf_res)
                #sf_B = scipy.misc.imresize(sf_B, self.sf_res)

                if not is_testing and np.random.random() > 0.5:
                        sf_A = np.fliplr(sf_A)
                        sf_B = np.fliplr(sf_B)

                sfs_A.append(sf_A)
                sfs_B.append(sf_B)

            sfs_A = np.array(sfs_A)/127.5 - 1.
            sfs_B = np.array(sfs_B)/127.5 - 1.

            yield sfs_A, sfs_B

    def load_sf(self, path):
        sf, sr = self.sfread(path)
        sf =  scipy.signal.resample(sf, int(float(len(sf)) / sr * self.rate ))
        sf = sf[self.duration]
        #sf = scipy.misc.imresize(sf, self.sf_res)
        #sf = sf/127.5 - 1.
        return sf[np.newaxis, :, :, :], self.rate

    def sfread(self, path):
        return sf.read(path)
