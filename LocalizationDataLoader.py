from torch.utils.data import Dataset
import numpy as np
from scipy.interpolate import interp1d

class LocalizationDataLoader(Dataset):
    def __init__(self, path_dataset, labels):
        self.path_dataset = path_dataset
        self.labels = labels
        self.array_labels = np.load(self.labels)
        if self.path_dataset[-1]!='/':
            print("Error! Missing character / at the end")

    def __len__(self):
        return len(self.array_labels)

    def getName(self, idx, folder):
        name_idx = idx
        name_folder = folder
        for i in range(6-len(idx)):
            name_idx = '0'+name_idx
        if len(folder) == 1:
            name_folder = '0'+name_folder
        return name_idx, name_folder

    def __getitem__(self, index):
        label = self.array_labels[index][3]
        folder = str(int(self.array_labels[index][4]))
        idx = str(int(self.array_labels[index][2]))
        idx, folder = self.getName(idx, folder)
        X = np.fromfile(self.path_dataset+folder+'/velodyne/'+idx+'.bin', dtype=float)
        #print(self.path_dataset+folder+'/velodyne/'+idx+'.bin')
        #X = np.delete(X, np.arange(3, X.size, 4))
        #f = interp1d(np.arange(len(X)), X, 'linear')
        #X = np.apply_along_axis(f, 0, np.linspace(0, len(X)-1, num=10000))
        X = X[:(X.shape[0]-len(X)%4)]
        X = X.reshape(len(X)//4, 4)
        X = X[:, :3]
        X = X[np.random.choice(X.shape[0], 21000, replace=False), :]
        return X, label