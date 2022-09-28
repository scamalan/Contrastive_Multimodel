from PIL import Image
from torchvision import transforms, utils
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader
import read_Pond
import numpy as np


class PondPair(VisionDataset):
    """Pond Dataset.
    
    x_train, y_train, x_val, y_val, x_test, y_test = read_Pond.get_Pond_data()
    data_s2 =x_train_s2;
    data_p =x_train_p;
    targets = y_train; """
    
    def __init__(self,root_dir,data_s2,data_p,target,train,transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            train:true for train data
        """
        super().__init__(root_dir,transform=transform,target_transform=None)
        
        self.root_dir = root_dir
        self.data_s2 = data_s2
        self.target = target
        self.train = train
        self.transform = transform
        self.data_p = data_p
        
        self.data_s2 = np.vstack(self.data_s2).reshape(-1, 4, 32, 32)
        self.data_s2 = self.data_s2.transpose((0, 2, 3, 1))  # convert to HWC
        
        self.data_p = np.vstack(self.data_p).reshape(-1, 4, 32, 32)
        self.data_p = self.data_p.transpose((0, 2, 3, 1)) 
        
    def __getitem__(self, index):
        img_s2, target = self.data_s2[index], self.target[index]
        img_p, target = self.data_p[index], self.target[index]
        img_s2 = Image.fromarray(img_s2)
        img_p = Image.fromarray(img_p)


        if self.transform is not None:
            pos_1 = self.transform(img_s2)
            pos_2 = self.transform(img_p)            
    
        return pos_1, pos_2, target  
    
    def __len__(self):
        return len(self.data_s2)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.479914744791592,0.485980244499565,0.473537796756613,0.240728921061493], [0.005253195647794,0.004134280736452,0.003441475836650,0.009876334371865])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.587571690362132,0.570905347169139,0.532160342688463,0.400953632514361], [0.009167041366748,0.007716662558361,0.005758577116085,0.008689966672193])])
