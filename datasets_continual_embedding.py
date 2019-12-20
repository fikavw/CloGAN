from __future__ import print_function
import os
import os.path
import numpy as np
import torch

import torch.nn as nn

import torch.backends.cudnn as cudnn
import gzip
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader
import sys
import pickle
import random

import k_means_clustering as km


manualSeed=999
# if manualSeed == 0:
#     manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
cudnn.benchmark = True


class WrapDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *features and targets: tensors that have the same size of the first dimension.
    """

    def __init__(self, features, targets, transform=None):
        self.features = features
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img = self.features[index,...]
        lbl = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        
        return (img, lbl)

    def __len__(self):
        return self.features.size(0)    

 
    
class dataset_clustering(Dataset):
    
    ''' Dataset class used for MNIST, FASHION and E-MNIST, SVHN  for continual learning
    
    1. for Cifar and SVHN there are 3 color channels due to color 
    2. Performs clustering and embedding 
    
    '''
        
    raw_folder = 'raw'
    processed_folder = 'processed'
    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')

    def __init__(self, dataset_name, root, label,  train=True,  per_class=None, max_labels = 10,  
                 n_cluster=1, clustering_method='kcenters', embed_model= 'AE', 
                 normalize=False, transform=None, target_transform=None):

        """args:
        txt_file - path to the .txt file containing a list of images
        root_dir - directory of dataset root
        clustering_method --> can be kcenters, gmm, None
        embed_method --> can be AE, VAE, None, etc
        """

        self.dataset_name = dataset_name
        self.label = label ## subset of labels. or entire set
        self.train = train  # training set or test set
        self.normalize = normalize
        self.root = root
        self.transform = transform
        self.target_transform= target_transform
        self.softmax = nn.Softmax()
        self.n_cluster=n_cluster
        self.clustering_method = clustering_method ### object 
        self.embed_model = embed_model ### object 




        self.transform_embed=transforms.Compose([
                            transforms.ToPILImage(),
                               transforms.Scale(32),
                               transforms.ToTensor()])



        ## ========== dataset specs =======================================

        if self.dataset_name=='mnist':
            self.raw_folder = 'raw'
            self.processed_folder = 'processed'
            self.training_file = 'training.pt'
            self.test_file = 'test.pt'
            self.max_labels=10
            
            if self.train==True:
                self.per_class_max=6000
            else:
                self.per_class_max = 1000
                
        elif self.dataset_name=='emnist':
            split = 'letters'
            if split not in self.splits: ## this is for E-MNIST only....
                    raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                        split, ', '.join(self.splits),
                    ))

            self.split = split
            self.training_file = self._training_file(split)
            self.test_file = self._test_file(split)
            if self.split=='letters':
                self.max_labels=26
                
            elif self.split=='balanced':
                self.max_labels==47

            if self.train==True:
                self.per_class_max=4800
            else:
                self.per_class_max=800            

        elif self.dataset_name=='fashion':
            x_train, y_train, x_test, y_test = load_fashionmnist(self.root)
            if self.train:
                self.images_fix = x_train
                self.labels_fix = y_train
                self.per_class_max=6000
            else:
                self.images_fix = x_test
                self.labels_fix = y_test
                self.per_class_max = 1000
                
            self.max_labels=10

        elif self.dataset_name =='svhn':
            
            self.balance_svhn = True
            
            self.max_labels=10
            
            self.images_fix, self.labels_fix = load_SVHN(self.root, self.train, self.balance_svhn)

            self.per_class_max = self.labels_fix.shape[0] ## which should be 4984 or something like that 

        elif self.dataset_name == 'cifar10':
            
            self.max_labels=10
            
            self.images_fix, self.labels_fix = load_cifar10(self.root, self.train)

            if self.train==True:
                self.per_class_max  = 5000
            else:
                self.per_class_max = 1000
        
        if self.dataset_name=='mnist' or self.dataset_name=='emnist':
            if self.train:
                self.images_fix, self.labels_fix = torch.load(
                    os.path.join(self.root, self.processed_folder, self.training_file))
            else:
                self.images_fix, self.labels_fix = torch.load(
                    os.path.join(self.root, self.processed_folder, self.test_file))
    
            self.images_fix = torch.unsqueeze(self.images_fix, dim=1)
       
            
        ## ====================================================================
        if per_class is not None:
            self.per_class = per_class
        else:
            self.per_class= self.per_class_max

                    
        print('per_class',self.per_class)

        
        

        self.images = self.images_fix
        self.labels = self.labels_fix
        self.labels = self.labels.type(torch.LongTensor)
        self.superlabels = torch.ones(self.labels.shape[0]).type(torch.LongTensor)

        # =========== compute subset ========

        if len(self.label)<self.max_labels or self.per_class < self.per_class_max: ## subset of dataset
            
            if self.clustering_method is not None:
                print('dataset contains only a subset of %d per class and %d labels'%(self.per_class, len(self.label)))
                self.images, self.labels, self.superlabels = self.subset_select_clustering(self.label, self.per_class, self.n_cluster)
            else: ## if not doing clustering
                print('doing no clustering')
                self.images, self.labels = self.subset_select_simple(self.label, self.per_class)
#                print('baaaaah',self.labels)
                self.superlabels = torch.from_numpy(np.ones((self.labels.shape[0],)))
        # ===== pre-processing ============

        if self.normalize:
            self.update_normalization()


    @staticmethod
    def _training_file(split):
        return 'training_{}.pt'.format(split)

    @staticmethod
    def _test_file(split):
        return 'test_{}.pt'.format(split)


    def update_normalization(self):

        if self.normalize:

            self.mean = self.images.float().mean()/255
            self.stand = self.labels.float().std()/255

            self.transform = transforms.Compose([self.transform, transforms.Normalize((self.mean,),(self.stand,))])
        else:
            self.transform = transform


    def __len__(self):
        return self.images.size(0)


    
    def subset_select_clustering(self,  classes, per_class, n_cluster): ##apply every time to generate new subset. 
            
        ''' Generate optimal subset for buffer according to clustering chosen 
        
        Picks from new data to insert into buffer
        
        Here, need to actually perform clustering and keep superlabels
        
        1. classes --> list of labels 
        2. per_class is number of samples per class (int)
        3. n_cluster is number of clusters for clustering method (int)
        4. self.clustering_method is clustering object of specific type (obj)
        5. self.embed_model ---> embedding of data to go into clustering (trained AE for example)'''
        if self.clustering_method is not None:
            self.clustering_method.update_n_cluster(n_cluster)
        
        set_labels = list(set(classes))
    
        for c, l in enumerate(set_labels): ##per class 
    
            inds_old = np.where(self.labels_fix.numpy() == l)[0] ## indices pertaining to that class from total dataset (new data)
    
#             label_images = self.images_fix[torch.from_numpy(inds_old).long(),...].type(torch.FloatTensor)
#             label_labels = self.labels_fix[torch.from_numpy(inds_old).long()].type(torch.LongTensor)
            
            label_images = self.images_fix[torch.from_numpy(inds_old).long(),...]
            label_labels = self.labels_fix[torch.from_numpy(inds_old).long()]
            
            self.label_images = label_images
            
            print(c,l)
            
            
            
            
            if self.clustering_method is not None:
                if self.embed_model is not None:
                    print('embed data for clustering with: ', str(self.embed_model))
                    print('input to embedding', label_images.shape)

    #                 images_toembed_in = label_images.type(torch.FloatTensor)
    #                 labels_toembed_in = label_images.type(torch.LongTensor)
                    images_toembed_in = label_images
                    labels_toembed_in = label_images
                    ### wrap images in dataset to apply transforms necessary
                    dset = WrapDataset(images_toembed_in, labels_toembed_in, self.transform_embed)
                    d_wrapped = DataLoader(dset, batch_size=int(label_labels.shape[0]), shuffle=False)

                    images_to_embed, labels_to_embed = next(iter(d_wrapped))

                    self.images_to_embed = images_to_embed

                    ## then run forward embed in Dataparallell wrapped model
                    embedded_images = self.embed_model.module.embed(images_to_embed.cuda()) ### returns flattened 

                    embedded_images = embedded_images.data.cpu()
                    self.flatten_cluster = False

                else:
                    self.flatten_cluster = True
                    embedded_images = label_images

                    
                print('clustering')
#                print('clustering with: ', str(self.clustering_method))
                chosen_inds, superlabels = self.clustering_method.pick_kcentered_images(embedded_images.numpy(), per_class, self.flatten_cluster)

            else:
                print('No clustering')
                print('No Embedding, because embedding is only for clustering')
    
                inds = np.random.permutation(label_images.shape[0])
                chosen_inds = np.array(inds[0:per_class])
                superlabels = np.ones((per_class,))
                
    
            if c ==0:
    
                images_new = label_images[torch.from_numpy(chosen_inds).long(),...]
                labels_new = label_labels[torch.from_numpy(chosen_inds).long()]
                superlabels_new = torch.from_numpy(superlabels).long()
    
            else:
    
                print(superlabels_new.shape)
                images_new = torch.cat((images_new, label_images[torch.from_numpy(chosen_inds).long(),...]), dim=0)
                labels_new = torch.cat((labels_new, label_labels[torch.from_numpy(chosen_inds).long()]), dim=0)
                superlabels=torch.from_numpy(superlabels).type(torch.LongTensor)
                
                superlabels_new = torch.cat((superlabels_new, superlabels), dim=0)
                
#         images_new = images_new.long()
#         labels_new = labels_new.long()
#         superlabels_new = superlabels_new.long()
            
        return images_new, labels_new, superlabels_new
    
        
    
    
    def subset_select_simple(self, classes, per_class): ##apply every time to generate new subset. 
        
        ''' Generate optimal subset for buffer according to rule chosen '''

        set_labels = list(set(classes))
        
        print('Generating subset')

        for c, l in enumerate(set_labels): ##per label 

            inds_old = np.where(self.labels_fix.numpy() == l)[0]

            label_images = self.images_fix[torch.from_numpy(inds_old).long(),...]
            label_labels = self.labels_fix[torch.from_numpy(inds_old).long()]
            
#            print(c,l)


#            print('choice for subset/add = none')

            inds = np.random.permutation(inds_old.shape[0])

            self.chosen_inds = np.array(inds[0:per_class])


            if c ==0:

                images_new = label_images[torch.from_numpy(self.chosen_inds).long(),...]
                labels_new = label_labels[torch.from_numpy(self.chosen_inds).long()]

            else:

                images_new = torch.cat((images_new, label_images[torch.from_numpy(self.chosen_inds).long(),...]), dim=0)
                labels_new = torch.cat((labels_new, label_labels[torch.from_numpy(self.chosen_inds).long()]), dim=0)
            # self.labels_perlabel = self.labels_perlabel.type(LongTensor)

        return images_new, labels_new

    

    def subset_select_make_room(self, per_class_mem):
    
        ''' Choose Subset to keep in buffer with per_class_mem 
            Return reduced buffer with room for new images 
        '''
        
        
        set_labels = list(set(self.labels.numpy()))
    
        for c, l in enumerate(set_labels): ##per class
            
            print('================== Appending + Making Room Buffer ================')
            
            print(c,l)
    
            inds_old = np.where(self.labels.numpy() == l)[0] ##indices of that class 
    
            label_images = self.images[torch.from_numpy(inds_old).long(),:,:]
            label_labels = self.labels[torch.from_numpy(inds_old).long()]
            superlabels_labels = self.superlabels[torch.from_numpy(inds_old).long()] ##superlabels of that class 
            
            
            print('num per label' + str(label_images.shape[0]))
    
            
            if self.clustering_method is not None: ## if previously doing clustering
    
                print('using superlabels generated with clustering')
                chosen_inds, superlabels_chosen = km.pick_fromgroups2(superlabels_labels.numpy(), per_class_mem)
                
            else: ## if never doing clustering
                print('no superlabels')
                inds = np.random.permutation(label_images.shape[0])
                chosen_inds = np.array(inds[0:per_class_mem])
                superlabels_chosen = np.ones((per_class_mem,))
    
                
            if c ==0:
    
                images_keep = label_images[torch.from_numpy(chosen_inds).long(),:,:]
                labels_keep = label_labels[torch.from_numpy(chosen_inds).long()]
                superlabels_keep = torch.from_numpy(superlabels_chosen).long()
    
            else:
    
                images_keep = torch.cat((images_keep, label_images[torch.from_numpy(chosen_inds).long(),:,:]), dim=0)
                labels_keep = torch.cat((labels_keep, label_labels[torch.from_numpy(chosen_inds).long()]), dim=0)
                superlabels_keep = torch.cat((superlabels_keep, torch.from_numpy(superlabels_chosen).long()), dim=0)
    
        
        return images_keep, labels_keep, superlabels_keep
    
    
    
    def append_memory(self, task, labels, per_class_mem, embed_model): ##append new old labels, at each task
            
        '''Call subset_select_clustering on new data
            Call subset_select_make_room for old data, to make room for new --> select exemplars to remove according to superlabel 
            clustering_method object is created previously
        '''
    
        self.embed_model = embed_model ## update embedder for new task
        
        
        ## ======= call subset_select_clustering  for new data  =====================
        self.data_m_new, self.labels_m_new, self.superlabels_m_new = self.subset_select_clustering(labels, per_class_mem, self.n_cluster)
    
        
        if task ==1:
            
            ## fill buffer for first time 
            self.images = self.data_m_new
            self.labels = self.labels_m_new
            self.superlabels = self.superlabels_m_new
            
            ##shuffle 
            inds = np.random.permutation(self.images.shape[0])
            
            self.images = torch.from_numpy(self.images[inds,:,:])
            
            self.labels = torch.from_numpy(self.labels[inds]).long()
            
            self.superlabels = torch.from_numpy(self.superlabels[inds]).long()
            
            print('Buffer_mem_img size at init:' + str(self.images.size(0)))
            print('Init per_class_mem size:' + str(per_class_mem))
            
        elif task >1:
                        
            ## transform to numpy
            self.data_m_new = self.data_m_new.numpy()
            self.labels_m_new = self.labels_m_new.numpy()
            
            print('Buffer before make_space', self.images.shape)
    
            ##filter buffer to keep best options 
            self.images, self.labels, self.superlabels = self.subset_select_make_room(per_class_mem) ## with new per_class number keep diversity of superlabels 
            
            print('Buffer after make_space', self.images.shape)
            
            ##append the new old labels
            self.images = np.append(self.images.numpy(), self.data_m_new, axis=0)
            self.labels = np.append(self.labels.numpy(), self.labels_m_new, axis=0)
            self.superlabels = np.append(self.superlabels.numpy(), self.superlabels_m_new, axis=0)
    
            ##shuffle 
            inds = np.random.permutation(self.images.shape[0])
            
            self.images =  torch.from_numpy(self.images[inds,:,:])
            self.labels =  torch.from_numpy(self.labels[inds]).long()
            self.superlabels =  torch.from_numpy(self.superlabels[inds]).long()
    
    
        if self.normalize:
            self.update_normalization()
            
    

    def __getitem__(self, idx):
        img = self.images[idx,...]
        lbl = self.labels[idx]

        if self.transform is not None: ##input the desired tranform 

            img = self.transform(img)
            
                        
            if self.dataset_name=='emnist':
                
                img = torch.from_numpy(rotate(img.cpu().data.view(img.shape[-1], img.shape[-1]).numpy()))
                
                img = img.unsqueeze(0)
            
        if self.target_transform is not None:
            
            lbl = self.target_transform(lbl)
            
        return img, lbl  
    
    


# ========================== dataset specific functions =============================




def load_fashionmnist(data_dir):
    
    '''Pre-Process FASHION MNIST'''
    data_dir = os.path.join(data_dir, 'fashion')

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    ##one-hot labels 
    # y_vec = np.zeros((len(y), 10), dtype=np.float)
    # for i, label in enumerate(y):
    #     y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    # y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)
    
    x_train = X[:60000,:]
    y_train = y[:60000]
    x_test = X[-10000:,:]
    y_test = y[-10000:]
    
    return x_train, y_train, x_test, y_test

# =


    
def load_SVHN(root, train=True, balance=True):

    root = os.path.expanduser(root)
    
    if train==True:
        filename = "train_32x32.mat"
    else:
        filename = "test_32x32.mat"

        
    import scipy.io as sio
    # reading(loading) mat file as array
    loaded_mat = sio.loadmat(os.path.join(root, filename))

    data = loaded_mat['X']
    # loading from the .mat file gives an np array of type np.uint8
    # converting to np.int64, so that we have a LongTensor after
    # the conversion from the numpy array
    # the squeeze is needed to obtain a 1D tensor
    labels = loaded_mat['y'].astype(np.int64).squeeze()

    # the svhn dataset assigns the class label "10" to the digit 0
    # this makes it inconsistent with several loss functions
    # which expect the class labels to be in the range [0, C-1]
    np.place(labels, labels == 10, 0)
    data = np.transpose(data, (3, 2, 0, 1))

    if train==True:
        max_ind_b = 4948
    else:
        max_ind_b = 1595
    
    
    if balance==True:
        
        inds_b = []
        random.seed(999)

        for i in range(10):

            arr = np.where(labels==i)[0]
            np.random.shuffle(arr)
            inds = arr[:max_ind_b]
            inds_b.extend(list(inds))

        inds_b = np.array(inds_b)
        inds_b = inds_b.astype(int)

        np.random.shuffle(inds_b)

        data = data[inds_b,...]
        labels = labels[inds_b]
        
        
    data = torch.from_numpy(data).type(torch.FloatTensor)
    # y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)
        
    return data, labels


def load_cifar10(root, train=True):
    
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    root = os.path.expanduser(root)

    if train:
        downloaded_list = train_list
    else:
        downloaded_list = test_list

    data = []
    labels = []

    # now load the picked numpy arrays
    for file_name, checksum in downloaded_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                labels.extend(entry['labels'])
            else:
                labels.extend(entry['fine_labels'])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC

    data = np.array(data)
    labels = np.array(labels)
    
    
    data = np.transpose(data, (0, 3, 1, 2))
    
    data = torch.from_numpy(data).type(torch.FloatTensor)
    # y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)
    
    return data, labels
            

def rotate(img):
# Used to rotate imagesfor E-MNIST
    flipped = np.fliplr(img)

    return np.rot90(flipped)
