'''To run script:

python3 -i CloGAN_main.py --dataset_name 'mnist' --dataroot '/lab/arios/data/MNIST' --main_out_path '/lab/arios/Documents/Results/CloGAN_results' --fraction_buff 0.001 --save_generated_images

python3 -i CloGAN_main.py --dataset_name 'svhn' --dataroot '/lab/arios/data/SVHN' --main_out_path '/lab/arios/Documents/Results/CloGAN_results' --fraction_buff 0.001 --save_generated_images

'''


from __future__ import print_function
import os
import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import argparse


import model_GAN_continual
import datasets_continual_embedding as dset_cluster
import generate_functions as gen
from main_cyclic_onlyACGAN import *
from embedding_training import *
import k_means_clustering as clu

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore")


''' Main Script to train CloGAN with chosen Dataset and buffer usage

1. GAN used is an ACGAN. Can use any conditional GAN
2. Dynamic Buffer Construction is done with per-class Kcenters on images directly.

'''


parser = argparse.ArgumentParser(description='CloGAN Training')
parser.add_argument('--dataset_name', type=str,  help='Dataset to Use with CloGAN. can be: mnist, fashion, svhn and emnist')
parser.add_argument('--version', type=int, default=1, help='version of experiment run with same hyperparameter.')
parser.add_argument('--main_out_path', help='path to folder where all result folders will be stored (each with a appropriate name)')
parser.add_argument('--manualSeed', type=int,  default=999, help='seed')
parser.add_argument('--workers', type=int,  default=4, help='number of workers for data loading pytorch')
parser.add_argument('--embedding',  action='store_true',  help='If doing embedding')
parser.add_argument('--latent_AE_size',type=int,  default=100, help='If doing embedding')
parser.add_argument('--dataroot', type=str,  help='path to where dataset is stored. For FASHION, MNIST, EMNIST and SVHN')
parser.add_argument('--fraction_buff', type=float,  help='Memory allotment of dynamic buffer. Percent of dataset to store in buffer. Buffer will have fized size throughout training')


parser.add_argument('--batchSize', type=int, default=50, help='batchsize for CloGAN training')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for CloGAN training')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM for CloGAN training')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM for CloGAN training')

parser.add_argument('--save_generated_images',action='store_true', help='If saving generated images for visualization with save_image_freq (saves a batch of 64)')
parser.add_argument('--save_image_freq', type=int, default=5, help='Frequency of saving generated image output')


opt = parser.parse_args()



if opt.manualSeed == 0:
    manualSeed = random.randint(1, 10000)
else:
    manualseed=opt.manualSeed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
cudnn.benchmark = True

gpus = torch.cuda.is_available()


# =========== filtering =============
## set to conditional filtering
main_filtering = True
second_filtering = 'none' ## drs or soft
version_gen_main=1
soft_percentile=80
gamma_drs=80
burnin_samples = 10000
batchsize_gen_burn = 100


# ========= other params ===============
generated_batch=64 ## batch for visualization
buff_real=None ## can optionally set a fixed size of buffer independent of dataset size
validate_classifier = True ## test classifier during training, just for results plotting
save_S_perepoch=True ##save only last epoch of each task
save_C_perepoch=True ##save only last epoch of each task
use_buff= True ##if divisive
save_best=False ## if saving best epoch per task
save_last=True ## if saving last

# ========= general ACGAN params =========
nz=110
ndf=64 ## num filters
params_model={}
params_model['nz']=nz
params_model['ndf']=ndf
model_type='ACGAN' ## main CloGAN conditional generative model
imageSize=32
dataset_param = model_type
betas=(opt.beta1, opt.beta2)


# =============embedding==============
## set to none typically, memory saving unless embedding is very good (still testing)
if opt.embedding==True:
    embedding='AE_conv3' ## name of embedding model
else:
    embedding =None
n_cluster=10 ##only


if opt.dataset_name == 'svhn':
    epoch_max = 50
    per_class_standard = 4948
    per_class_standard_test = 1595

    nc=3
    all_labels = [0,1,2,3,4,5,6,7,8,9]
    all_labels_GEN = all_labels


    tasks_max = 5 ##number of tasks to learn. 5 for 10 labels with 2 per task
    num_per_task = 2

elif opt.dataset_name=='emnist':
    epoch_max=25

    per_class_standard = 4800
    per_class_standard_test = 800

    tasks_max = 13 ##number of tasks to learn. 5 for 10 labels with 2 per task
    num_per_task = 2

    max_labels=tasks_max*num_per_task

    all_labels = list(range(1,max_labels+1))
    all_labels_GEN = list(range(max_labels))
    nc=1

elif opt.dataset_name=='mnist':

    epoch_max = 20
    per_class_standard = 6000
    per_class_standard_test = 1000

    all_labels = [0,1,2,3,4,5,6,7,8,9]
    all_labels_GEN = all_labels

    tasks_max = 5 ##number of tasks to learn. 5 for 10 labels with 2 per task
    num_per_task = 2
    nc=1

elif opt.dataset_name=='fashion':

    epoch_max = 20
    per_class_standard = 6000
    per_class_standard_test = 1000

    all_labels = [0,1,2,3,4,5,6,7,8,9]
    all_labels_GEN = all_labels

    tasks_max = 5 ##number of tasks to learn. 5 for 10 labels with 2 per task
    num_per_task = 2
    nc=1

# =====================================================

max_labels=tasks_max*num_per_task

if buff_real is not None:
    buff_real=buff_real
else:
    buff_real = int(per_class_standard*max_labels*opt.fraction_buff)  ## max per_class should be 60000.

dim_onehot=max_labels
num_nodes=max_labels

assert opt.batchSize*(max_labels-num_per_task)/max_labels < buff_real


# =================== label transforms ==============

def foo(target):

    swap_list=np.array(all_labels)
    map_list=np.array([0,1,2,3,4,5,6,7,8,9])
    pos_swap=np.where(swap_list==target)[0]
    remap=map_list[pos_swap]

    return int(remap)


transform_label =  transforms.Lambda(lambda x: foo(x))

if opt.dataset_name=='emnist':
    transform_label =  transforms.Lambda(lambda x: x-1)
else:
    transform_label =  transforms.Lambda(lambda x: foo(x))





# ==================Paths =============================================



outf = '%s/CloGAN_%s_buff_%.6f__Version_%d/'%(opt.main_out_path, opt.dataset_name, opt.fraction_buff, opt.version)
if not os.path.exists(outf):
    os.makedirs(outf)
try:
    os.makedirs(outf)
except OSError:
    pass

log_out='%slog.txt' % (outf)
logfile_url= '%saccuracys_train.txt' % (outf)
logfile_url_test= '%saccuracy_test.txt' % (outf)
logfile_perlabel= '%saccuracy_test_pertask.txt' % (outf)
log_out_perlabel='%slog_pertask.txt' % (outf)




# =========================================================================
# ===============================Variables=================================
# =========================================================================

# ============== ACGAN filtering =========================
filtering_params={}
if second_filtering=='soft':
    filtering_params['soft_percentile']=soft_percentile

elif second_filtering=='drs':
    filtering_params['DRS'] = gen.DRS(nz, dim_onehot, imageSize, nc, epsilon=1e-8, gamma_percentile=gamma_drs)
else:
    filtering_params={}




classifier_net = model_GAN_continual.netDC_small(nc, dim_onehot) ##num_nodes
generator_net = model_GAN_continual.netG_small(params_model['nz'], nc)
classifier_net = nn.DataParallel(classifier_net).cuda()
generator_net = nn.DataParallel(generator_net).cuda()

c_label = torch.LongTensor(opt.batchSize)
input = torch.FloatTensor(opt.batchSize, nc, imageSize, imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
d_label = torch.FloatTensor(opt.batchSize)
#     d_c_loss = nn.CrossEntropyLoss()
d_loss = nn.BCELoss()
#     d_loss = nn.BCEWithLogitsLoss()
d_c_loss = nn.NLLLoss()

input = Variable(input.cuda())
c_label = Variable(c_label.cuda())
noise = Variable(noise.cuda())
fixed_noise = Variable(fixed_noise.cuda())
d_label = Variable(d_label.cuda())
d_loss.cuda()
d_c_loss.cuda()

optimizerDC = optim.Adam(classifier_net.parameters(), lr=opt.lr, betas=betas)
optimizerG = optim.Adam(generator_net.parameters(), lr=opt.lr, betas=betas)

training = Continual_simple_cyclic(epoch_max, dim_onehot, d_loss, d_c_loss, optimizerG, optimizerDC, main_filtering, second_filtering, filtering_params, version_gen_main)

training.tensors_create(opt.batchSize, nc, imageSize, nz) ##creates tensor variables

training.outputs_log(log_out, logfile_url, logfile_url_test, outf, save_C=save_C_perepoch, save_best=save_best, save_last=save_last)



# ================ Initialize Cluster Object =========================

cluster = clu.clustering(n_cluster, cluster_type='kcenters')


# ================ initialize Embedding ===============================
if embedding is not None:
    import embedding_archs as emb

    criterion_AE = nn.MSELoss(size_average=False)


    embedding_model = emb.AE_conv3(nc, opt.latent_AE_size)

    embedding_model = nn.DataParallel(embedding_model).cuda()
    optimizer = optim.Adam(embedding_model.parameters(), lr=1e-3)

    epoch_div = main.split_integer_intervals(epoch_max, num_per_task)
    epoch_div_cum = [sum(epoch_div[:i+1]) for i in range(len(epoch_div))]
    print('train embedding epochs:', epoch_div_cum)
    prev_class = 0

else:
    embedding_model = None
# =========================================================================
# ===============================Main======================================
# =========================================================================



    # all_labels = [0,1,2,3,4,5,6,7,8,9]

# create memory object

for task in range(tasks_max):

    print('task: %d'%(task))


    if task ==0: ##learn 0,1. No memory since it is first task
        old_labels=[]
        old_labels_GEN=[]
        new_labels=all_labels[:num_per_task]
        new_labels_GEN = all_labels_GEN[:num_per_task]
        print('old_labels empty')
        print('new_labels:'+str(new_labels))

        per_class_real = 0
        per_g = 0
        batch_real =0

        batch_new = opt.batchSize

        nb_labels = len(list(set(old_labels+new_labels)))

    elif task>0: ##learn all others
        old_labels=all_labels[:(task)*num_per_task]
        new_labels=all_labels[num_per_task*(task):num_per_task+(task)*num_per_task]
        old_labels_GEN = all_labels_GEN[:(task)*num_per_task]
        new_labels_GEN=all_labels_GEN[num_per_task*(task):num_per_task+(task)*num_per_task]


        print('old_labels:'+str(old_labels))
        print('old_labels_GEN:'+str(old_labels_GEN))
        print('new_labels:'+str(new_labels))
        print('new_labels_GEN:'+str(new_labels_GEN))


        per_class_real = int(np.ceil(buff_real/len(old_labels))) ###per class in memory


        if per_class_real<per_class_standard: ##cant be bigger than standard since its an error
            pass
        else:
            per_class_real=per_class_standard

        nb_labels = len(list(set(old_labels+new_labels)))

        if task ==1:
            batch_old = int(np.ceil(0.5*opt.batchSize))
            batch_new = opt.batchSize - batch_old
        else:
            batch_old = int(np.ceil((float(len(old_labels))/nb_labels)*opt.batchSize))
            batch_new = opt.batchSize - batch_old


        batch_gan = int(np.ceil(0.5*batch_old))
        per_g = int(batch_gan/len(old_labels))


        batch_real_fix = batch_old

        batch_real = batch_old - int(per_g*len(old_labels))


        if per_class_real<n_cluster:

            n_cluster=int(per_class_real)-1



            print('change_n_cluster',n_cluster)



        if task>0:

            ## only use after it has been trained

            if task==1:

                print('get subset for memory')

#                assert embedding_model is not None

                dataset_memory = dset_cluster.dataset_clustering(opt.dataset_name, opt.dataroot, old_labels[-num_per_task:],  train=True,  per_class=per_class_real,  max_labels=max_labels,
                             n_cluster=n_cluster, clustering_method=cluster, embed_model= embedding_model,
                             normalize=False, transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Scale(imageSize),
                               transforms.ToTensor()]), target_transform=transform_label)

            elif task>1:

                print('get append for memory')

                ## update n_cluster parameter if necessary
                dataset_memory.n_cluster = n_cluster ## update n_cluster size

                dataset_memory.embed_model = embedding_model ## update AE model, as it is trained

                dataset_memory.append_memory(task+1, old_labels[-num_per_task:], per_class_real, embedding_model)

            print('Real buffer length:' + str(dataset_memory.labels.numpy().shape))


                ### can opt for generating max number of real images but only pass forward a percentage
            dataloader_memory = torch.utils.data.DataLoader(dataset_memory, batch_size=batch_real_fix,
                                                     shuffle=True, num_workers=int(opt.workers), drop_last=True)




    # =================================== new and test datasets===========================================================



    print('get new labels')

    dataset_new = dset_cluster.dataset_clustering(opt.dataset_name, opt.dataroot, new_labels,  train=True, max_labels=max_labels, per_class=per_class_standard,
                             n_cluster=n_cluster, clustering_method=None, embed_model= None,
                             normalize=False, transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Scale(imageSize),
                               transforms.ToTensor()]), target_transform=transform_label)


    dataloader_new = torch.utils.data.DataLoader(dataset_new, batch_size=batch_new,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)

    print('get test labels')

    dataset_test = dset_cluster.dataset_clustering(opt.dataset_name, opt.dataroot, list(set(old_labels+new_labels)),  train=False, max_labels=max_labels, per_class=per_class_standard_test,
                         n_cluster=n_cluster, clustering_method=None, embed_model= None,
                         normalize=False, transform=transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.Scale(imageSize),
                           transforms.ToTensor()]), target_transform=transform_label)


    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=100,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)


    if embedding is not None:
        dataloader_new_embed = dataloader_new
        dataset_test_embed = dset_cluster.dataset_clustering(opt.dataset_name, opt.dataroot, new_labels,  train=False, max_labels=max_labels, per_class=per_class_standard_test,
                                     n_cluster=n_cluster, clustering_method=None, embed_model= None,
                                     normalize=False, transform=transforms.Compose([
                                       transforms.ToPILImage(),
                                       transforms.Scale(imageSize),
                                       transforms.ToTensor()]), target_transform=transform_label)
        dataloader_test_embed = torch.utils.data.DataLoader(dataset_test_embed, batch_size=100,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)


    #=======fixed noise for printing ======================

    fixed_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
    random_label = np.random.randint(0, len(old_labels_GEN)+len(new_labels_GEN), opt.batchSize) ###generate random class label
    print('fixed label:{}'.format(random_label))
    random_onehot = np.zeros((opt.batchSize, dim_onehot))
    random_onehot[np.arange(opt.batchSize), random_label] = 1 ##distribute 1s according to label
    fixed_noise_[np.arange(opt.batchSize), :dim_onehot] = random_onehot[np.arange(opt.batchSize)]
    fixed_noise_ = (torch.from_numpy(fixed_noise_))
    fixed_noise_ = fixed_noise_.resize_(opt.batchSize, nz, 1, 1)
    fixed_noise.data.copy_(fixed_noise_)


    iter_len = len(dataloader_new)


    ## Perform DRS computation... on old_labels. Before training on new begins
    if second_filtering=='drs':
        if task>0:
            filtering_params['DRS'].compute_M_max(classifier_net, generator_net, burnin_samples, old_labels_GEN, batchsize_gen_burn)



    for epoch in range(epoch_max): ##train and evaluate

        training.init_epoch(epoch, task, old_labels, old_labels_GEN, new_labels, new_labels_GEN)

        for i in range(iter_len):

            data_new = next(iter(dataloader_new))


            if task >0:
                data_real_fix = next(iter(dataloader_memory))

            else:

                data_real_fix = data_new

            data_real = (data_real_fix[0][:batch_real, ...], data_real_fix[1][:batch_real]) ## mix buff with gan on the fly

            if task==0:
                data_real=data_new

            training.train(task, i, main_filtering, per_g, iter_len, classifier_net, generator_net, data_real, data_new, data_real_fix=None)



        if embedding is not None:

            ### train on whatever part of the task set is chosen for current task.
            train_embedding(epoch, embedding_model, dataloader_new_embed, criterion_AE, optimizer, outf)
            test_embedding(epoch, embedding_model, dataloader_test_embed, criterion_AE, outf)



        training.log_epoch(generator_net, classifier_net, epoch)

        # ===========  validate classifier ==========

        validate_continual(epoch, dataloader_test, classifier_net, input, c_label, log_out, logfile_url_test, classifier_model=dataset_param)

        validate_continual_pertask(classifier_net, list(set(old_labels+new_labels)), num_per_task, epoch, epoch_max-1, opt.dataset_name, opt.dataroot, imageSize, input, c_label, log_out_perlabel, logfile_perlabel, classifier_model=dataset_param, max_labels=max_labels, target_transform_validate=transform_label)


        #=================save images with fixed noise vector per task=================
        if opt.save_generated_images==True:
            if epoch % opt.save_image_freq ==0:

                generated = generator_net(fixed_noise)


                vutils.save_image(generated,'%s/fake_samples_labels_%d_epoch_%03d.png' % (outf,len(old_labels)+len(new_labels), epoch))


        torch.cuda.empty_cache()
