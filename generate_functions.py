
from __future__ import print_function

import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn.parallel

import torch.utils.data

from torch.autograd import Variable


def split_integer_intervals(num, div):
    '''num = Total samples 
    div = How many classes
    wish to get division of samples per class'''
    
    remainder=num%div 
    integer = int(num/div)
    splits=[]
    for i in range(div):
        splits.append(integer)
    for i in range(remainder):
        splits[i]+=1
    return splits

# ================================================ Generate Functions =====================================================

def filter_hard(gen_batchsize, pred, label_v):
    indices=torch.LongTensor(gen_batchsize).cuda()
    num_ind=0
    for k in range(gen_batchsize):
        if pred[k]==label_v[k]:
            indices[num_ind]=k
            num_ind=num_ind+1
            
    if num_ind>0:
        indices=indices[:num_ind]
    else:
        indices = torch.Tensor([])
        print('not good generator, return empty indices tensor', num_ind)
    return indices


def filter_soft(d_out, percentile=80):
    d_score = torch.exp(d_out).detach().cpu().numpy()
    th_soft = np.percentile(d_score, percentile)
    indices=np.where(d_score>=th_soft)[0]
    indices = torch.from_numpy(indices).type(torch.LongTensor).cuda() 
    return indices
    
class DRS():
    def __init__(self, nz, dim_onehot, imageSize, nc, epsilon=1e-8, gamma_percentile=0.8):
        

        self.nz = nz
        self.dim_onehot = dim_onehot
        self.imageSize = imageSize
        self.nc = nc 
        self.epsilon = epsilon
        self.gamma_percentile = gamma_percentile
        
        
        
    def compare_max(self, logits):

        batch_ratio = torch.exp(logits.detach()).cpu().numpy()
        max_idx = np.argmax(batch_ratio)
        max_ratio = batch_ratio[max_idx]

        #update max_M if larger M is found
        if max_ratio > self.max_M:
            self.max_M = max_ratio[0]
            self.max_logit = logits[max_idx].cpu().numpy()[0]
            
    def sigmoid(self, F):
        return 1/(1 + np.exp(-F))
            
    def compute_M_max(self, netDC, netG, burnin_samples, labels, batchsize_gen_burn):
        
#         netDC.eval()
        netG.eval()
        print ("Burn In to compute D_M...")
        self.batchsize_gen_burn = batchsize_gen_burn
        self.burnin_samples = burnin_samples
        self.labels = labels
        
        
        input_fake = torch.FloatTensor(self.batchsize_gen_burn, self.nc, self.imageSize, self.imageSize)
        self.input_fake = Variable(input_fake.cuda())
        noise = torch.FloatTensor(self.batchsize_gen_burn, self.nz, 1, 1)
        self.noise = Variable(noise.cuda())
        
        
        self.max_M = 0.0
        self.max_logit = 0.0
        
        splits_labels = split_integer_intervals(self.burnin_samples, len(labels))
        
        
        processed_samples = []
        for d in range(len(splits_labels)): ## per label 
            processed_samples.append(0)
            while processed_samples[-1] < splits_labels[d]:
                
                # ================ generate images and evaluate discriminator score ============
                fake_data, label_v = generate(netG, self.noise, self.batchsize_gen_burn, labels[d], self.dim_onehot, self.nz)
                self.input_fake.data.resize_(fake_data.size()).copy_(fake_data)
                with torch.no_grad():
                    logits = netDC.module.forward_D_logit(self.input_fake)
                
                ## compute max comparisom
                self.compare_max(logits.detach())
                
                processed_samples[-1] += self.batchsize_gen_burn
            
                print("Processing BurnIn...%d/%d"%(sum(processed_samples), self.burnin_samples))
                print(self.max_M, self.max_logit)
                
        print('Final values:', self.max_M, self.max_logit)

    def filter_gen(self, fake_data, netDC):
    
#         netDC.eval()
        '''Compute Filter step per class label --> aka, generate certain amount of data filtered, per label
        Fake_data is a tensor output of G
        '''
        ## ============ Burn In: Evaluate M max ====================
#         print ("Start Sampling...")
        counter = 0
        rejected_counter = 0
        indices = []
        print(fake_data.shape[0])
#         while counter < per_class:
            
        # ============= Generate batch of data =================
#             fake_data, label_v = generate(netG, self.noise, self.batchsize_gen_drs, labels[d], self.dim_onehot, self.nz)
        self.input_fake.data.resize_(fake_data.size()).copy_(fake_data)
        with torch.no_grad():
            logits = netDC.module.forward_D_logit(self.input_fake)

        # ===== update M if need be =====
        self.compare_max(logits.detach())

        logits = logits.detach().cpu().numpy()

        #calculate F_hat and pass it into sigmoid
        # set gamma dynamically (80th percentile of F)
        Fs = logits - self.max_logit - np.log(1 - np.exp(logits - self.max_logit - self.epsilon))
        gamma = np.percentile(Fs, self.gamma_percentile)
#         print('gamma: ', gamma)
        
        F_hat = Fs - gamma
        acceptance_prob = self.sigmoid(F_hat).flatten()

#         print('acceptance prob: ', acceptance_prob)
        ## Filter
        probability = np.random.uniform(0, 1, acceptance_prob.shape[0])
        
        
        indices = np.where(probability<=acceptance_prob)[0]
        
        counter = len(indices)
        counter_rejected = len(probability) - len(indices)

                    
        indices = torch.from_numpy(np.array(indices)).type(torch.LongTensor).cuda()
        
        return indices
                    

        
        
    
def generate(netG, noise, batchsize, label_in, dim_onehot, nz):
    noise.data.resize_(batchsize, nz, 1, 1)
    noise.data.normal_(0, 1)
    label = np.random.randint(label_in, label_in+1, batchsize)    
    noise_ = np.random.normal(0, 1, (batchsize, nz))
    label_onehot = np.zeros((batchsize, dim_onehot))
    label_onehot[np.arange(batchsize), label] = 1
    noise_[np.arange(batchsize), :dim_onehot] = label_onehot[np.arange(batchsize)]
    noise_ = (torch.from_numpy(noise_))
    noise_ = noise_.resize_(batchsize, nz, 1, 1)
    noise.data.copy_(noise_) # feed noise and labels in the graph
    label_v=(torch.from_numpy(label))
    label_v = label_v.cuda()
    fake=netG(noise) ##img
    fake_data=fake.data
    return fake_data, label_v


# =============== Elsewhere in code Initialize DRS==========================
## 1. Then at each Task, run Burn_In for old labels 
## 2. At each iteration then filter samples 
def generate_olddata_copy(netG_old, label_in, batch_gan, nz, dim_onehot):
        
    '''Generate batch of data from ACGAN, typically old copy as in MeRGAN'''
    
    noise = torch.FloatTensor(batch_gan, nz, 1, 1)
    noise = Variable(noise.cuda())
    noise.data.normal_(0, 1)
    
    label = np.random.randint(label_in[0], label_in[-1]+1, batch_gan)    
    noise_ = np.random.normal(0, 1, (batch_gan, nz))
    
    label_onehot = np.zeros((batch_gan, dim_onehot))
    label_onehot[np.arange(batch_gan), label] = 1
    
    noise_[np.arange(batch_gan), :dim_onehot] = label_onehot[np.arange(batch_gan)]
    noise_ = (torch.from_numpy(noise_))
    noise_ = noise_.resize_(batch_gan, nz, 1, 1)
    
    noise.data.copy_(noise_) # feed noise and labels in the graph
    
    label_v=(torch.from_numpy(label))
    lbl_old = label_v.cuda()
    fake=netG_old(noise) ##img
    im_old=fake.data
    
    return im_old, lbl_old


def Generate_Forward_cond_singlelabel(netG, netDC, label_in, per_class, batchsize, nz, 
                                      imageSize, dim_onehot, nc, logfile, filtering_params, 
                                      main_filtering=True, second_filtering='drs',
                                      calc_acc=False, breakpoint=50): 
    
    '''generate images per label
    Corrected for eval mode
    Actually, need to modify this because HARD filtering may always be necessary. 
    Option to couple filetring styles!!!
    
    Filtering types:
    1. hard
    2. soft
    3. drs: Needs to receive DRS class as parameter
    
    batchsize is the amount you want to generate at each iteration
    '''
    
    print('==================== Start Generation =========================')
#     print('label_in: ', label_in)
#     print('per_class: ', per_class)
#     print('batchsize_gen: ', batchsize)
#     print('nz: ', nz)
#     print('imageSize: ', imageSize)
#     print('dim_onehot: ', dim_onehot)
#     print('nc: ', nc)
#     print('filtering params: ', filtering_params)
#     print('main_filtering: ', main_filtering)
#     print('second_filtering: ', second_filtering)
    
    
    netG.eval()
#     netDC.eval()
    input_fake = torch.FloatTensor(batchsize, nc, imageSize, imageSize) ##input for classifier 
    input_fake=Variable(input_fake.cuda())
    noise = torch.FloatTensor(batchsize, nz, 1, 1)
    noise = Variable(noise.cuda())
    
    
    
    Out_G_saved=torch.FloatTensor(per_class, nc, imageSize,imageSize).cuda()
    Out_G_labels=torch.zeros(per_class,  dtype=torch.int32).cuda()
    
    if calc_acc:
        c_fake_labels=torch.FloatTensor(per_class).cuda()
        accuracy_class_fake=torch.FloatTensor(per_class).cuda()
        
    i=0
    count=[0] ## while I don't have the number desired
    while count[-1] < per_class: 
        
        print('iteration: ', i)
        print('number of good images:', count[-1])
        
        if i > breakpoint: ##breakpoint should be set so as batchsize*i >...> per_class 
            ## fill the rest of indices with fake_data remaining without filtering
            print('Generator Bad: Exceeded breakpoint')
            missing = per_class - count[-1]
            fake_data, label_v = generate(netG, noise, missing, label_in, dim_onehot, nz)
            indices = torch.from_numpy(np.arange(missing)).type(torch.LongTensor).cuda() 
            
            log = open(logfile, 'a+')
            log.write("Bad generator, exceeded breakpoint at iter: "+str(i)+"\n")
            log.close()
        
        else:
            
            # ============== generate batchsize of images ==================
                
#             print(label_in, batchsize, nz, dim_onehot)
            
# #             30 0 10 110
# #             0 30 110 10
            
            fake_data, label_v = generate(netG, noise, batchsize, label_in, dim_onehot, nz)
        
#             label_v[0:5]=2
            
#             print(label_v)
            
            # ===================== Main Filter =============================
            if main_filtering==True: ## keep only generated with correct label prediction
            #========================= Filter ================================
                
                # =========== Run through Dicriminator/classifier =============
                input_fake.data.resize_(fake_data.size()).copy_(fake_data)
                d_out, class_output = netDC(input_fake.detach())
                # ============ Filter =======================================
                pred = class_output.data.max(1)[1]
                indices = filter_hard(batchsize, pred, label_v) ## may return an empty tensor of indices 
                
                print('Doing hard filtering. Number of correctly classified: ', len(indices))
                
#                 print(indices)
                
#                 print(label_v)
#                 print(pred)
                
            # ===================== Secondary Filter =============================
            if second_filtering=='soft':
                
                if main_filtering==True:
                    
                    if indices.shape[0]>0:
                        fake_data = fake_data[indices, ...] ## only pass through what is classified correctly 
                        label_v = label_v[indices]
                        print('Doing Hard + Soft', fake_data.shape)
                   
                if fake_data.shape[0]>0:
                    ## Soft filtering is just if Discriminator output is above certain probability
                    # ============== generate batchsize of images ==================
                    input_fake.data.resize_(fake_data.size()).copy_(fake_data)
                    # =========== Run through Dicriminator/classifier =============
                    d_out, class_output = netDC(input_fake.detach())
                    # ============ Filter =======================================
                
                    indices = filter_soft(d_out, percentile=filtering_params['soft_percentile'])
                
                ## will return a second bash of indices, for an already potentially reduced data
                
                print('Doing Soft. Number of images to pass soft: ', len(indices))
                                
            elif second_filtering=='drs':
                
                if main_filtering==True:
                    if indices.shape[0]>0:
                        fake_data = fake_data[indices, ...] ## only pass through what is classified correctly
                        label_v = label_v[indices]
                        print('Doing Hard + DRS',fake_data.shape)

                ## do Discriminator rejection sampling
                if fake_data.shape[0]>0:
                    indices = filtering_params['DRS'].filter_gen(fake_data, netDC)

                    print('Doing DRS. Number of images to pass drs: ', len(indices))
                ## will return a second bash of indices, for an already potentially reduced data
                

            if (main_filtering==False) and (second_filtering=='none'):
                ##keep all
                print('Not doing any filtering, let everything pass')
                assert batchsize == fake_data.shape[0]
                indices = torch.from_numpy(np.arange(batchsize)).type(torch.LongTensor).cuda() 
                      
                    
                    
        count.append(count[i]+len(indices))
        
        # ================== calculate accuracy =========================
        if calc_acc:
            d_out, class_output = netDC(input_fake.detach())
            pred = class_output.data.max(1)[1]
            x=np.zeros((batchsize))
            for j in range(batchsize):
                if pred[j]==label_v[j]:
                    x[j]=1

            accuracy=sum(x)
            accuracy_class_fake[i]=accuracy ##how the classifier is approximating the labels
                    

        # ============================== put generated data into Tensors ===================  
        if len(indices)>0:
            print('store images')
            if  (count[i] < per_class) and (count[i+1] < per_class):
                Out_G_saved[count[i]:count[i+1],:]=torch.index_select(fake_data, 0, indices)
                Out_G_labels[count[i]:count[i+1]]=torch.index_select(label_v, 0, indices)

            elif  (count[i] < per_class) and (count[i+1] >= per_class):
                Out_G_saved[count[i]:per_class,:]=torch.index_select(fake_data, 0, indices)[:len(Out_G_labels[count[i]:per_class])]
                Out_G_labels[count[i]:per_class]=torch.index_select(label_v, 0, indices)[:len(Out_G_labels[count[i]:per_class])]

            elif  (count[i] >= per_class):
                Out_G_saved=Out_G_saved[:per_class,:]
                Out_G_labels=Out_G_labels[:per_class]
                
#             print('number of filtered so far: ', torch.nonzero(Out_G_labels).size(0), Out_G_labels.type(torch.LongTensor).cpu().numpy())

        i=i+1
        
    Out_G_saved=Out_G_saved[0:per_class,:]
    Out_G_labels=Out_G_labels[0:per_class].type(torch.LongTensor)

    return Out_G_saved, Out_G_labels, i     

        
        
def Generate_Forward_cond_singlelabel_version1(netG, netDC, label_in, per_class, batchsize, nz, imageSize, dim_onehot, nc, conditional=True, calc_acc=False, breakpoint=1000): 
    
    '''generate images per label
    Corrected for eval mode'''
    
    netG.eval()
    input_fake = torch.FloatTensor(batchsize, nz, 1, 1) ##input for classifier 
    input_fake=Variable(input_fake.cuda())
    noise = torch.FloatTensor(batchsize, nz, 1, 1)
    noise = Variable(noise.cuda())
    
    Out_G_saved=torch.FloatTensor(per_class, nc, imageSize,imageSize).cuda()
    Out_G_labels=torch.FloatTensor(per_class).cuda()
    
    if calc_acc:
        c_fake_labels=torch.FloatTensor(per_class).cuda()
        accuracy_class_fake=torch.FloatTensor(per_class).cuda()
        
    i=0
    count=[0]
    # while i <= int(round(per_class/batchsize)):
    while count[-1] <= per_class: 
        
        # ============== generate batchsize of images ==================
        noise.data.resize_(batchsize, nz, 1, 1)
        noise.data.normal_(0, 1)
        label = np.random.randint(label_in, label_in+1, batchsize)    
        noise_ = np.random.normal(0, 1, (batchsize, nz))
        label_onehot = np.zeros((batchsize, dim_onehot))
        label_onehot[np.arange(batchsize), label] = 1
        noise_[np.arange(batchsize), :dim_onehot] = label_onehot[np.arange(batchsize)]
        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batchsize, nz, 1, 1)
        noise.data.copy_(noise_) # feed noise and labels in the graph
        label_v=(torch.from_numpy(label))
        label_v = label_v.cuda()
        fake=netG(noise) ##img
        fake_data=fake.data

        
        # ============== Classify those images ==================
        input_fake.data.resize_(fake_data.size()).copy_(fake_data)
        d_out, class_output = netDC(input_fake.detach())
        pred = class_output.data.max(1)[1]
               
            
        # ================== calculate accuracy =========================
        if calc_acc:
            x=np.zeros((batchsize))
            for j in range(batchsize):
                if pred[j]==label_v[j]:
                    x[j]=1

            accuracy=sum(x)
            accuracy_class_fake[i]=accuracy ##how the classifier is approximating the labels
            
            
        #========================= Filter ================================
        if conditional: ## keep only generated with correct label prediction   
            
            indices=torch.LongTensor(batchsize).cuda()
            num_ind=0
            for k in range(batchsize):
                if pred[k]==label_v[k]:
                    indices[num_ind]=k
        #             indices.append(k)
                    num_ind=num_ind+1
            if num_ind>0:
                indices=indices[:num_ind]
            else:
                indices = torch.from_numpy(np.arange(batchsize)).type(torch.LongTensor).cuda()
                print('not good generator')
        else: ##keep all
            indices = torch.from_numpy(np.arange(batchsize)).type(torch.LongTensor).cuda() 
            
        count.append(count[i]+len(indices))

        
        # ============================== put generated data into Tensors ===================  
        
        if  (count[i] < per_class) and (count[i+1] < per_class):
            Out_G_saved[count[i]:count[i+1],:]=torch.index_select(fake_data, 0, indices)
            Out_G_labels[count[i]:count[i+1]]=torch.index_select(label_v, 0, indices)
            # c_fake_labels[count[i]:count[i+1]]=torch.index_select(pred.float().cpu(), 0, indices) 
            
    #         torch.from_numpy(np.std(class_output.data.cpu().numpy()[indices],1))
        elif  (count[i] < per_class) and (count[i+1] >= per_class):
            Out_G_saved[count[i]:per_class,:]=torch.index_select(fake_data, 0, indices)[:len(Out_G_labels[count[i]:per_class])]
            Out_G_labels[count[i]:per_class]=torch.index_select(label_v, 0, indices)[:len(Out_G_labels[count[i]:per_class])]
            # c_fake_labels[count[i]:per_class]=torch.index_select(pred.float().cpu(), 0, indices)[:len(Out_G_labels[count[i]:per_class])] ##from netC --> predictions 
        
        elif  (count[i] >= per_class):
            Out_G_saved=Out_G_saved[:per_class,:]
            Out_G_labels=Out_G_labels[:per_class]
            # c_fake_labels=c_fake_labels[:per_class]
                    
        i=i+1
        
        if i > breakpoint: ##breakpoint should be set so as batchsize*i >...> per_class 
            break

#     print(count[-1])
#     print(per_class)
        
    Out_G_saved=Out_G_saved[0:per_class,:]
    Out_G_labels=Out_G_labels[0:per_class]
    # c_fake_labels=c_fake_labels[0:per_class]
    
    # std_classes=std_classes[0:i]
    # accuracy_class_fake=accuracy_class_fake[0:i]

    ####statistical normalization

    return Out_G_saved, Out_G_labels, i     


    
        
        
        
        
        
        
        
        
        