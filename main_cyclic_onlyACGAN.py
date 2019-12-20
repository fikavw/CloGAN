import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

import datasets_continual_embedding as dset_cluster

from k_means_clustering import *
import generate_functions as gen

#======================================================
# ======= main script functions =======================
#======================================================


manualSeed=999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
cudnn.benchmark = True



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

def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)


class Continual_simple_cyclic(object):
    
    def __init__(self, epoch_max, dim_onehot, d_criterion, d_c_criterion, optimizerG, optimizerD, main_filtering, second_filtering, filtering_params={}, version_gen=2):
        
        self.dim_onehot = dim_onehot
        self.d_criterion = d_criterion
        self.d_c_criterion = d_c_criterion
        
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        
        self.epoch_max = epoch_max
        
        self.filtering_params = filtering_params
        self.main_filtering = main_filtering
        self.second_filtering = second_filtering
        
        self.version_gen = version_gen
        
        
    def outputs_log(self, log_out, logfile_url, logfile_url_test, outf, save_C=True, save_best=True, save_last=False):
        
        self.log_out = log_out
        self.logfile_url = logfile_url
        self.logfile_url_test = logfile_url_test
        self.outf = outf
        self.save_C = save_C
        self.save_best=save_best
        self.save_last=save_last
        
    def tensors_create(self, batchSize, nc, imageSize, nz):
                
        self.nz = nz 
        
        self.nc = nc
        
        self.imageSize = imageSize 
        
        input = torch.FloatTensor(batchSize, nc, imageSize, imageSize)
        
        d_label = torch.FloatTensor(batchSize)
        c_label = torch.LongTensor(batchSize)    

        noise = torch.FloatTensor(batchSize, self.nz, 1, 1)

        self.input = Variable(input.cuda())
        self.d_label = Variable(d_label.cuda())
        self.c_label = Variable(c_label.cuda())
              
        self.noise = Variable(noise.cuda())

    def init_epoch(self, epoch, task, old_labels, old_labels_GAN, new_labels, new_labels_GAN):
    
        self.task = task 
        
        self.old_labels = old_labels 
        self.new_labels = new_labels 
        self.old_labels_GAN = old_labels_GAN
        self.new_labels_GAN = new_labels_GAN
        
        self.acc_classifier_dc=[]
        
        self.real_label = 1
        self.fake_label = 0
        
        self.epoch = epoch 
        
   
        
    def train(self, task, i, cond_generate, per_g, iter_len, netD, netG, data_real, data_new, data_real_fix=None):

        #implements weighted learning, multitask learning. 
        # ======== adjust batch size so that new data is seen fully ======= #
        
        

        if task>0:
            
            # print('already has old labels')
            img_r, lbl_r = data_real
            img_new, lbl_new = data_new
            img_r = img_r.cuda()
            img_new = img_new.cuda()
            lbl_r = lbl_r.cuda()
            lbl_new = lbl_new.cuda()
            

#            print('GAN and real for old labels at iteration: %d' %(i))

            for c, l in enumerate(self.old_labels_GAN):

#                print('================ generating images Main Train =================')

# #                    im, lb ,_ = Generate_Forward_cond_singlelabel(netG, netD, l, per_g, per_g, self.nz, self.imageSize, dim_onehot=self.dim_onehot, nc=self.nc, conditional=cond_generate, calc_acc=False, breakpoint=1000)
                
                
#                print('generate for label:', l, per_g)
        
        
                if self.version_gen==1:
#                    print('using generator version 1')
                    im, lb ,_  = gen.Generate_Forward_cond_singlelabel_version1(netG, netD, l, per_g, per_g, self.nz, self.imageSize, self.dim_onehot, self.nc, conditional=self.main_filtering, calc_acc=False, breakpoint=500)
            
                elif self.version_gen==2:
#                    print('using generator version 2')
                    im, lb ,_ = gen.Generate_Forward_cond_singlelabel(netG, netD, l, 
                                              per_g, per_g, self.nz, self.imageSize, 
                                              self.dim_onehot, self.nc, self.log_out,
                                              filtering_params=self.filtering_params, 
                                              main_filtering=self.main_filtering, 
                                              second_filtering=self.second_filtering, 
                                              calc_acc=False, breakpoint=50)


                
                lb = lb.type(torch.LongTensor).cuda()
                im = im.cuda()


                if c==0:

                    img_g = im
                    lbl_g = lb

                else:

                    img_g = torch.cat((img_g, im), 0)
                    lbl_g = torch.cat((lbl_g, lb), 0)

                img_=torch.cat((img_g, img_r, img_new), 0)
                label_=torch.cat((lbl_g, lbl_r, lbl_new), 0)


            indices_shu = torch.randperm(img_.size(0)).cuda()
            img=img_[indices_shu,:]
            label=label_[indices_shu]
            
        else:
            
#            print('task zero only has new labels at iteration: %d' %(i))
            
            img, label = data_new ##new images


        self.batch_images = img
        self.batch_labels = label
        
        
        netD.train()
        netG.train()
        netD.zero_grad()
        


        # ===============================
        
        ###########################
        # (1) Update D network
        ###########################
        # train with real
    
        batch_size = img.size(0)
        self.input.data.resize_(img.size()).copy_(img)
        self.c_label.data.resize_(batch_size).copy_(label)


        self.d_label.data.resize_(batch_size).fill_(self.real_label)
        
        d_output_real, d_c_output_real = netD(self.input)
        
    
        d_errD_real = self.d_criterion(d_output_real, self.d_label) ## binary cross entropy
        c_errD_real = self.d_c_criterion(d_c_output_real, self.c_label) ## Negative log likelihood
        
        
        errD_real = d_errD_real + c_errD_real
        
        errD_real.backward()
                
        D_x = d_output_real.data.mean()
        
        
        ##########Test accuracies#######
        
        self.correct_dc, self.length_dc = test(d_c_output_real, self.c_label)


        #========== set up one-hot representation of labels into noise vector ==================

        # train with fake
        self.noise.data.resize_(batch_size, self.nz, 1, 1)
        self.noise.data.normal_(0, 1)

        if task>0:
            label = np.random.randint(self.old_labels_GAN[0], self.new_labels_GAN[-1]+1, batch_size)
        else:
            label = np.random.randint(self.new_labels_GAN[0], self.new_labels_GAN[-1]+1, batch_size)
            
        noise_ = np.random.normal(0, 1, (batch_size, self.nz))
        label_onehot = np.zeros((batch_size, self.dim_onehot))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :self.dim_onehot] = label_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, self.nz, 1, 1)
        self.noise.data.copy_(noise_)
        

        # put new random labels in c_label
        self.c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
        self.d_label.data.fill_(self.fake_label)

        fake = netG(self.noise)
        
        d_output_fake, d_c_output_fake = netD(fake.detach())
        
        self.d_c_output_fake = d_c_output_fake
        self.label_fake = torch.from_numpy(label).type(torch.LongTensor).cuda()
        
        self.correct_fake, self.length_fake = test(self.d_c_output_fake, self.label_fake)
#        print('Fake Accuracy: ', float(self.correct_fake)/float(self.length_fake))
        
        
        d_errD_fake = self.d_criterion(d_output_fake, self.d_label)
        c_errD_fake = self.d_c_criterion(d_c_output_fake, self.c_label)
        
        errD_fake = d_errD_fake + c_errD_fake

        errD_fake.backward()
        
        loss_D = errD_fake + errD_real
        
        # d_errD = 0.5 * (torch.mean((d_output_real - 1)**2) + torch.mean(d_output_fake**2)) ##combine pass on real and fake data. 
        
        D_G_z1 = d_output_fake.data.mean()
        
        self.optimizerD.step()

        ###########################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        self.d_label.data.fill_(self.real_label)  # fake labels are real for generator cost
        
        d_output, c_output = netD(fake)
        d_errG = self.d_criterion(d_output, self.d_label)
        c_errG = self.d_c_criterion(c_output, self.c_label)
        
        loss_G = d_errG + c_errG
        
        loss_G.backward()
        
        D_G_z2 = d_output.data.mean()
        
        self.optimizerG.step()
          

        self.acc_classifier_dc.append(float(self.correct_dc)/float(self.length_dc))
#         print(float(self.correct_dc)/float(self.length_dc))
            
        if i % 10 == 0:
            logfile = open(self.log_out, 'a+')
            log_output='[%d/%d/%d][%d/%d] Loss_D: %.4f  Loss_G: %.4f Accuracy_DC: %.4f / %.4f = %.4f' % (self.epoch, self.epoch_max, task, i, iter_len , loss_D.data[0], loss_G.data[0], self.correct_dc, self.length_dc, (float(self.correct_dc)/float(self.length_dc)))
            logfile.write(log_output+"\n")
            print(log_output)
            logfile.close()
            

            

    def log_epoch(self, netG, netDC, epoch):
            
        self.epoch=epoch
        
        acc_tot_dc=sum(self.acc_classifier_dc)/len(self.acc_classifier_dc)
#         print(acc_tot_dc)
        
        logfile = open(self.logfile_url, 'a+')
        logfile.write(str(self.epoch) + '\t' + str(acc_tot_dc) + '\n')
        logfile.close()
        

        if (self.epoch==0):
            self.past_best=acc_tot_dc
            self.best=True
        if (self.past_best < acc_tot_dc):
            self.past_best=acc_tot_dc
            self.best=True
        else:
            self.best=False

        if self.save_C:
            if self.save_last:
                if self.epoch == self.epoch_max-1:
                    torch.save(netG.state_dict(), '%s/netG_task_%d_epoch_%d.pth' % (self.outf, self.task, epoch))
                    torch.save(netDC.state_dict(), '%s/netDC_task_%d_epoch_%d.pth' % (self.outf, self.task, epoch))
            elif self.save_best:
                if self.best==True:
                    torch.save(netG.state_dict(), '%s/netG_task_%d_savebest.pth' % (self.outf, self.task))
                    torch.save(netDC.state_dict(), '%s/netDC_task_%d_savebest.pth' % (self.outf, self.task))



                               
                               
# ========================================== Validate ========================================================
def validate_continual(epoch, dataloader_test, classifier, input, c_label, log_out, logfile_url_test, classifier_model = 'ACGAN'):
    
    #====================================
    # use input and c_label as one of the tensors created previously
    #======turn on evaluation mode=======

    classifier.eval()

    acc_classifier_test=[]

    for i, data in enumerate(dataloader_test):
       ##test set
        img_t, lbl_t=data
        input.data.resize_(img_t.size()).copy_(img_t)
        c_label.data.resize_(img_t.size(0)).copy_(lbl_t)

        if classifier_model=='ACGAN':
            dc_d_output_t, dc_c_output_t = classifier(input)#forward pass on real data Discriminator
        elif classifier_model=='classifier':
            dc_c_output_t = classifier(input)#forward pass on real data Discriminator


        correct_t, length_t = test(dc_c_output_t, c_label)
        acc_classifier_test.append(float(correct_t)/float(length_t))

        if i % 20 == 0:
            logfile = open(log_out, 'a+')
            log_output='[%d/%d] Test_Accuracy for DC:  %.4f' % ( i, len(dataloader_test), (float(correct_t)/float(length_t)))
            logfile.write(log_output+"\n")
            print(log_output)
            logfile.close()

    acc_tot_test = sum(acc_classifier_test)/len(acc_classifier_test)
    logfile_test = open(logfile_url_test, 'a+')
    logfile_test.write(str(epoch) +'\t'+  str(acc_tot_test)+ '\n')
    logfile_test.close()

    
    
def validate_continual_pertask(classifier, labels, num_per_task, epoch, epoch_max, dataset_name, dataroot_dataset, imageSize, input, c_label, log_out, logfile_perlabel, classifier_model='ACGAN', max_labels=10, target_transform_validate=None):
    
    # labels_task = old_labels + new_labels
    
    #====================================
    # use input and c_label as one of the tensors created previously
    #======turn on evaluation mode=======
    classifier.eval()
    
    if dataset_name=='mnist':
        nc=1 ##MNISt has 1 channel --> Black and white (gray)
        per_class_test=1000
    elif dataset_name =='fashion':
        nc=1
        per_class_test=1000
    elif dataset_name =='emnist':
        nc=1
        per_class_test=800
    elif dataset_name =='svhn':
        nc=3
        per_class_test=1595
    elif dataset_name =='cifar10':
        nc=3
        per_class_test=1000



    current_task = int(float(len(labels))/num_per_task)
    
    for task in range(current_task):
        
        #select subset of labels per task being evaluated 
        labels_task=labels[num_per_task*(task):num_per_task+(task)*num_per_task]


    
    
        dataset_test = dset_cluster.dataset_clustering(dataset_name, dataroot_dataset, labels_task,  train=False, max_labels=max_labels, per_class=per_class_test,
                         n_cluster=1, clustering_method=None, embed_model= None, 
                         normalize=False, transform=transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.Scale(imageSize),
                           transforms.ToTensor()]), target_transform=target_transform_validate)
    
    
    
        
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=100,
                                         shuffle=True, num_workers=2, drop_last=True)

        ##now run fwd pass only for that task
        
        acc_test_pertask=[]

        for i, data in enumerate(dataloader_test):
           ##test set
            img_t, lbl_t=data
            input.data.resize_(img_t.size()).copy_(img_t)
            c_label.data.resize_(img_t.size(0)).copy_(lbl_t)

            if classifier_model=='ACGAN':
                dc_d_output_t, dc_c_output_t = classifier(input)#forward pass on real data Discriminator
            elif classifier_model=='classifier':
                dc_c_output_t = classifier(input)#forward pass on real data Discriminator

            
            correct_t, length_t = test(dc_c_output_t, c_label)
            acc_test_pertask.append(float(correct_t)/float(length_t))

            
            if i % 20 == 0:
                logfile = open(log_out, 'a+')
                log_output='[%d/%d] Task/Task_Max_Current: [%d/%d].Test_Accuracy for DC:  %.4f. ' % ( i, len(dataloader_test), task, int(float(len(labels))/num_per_task), (float(correct_t)/float(length_t)))
                logfile.write(log_output+"\n")
                print(log_output)
                logfile.close()
                

        
        acc_tot_test = sum(acc_test_pertask)/len(acc_test_pertask)
        logfile_test = open(logfile_perlabel, 'a+')
        
        logfile_test.write(str(epoch) +'\t' + str(epoch_max) +'\t' +  str(task) + '\t' + str(current_task - 1) + '\t' + str(acc_tot_test) + '\n')
        
        logfile_test.close()


    