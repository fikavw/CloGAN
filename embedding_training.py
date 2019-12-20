#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:38:42 2019

@author: arios
"""


from __future__ import print_function
import torch

from torch.autograd import Variable


def train_embedding(epoch, model, train_loader, criterion, optimizer, out_dir):
    #Sets the module in training mode.
    
    '''Train AE. First type of embedding tried'''
    model.train()
    
        
    train_loss = 0 ## loss per epoch 

#     batch_idx, (data, label) =enumerate(train_loader).__next__()
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.cuda() #[64, 1, 28, 28]
        label = label.cuda()
        
        optimizer.zero_grad()
        
        data = Variable(data)
        label = Variable(label)
        
        # 1) AE training 
        recon_batch = model(data) ## encode 
        loss = criterion(recon_batch, data)
        
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 10 == 0:
            output = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data))
            print(output)
            logfile = open(out_dir+'/log.txt', 'a+')
            logfile.write(output+"\n")
            logfile.close()
            
#         if batch_idx == 0:
#             vutils.save_image(recon_batch,
#                               '%s/AE_training_reconstruction_%03d.png' 
#                               % (out_dir, epoch))


    output_epoch = '====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset))
    print(output_epoch)
    
    logfile = open(out_dir+'/log.txt', 'a+')
    logfile.write(output_epoch+"\n")
    logfile.close()
        


def test_embedding(epoch, model, test_loader, criterion, out_dir):
    #Sets the module in evaluation mode
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.cuda()
            label = label.cuda()
            
            recon_batch = model(data) ## encode 
            loss = criterion(recon_batch, data)  
            
            test_loss += loss.item()

#            if i == 0:
#                n = min(data.size(0), 8)
##                 print('---',recon_batch.shape)
#                comparison = torch.cat([data[:n],
#                                      recon_batch[:n]])
#                 save_image(comparison.cpu(),
#                          out_dir+'/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    
    output_epoch ='====> Test set loss: {:.4f}'.format(test_loss)
    print(output_epoch)
    


