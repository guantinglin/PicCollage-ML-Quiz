import torch
from torch import nn
import torch.optim
import torch.utils.data
from models import CNN_with_regressor
from datasets import *

data_folder = './data'
out_foler = './trained'
batch_size = 32
epochs = 50
initial_lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
workers = 1 

def main():
    
    global BEST_name, lowest_loss, best_epoch
    lowest_loss = float('inf')
    
    print ('Initializing the network settings')
    net = CNN_with_regressor()    
    optimizer = torch.optim.Adam(params=net.parameters(), lr=initial_lr)    
    net.to(device)
    criterion = nn.SmoothL1Loss().to(device)
    criterion_L1 = nn.L1Loss()
    criterion_L2 = nn.MSELoss()
    criterion_L1 = criterion_L1.to(device)
    criterion_L2 = criterion_L2.to(device)




    
    print ('Loading datasets')
    train_loader = torch.utils.data.DataLoader(RegressionDataset(data_folder,'TRAIN',transform=None),                                    
                                               batch_size=batch_size, shuffle=False, 
                                               num_workers=workers, pin_memory=True)
                                    
    val_loader = torch.utils.data.DataLoader(RegressionDataset(data_folder,'VAL',transform=None),
                                             batch_size=batch_size, shuffle=False, 
                                             num_workers=workers, pin_memory=True)
                                    
    test_loader = torch.utils.data.DataLoader(RegressionDataset(data_folder,'TEST',transform=None),
                                              batch_size=batch_size, shuffle=False, 
                                              num_workers=workers, pin_memory=True)         
    
    print ('Testing with random initialization')                                      
    loss_L1 = validate(test_loader,net,criterion_L1)
    loss_L2 = validate(test_loader,net,criterion_L2)        
    print ('Testing L1 loss: %.5f ' % (loss_L1))
    print ('Testing L2 loss: %.5f ' % (loss_L2))                                              
                                              
    print ('Start training process...')        
    num_params = caculate_param(net)/1000000.
    print ('Total number of params: %2.4f M' % (num_params)) 
    for epoch in range(1,epochs+1):
        train(train_loader=train_loader,
              network = net,  
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        
        loss = validate(val_loader,net,criterion)
        
        if loss < lowest_loss:
            is_best = True
            lowest_loss = loss
        else:
            is_best = False
        save_checkpoint(epoch,net,is_best)
    # test with the lowest loss model of validation set
    best_checkpoint = torch.load(BEST_name)
    best_network = best_checkpoint['network']
    
    print ('Testing with best checkpoint: epoch %d' % (best_epoch))
    loss_L1 = validate(test_loader,best_network,criterion_L1)
    loss_L2 = validate(test_loader,best_network,criterion_L2)        
    print ('Testing L1 loss: %.5f' % (loss_L1))
    print ('Testing L2 loss: %.5f' % (loss_L2))
    
    
def train(train_loader,network,criterion,optimizer,epoch,print_period = 1000):
    """
    Train model 

    :param train_loader: training data loader
    :param network: network 
    :param criterion: training criterion
    :param optimizer: optimizer
    :param epoch: epoch now
    :param print_period: period for sreen show
    
    """
    
    network.train()
    
    for i, (corrs,imgs) in enumerate(train_loader,start = 1):
        imgs = imgs.to(device)
        
        targets = corrs.to(device)
        
        
        pred_corrs = network(imgs)
        
        loss = criterion(pred_corrs,targets)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % print_period == 0:
            print ('Epoch: %2d/%2d\t'
                   'Iter: %4d/%4d \t' 
                   'Loss: %.5f' % (epoch,epochs,i,len(train_loader),loss))
        
def validate(val_loader,network,criterion):
    """
    Evaluate model checkpoint.

    :param val_loader: the epoch now
    :param network: network 
    :param criterion: evaluation criterion
    return the average loss on validation set
    """
    
    network.eval()
    loss_tmp = 0
    with torch.no_grad():
        for i, (corrs,imgs) in enumerate(val_loader):
            imgs = imgs.to(device)
            targets = corrs.to(device)
            
            pred_corrs = network(imgs)
            
            loss = criterion(pred_corrs,targets)     
            loss_tmp = loss_tmp + loss.item()
    return loss_tmp/len(val_loader)
    
def save_checkpoint(epoch, network,is_best):
    """
    Saves model checkpoint.

    :param epoch: the epoch now
    :param network: trained_network to be saved
    :param is_best: is this checkpoint the best so far?
    """
    global BEST_name,best_epoch
    state = {'epoch': epoch,
             'network' :network}             
             
             
    filename = out_foler+'/checkpoint_epoch_' + str(epoch) + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        best_epoch = epoch
        BEST_name = out_foler+ '/BEST_checkpoint.pth.tar'
        torch.save(state, BEST_name)    
 
def caculate_param(network):
    """
    Return the total numbers of trainable parameters
    
    :param network: network that is aimed to be caculated
    """
    return sum(p.numel() for p in network.parameters() if p.requires_grad)
 
 
if __name__ == '__main__':
    main()