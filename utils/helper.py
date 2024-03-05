import torch

#https://wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE
#https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/

def checkpoint(model, filename, epoch, loss, optimizer, scheduler):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}, filename)
    
def checkpoint2(model, filename, epoch, loss, optimizer):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'optimizer_state_dict': optimizer.state_dict()}, filename)
    
def resume(model, path, filename, optimizer):
    model.load_state_dict(torch.load(path + filename))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, epoch, loss
    
def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    return model, optimizer, scheduler, start_epoch
