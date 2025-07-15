import torch
import os
from collections import OrderedDict

def save_ramdom_state(chk_dir, ramdom_state, np_stats, torch_state, torch_cuda_state):
    torch.save({'random': ramdom_state,
                'np': np_stats,
                'torch': torch_state,
                'torch_cuda': torch_cuda_state
               }, os.path.join(chk_dir, 'random_state.pkl'))

def save_checkpoint(chk_dir, epoch, model, classifier, optimizer, scheduler=None, scaler=None, lr=None):
    torch.save({'model': model.state_dict(),
                'classifier': classifier.state_dict() if classifier else None,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
                'scheduler': scheduler.state_dict() if scheduler else None,
                'lr': lr
               }, os.path.join(chk_dir, 'epoch_%d.pkl' % epoch))
    
def load_pretrained_modules(model, ckpt):
    model_info = ckpt
    state_dict = OrderedDict()
    for k, v in model_info['model'].items():
        name = k.replace("module.", "").replace("convolution_", "convolution_module.")   # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)

    return model