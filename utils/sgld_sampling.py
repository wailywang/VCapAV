import torch as t
import numpy as np
import random

def get_buffer(args, **kwargs):
#     model_cls = F if args.uncond else CCF
    if not args.uncond:
        assert args.buffer_size % 2 == 0, "Buffer size must be divisible by two classes (genuine and fake)"

    # make replay buffer
    replay_buffer = init_random(args, args.buffer_size)

    return replay_buffer


def init_random(args, bs):
    # return t.FloatTensor(bs, 320, 16, 16).uniform_(-0.1, 0.1) # 10000,230,16,16
    if args.model == 'LightCNN_eow' or args.model == 'LightCNN_partition_ASP':
        # frame_size = random.randint(31, 78)
        frame_size = 109
        return t.FloatTensor(bs, 32, 10, frame_size).uniform_(-0.1, 0.1) # fbank LCNN
    elif args.model == 'ResNet18_ASP_eow':
        frame_size = 438
        return t.FloatTensor(bs, 32, 40, frame_size).uniform_(-0.1, 0.1) # fbank_ResNet18_ASP
    elif args.model == 'WideResnet_eow':
        # [bs, 64, 30, t]
        frame_size = 110
        return t.FloatTensor(bs, 64, 40, frame_size).uniform_(-0.1, 0.1) # fbank_WideResNet
    else:
        AssertionError('Please specify the model name')

def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes # y is none so buffer_size = 10,000
        inds = t.randint(0, buffer_size, (bs,)) #从0~9999中取出batch_size个序号
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        inds = inds.long() # 变为长整形
        buffer_samples = replay_buffer[inds] #从10000个缓冲区随机初始化embd中取出batch_size个
        random_samples = init_random(args, bs) # 再初始化一次长度为batch_size的缓冲区，此处为64
        choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None] # shape为batch_size的随机浮点数矩阵，小于0.05时被选出
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples # batch size 中的%5被重新初始化，95%遵循之前的缓冲区初始化。
        return samples.to(device), inds

    def sample_q(f, c, replay_buffer, n_steps=args.n_steps): # repaly_buffer:[10000,320,16,16]
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        c.eval()
        # get batch size
        bs = args.batch_size
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs)# 返回第一次初始化的embd以及缓冲区中的初始化位置
        embedding_k = t.autograd.Variable(init_sample, requires_grad=True)# 变为梯度可学习的变量
        # sgld
        for k in range(1, n_steps +1):# SGLD迭代100步
            negative_energy = -t.log(t.softmax(c(f(embedding_k, in_embd=True)), dim=1).squeeze()[:, -1] + 1e-12) # 此处的能量函数为F的最后几层, 返回对logits 计算softmax后的负对数，并降维
            f_prime = t.autograd.grad(negative_energy.sum(), [embedding_k], retain_graph=True)[0] # 计算梯度
            embedding_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(embedding_k) # 更新embedding_k += 1*梯度 + 0.01*embd同shape噪声 
        # 1/0
        f.train()
        c.train()
        final_samples = embedding_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
    return sample_q