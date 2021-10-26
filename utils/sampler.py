import numpy as np
import torch
from torch.distributions import Dirichlet

#--------------------Maggie from amr twogan.py的几种mix模式------------------------------------
def BetaSampler(bs, f, is_2d, p=None, beta_alpha=1):  # 有四种采样函数, p是伯努利参数，当p=None时，指示p从均匀分布U(0,1)中采样
    """Mixup sampling function
    :param bs: batch size=w.size(0)=[14,512]中的14              # 由于本实验中是逐个mixup,batchsize=1,所以此处是1 后来rmt训练时改成了batch整体采样混合
    :param f: number of features / channels=w.size(1)=[14,512]中的512   # 本实验中也是512
    :param is_2d: should sampled alpha be 2D, instead of 4D?                #本实验中is_2d=true
    :param beta_alpha: Beta parameter `beta_alpha`. If this 1, then we simply sample p ~ beta(beta_alpha,beta_alpha).   #本实验中p=None，所以alpha从beta(alpha,alpha)中采样
    :returns: an alpha of shape (bs, 1) if `is_2d`, otherwise (bs, 1, 1, 1).
    :rtype: 
    """
    shp = (bs, 1) if is_2d else (bs, 1, 1, 1)
    # print('alphas shp:')                                  #   torch.Size([1, 1])
    if p is None:
        alphas = []
        for i in range(bs):
            # alpha = np.random.uniform(0, 1)       
            alpha = np.random.beta(beta_alpha, beta_alpha)      #sample from beta(beta_alpha,beta_alpha)
            alphas.append(alpha)
    else:
        alphas = [p]*bs
    alphas = np.asarray(alphas).reshape(shp)
    alphas = torch.from_numpy(alphas).float()
    
    # # print(alphas.shape)    
    # use_cuda = True if torch.cuda.is_available() else False
    # if use_cuda:
    #     alphas = alphas.cuda()
    # # raise Exception("error")
    return alphas       #   cpu tensor

def UniformSampler(bs, f, is_2d, p=None):  # 有四种采样函数, p是伯努利参数，当p=None时，指示p从均匀分布U(0,1)中采样
    """Mixup sampling function
    :param bs: batch size=w.size(0)=[14,512]中的14              # 由于本实验中是逐个mixup,batchsize=1,所以此处是1
    :param f: number of features / channels=w.size(1)=[14,512]中的512   # 本实验中也是512
    :param is_2d: should sampled alpha be 2D, instead of 4D?                #本实验中is_2d=true
    :param p: Bernoulli parameter `p`. If this is None, then we simply sample p ~ U(0,1).   #本实验中p=None，所以alpha从U(0,1)中采样
    :returns: an alpha of shape (bs, 1) if `is_2d`, otherwise (bs, 1, 1, 1).
    :rtype: 
    """
    # print('flag:UniformSampler ing')

    shp = (bs, 1) if is_2d else (bs, 1, 1, 1)
    # print('alphas shp:')                                  #   torch.Size([1, 1])
    if p is None:
        alphas = []
        for i in range(bs):
            alpha = np.random.uniform(0, 1)
            alphas.append(alpha)
    else:
        alphas = [p]*bs
    alphas = np.asarray(alphas).reshape(shp)
    alphas = torch.from_numpy(alphas).float()
    
    # # print(alphas.shape)    
    # use_cuda = True if torch.cuda.is_available() else False
    # if use_cuda:
    #     alphas = alphas.cuda()
    # # raise Exception("error")

    return alphas

def UniformSampler2(bs, f, is_2d, p=None):
    """Mixup2 sampling function
    :param bs: batch size
    :param f: number of features / channels
    :param is_2d: should sampled alpha be 2D, instead of 4D?
    :param p: Bernoulli parameter `p`. If this is None, then we simply sample p ~ U(0,1).       
    :returns: an alpha of shape (bs, f) if `is_2d`, otherwise (bs, f, 1, 1).
    :rtype:
    """
    print('flag:UniformSampler2 ing')
    shp = (bs, f) if is_2d else (bs, f, 1, 1)
    # print('alphas shp:')
    # print(shp)

    if p is None:
        alphas = np.random.uniform(0, 1, size=shp)
    else:
        alphas = np.zeros(shp)+p
    alphas = torch.from_numpy(alphas).float()
    
    # print(alphas.shape)    
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alphas = alphas.cuda()
    return alphas

def BernoulliSampler(bs, f, is_2d, p=None):
    """Bernoulli mixup sampling function
    :param bs: batch size
    :param f: number of features / channels
    :param is_2d: should sampled alpha be 2D, instead of 4D?
    :param p: Bernoulli parameter `p`. If this is `None`, then we simply sample m ~ Bern(p), where p ~ U(0,1).  #本实验中p都是none, 所以p从U(0,1)中采样
    :returns: an alpha of shape (bs, f) if `is_2d`, otherwise (bs, f, 1, 1).
    :rtype:
    """
    shp = (bs, f) if is_2d else (bs, f, 1, 1)
    # print('alphas shp:',shp)                            #   (1, 512)

    if p is None:
        # print("sample here------")
        alphas = torch.bernoulli(torch.rand(shp)).float()
        # print('alphas.shape:',alphas.shape)                            #  alphas.shape: torch.Size([1, 512])
        # print('alphas:',alphas)                            #  [0,1]mask

    else:
        rnd_state = np.random.RandomState(0)
        rnd_idxs = np.arange(0, f)
        rnd_state.shuffle(rnd_idxs)
        rnd_idxs = torch.from_numpy(rnd_idxs)
        how_many = int(p*f)
        alphas = torch.zeros(shp).float()
        if how_many > 0:
            rnd_idxs = rnd_idxs[0:how_many]
            alphas[:, rnd_idxs] += 1.
    # print(alphas.shape)    

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alphas = alphas.cuda()
    # raise Exception("maggie 20210905")
    return alphas

def BernoulliSampler2(bs, f, is_2d, p=None):
    """Bernoulli mixup sampling function. Has same expectation as fm but higher variance.Bernoulli混合采样函数。与fm期望值相同，但方差更高。
    :param bs: batch size
    :param f: number of features / channels
    :param is_2d: should sampled alpha be 2D, instead of 4D?
    :param p: Bernoulli parameter `p`. If this is `None`, then we simply sample m ~ Bern(p), where p ~ U(0,1).
    :returns: an alpha of shape (bs, f) if `is_2d`, otherwise (bs, f, 1, 1).
    :rtype: 
    """
    shp = (bs, f) if is_2d else (bs, f, 1, 1)
    # print('alphas shp:')
    # print(shp)

    if p is None:
        this_p = torch.rand(1).item()
        alphas = torch.bernoulli(torch.zeros(shp)+this_p).float()
    else:
        rnd_state = np.random.RandomState(0)
        rnd_idxs = np.arange(0, f)
        rnd_state.shuffle(rnd_idxs)
        rnd_idxs = torch.from_numpy(rnd_idxs)
        how_many = int(p*f)
        alphas = torch.zeros(shp).float()
        if how_many > 0:
            rnd_idxs = rnd_idxs[0:how_many]
            alphas[:, rnd_idxs] += 1.

    # print(alphas.shape)    
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alphas = alphas.cuda()
    return alphas

def DirichletSampler(bs, f, is_2d, dirichlet_gama=9.0):
    """Uniform Sample for 3 ws mix """                  #   本实验中 is_2d = true
    # print('flag:DirichletSampler ing')                #   相当于3mix场景下的uniformsample,为512维分量分配相同alpha
    # Dirichlet_gama = 9.0
    # print("is_2d:",is_2d)
    with torch.no_grad():
        # dirichlet = Dirichlet(torch.FloatTensor([1.0, 1.0, 1.0]))
        dirichlet = Dirichlet(torch.FloatTensor([dirichlet_gama, dirichlet_gama, dirichlet_gama]))
        alpha = dirichlet.sample_n(bs)
        if not is_2d:
            alpha = alpha.reshape(-1, alpha.size(1), 1, 1)                              
    # print(alpha.shape)                                                                  # torch.Size([1, 3])

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alpha = alpha.cuda()
    return alpha  
    
def BernoulliSampler3(bs, f, is_2d):            #   相当于3mix场景下的bernoullisample,为512维分量分配不同alpha
    """Bernoulli Sample for 3 ws mix """
    # print('flag:BernoulliSampler3 ing')
    # print("is_2d:",is_2d)       #is_2d: True

    if is_2d:
        alpha = np.zeros((bs, 3, f)).astype(np.float32)    
        # print("alpha.shape:",alpha.shape)     # alpha.shape: (1, 3, 512)
    else:
        alpha = np.zeros((bs, 3, f, 1, 1)).astype(np.float32)
    for b in range(bs):
        for j in range(f):
            alpha[b, np.random.randint(0,3), j] = 1.
    alpha = torch.from_numpy(alpha).float()    

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alpha = alpha.cuda()
    return alpha  