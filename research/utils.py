import torch
from torch.nn import functional as F

def repeat_expand_2d(content, target_len, mode = 'left'):
    # content : [h, t]
    return repeat_expand_2d_left(content, target_len) if mode == 'left' else repeat_expand_2d_other(content, target_len, mode)

def repeat_expand_2d_left(content, target_len):
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len+1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos+1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target

# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
def repeat_expand_2d_other(content, target_len, mode = 'nearest'):
    # content : [h, t]
    content = content[None,:,:]
    target = F.interpolate(content,size=target_len,mode=mode)[0]
    return target



# f0, uv = np.load(wav_dir.replace('.wav','.f0.npy'),allow_pickle=True)
# f0 = torch.FloatTensor(np.array(f0*uv,dtype=float))

# f0 = utils.repeat_expand_2d(f0[None,:], hubert.shape[-1], mode='nearest').squeeze()

# f0 encode to lf0
# lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
# f0 decode from lf0
# f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)