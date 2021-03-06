import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from random import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2 as cv

class Util(object):
    def __init__(self, args):
        self.args = args

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def random_bbox(self):
        img_shape = self.args.IMG_SHAPE                                       # [512, 680, 3]
        img_height = img_shape[0]                                             # 512
        img_width = img_shape[1]                                              # 680

        maxt = img_height - self.args.VERTICAL_MARGIN - self.args.MASK_HEIGHT # 512 - 0 - 102 = 410
        maxl = img_width - self.args.HORIZONTAL_MARGIN - self.args.MASK_WIDTH # 680 - 0 - 170 = 510

        t = randint(self.args.VERTICAL_MARGIN, maxt)                          # 0~410 중 랜덤한 숫자 반환
        l = randint(self.args.HORIZONTAL_MARGIN, maxl)                        # 0~680 중 랜덤한 숫자 반환
        h = self.args.MASK_HEIGHT                                             # 102
        w = self.args.MASK_WIDTH                                              # 170
        return (t, l, h, w)                                                   # (0~410 중 하나, 0~680 중 하나, 102, 170)

    def bbox2mask(self, bbox, in_image):                                                # bbox : (0~410 중 하나, 0~680 중 하나, 102, 170)
        """Generate mask tensor from bbox.

        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

        Returns:
            tf.Tensor: output with shape [B, 1, H, W]

        """

        # def npmask(bbox, height, width, delta_h, delta_w):
        #     mask = np.zeros((1, 1, height, width), np.float32)
        #     h = np.random.randint(delta_h//2+1)
        #     w = np.random.randint(delta_w//2+1)
        #     mask[:, :, bbox[0]+h : bbox[0]+bbox[2]-h,
        #          bbox[1]+w : bbox[1]+bbox[3]-w] = 1.
        #     return mask

        def npmask(h, w):                                                       # 102,170
            max_width = 128
            if h > 512 or w > 680:                                              # h, w가 너무 크면 에러 발생
                raise Exception("width and height of mask be at least 680x512!")
            np.random.seed(0)  ## maskseed = 22, 100 for test
            number = np.random.randint(16, 30)                                  # 16~30 랜덤한 정수값
            mask = np.zeros((h, w))                                             # 0으로 가득찬 (512, 680) 크기 마스크 생성
            for _ in range(number):
                model = np.random.random()                                      # 0~1 사이값
                if model < 0.6:                                                 # Draw random lines
                    x1, x2 = np.random.randint(1, h), np.random.randint(1, h)
                    y1, y2 = np.random.randint(1, w), np.random.randint(1, w)
                    thickness = np.random.randint(4, max_width)
                    cv.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

                elif model >= 0.6 and model < 0.8:                              # Draw random circle
                    x1, y1 = np.random.randint(1, h), np.random.randint(1, w)
                    radius = np.random.randint(4, max_width)
                    cv.circle(mask, (x1, y1), radius, (1, 1, 1), -1)

                else:                                                           # Draw random ellipses
                    x1, y1 = np.random.randint(1, h), np.random.randint(1, w)
                    s1, s2 = np.random.randint(1, h), np.random.randint(1, w)
                    a1 = np.random.randint(3, 180)
                    a2 = np.random.randint(3, 180)
                    a3 = np.random.randint(3, 180)
                    thickness = np.random.randint(4, max_width)
                    cv.ellipse(mask, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)
            mask4d = np.zeros((2, 3, h, w), np.float32)                         # 0으로 가득찬 (1, 1, 512, 680) 크기 마스크 생성
            mask4d[0][0] = mask                                                 #
            return mask4d

        img_shape = self.args.IMG_SHAPE                                         # [512, 680, 3]
        height = img_shape[0]                                                   # 512
        width = img_shape[1]                                                    # 680
        # mask = npmask(bbox, height, width, self.args.MAX_DELTA_HEIGHT, self.args.MAX_DELTA_WIDTH) # ((0~410 중 하나, 0~680 중 하나, 102, 170), 512, 680, 64, 64)
        mask = npmask(in_image.shape[2], in_image.shape[3])

        return torch.FloatTensor(mask)

    def local_patch(self, x, bbox):
        '''
        bbox[0]: top
        bbox[1]: left
        bbox[2]: height
        bbox[3]: width
        '''
        x = x[:, :, bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]]
        return x


class Discounted_L1(nn.Module):
    def __init__(self, args, size_average=True, reduce=True):
        super(Discounted_L1, self).__init__()
        self.reduce = reduce
        self.discounting_mask = spatial_discounting_mask(args.MASK_WIDTH,
                                                         args.MASK_HEIGHT,
                                                         args.SPATIAL_DISCOUNTING_GAMMA)
        self.size_average = size_average

    def forward(self, input, target):
        self._assert_no_grad(target)
        return self._pointwise_loss(lambda a, b: torch.abs(a - b), torch._C._nn.l1_loss,
                                    input, target, self.discounting_mask, self.size_average, self.reduce)

    def _assert_no_grad(self, variable):
        assert not variable.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

    def _pointwise_loss(self, lambd, lambd_optimized, input, target, discounting_mask, size_average=True, reduce=True):
        if target.requires_grad:
            d = lambd(input, target)
            d = d * discounting_mask
            if not reduce:
                return d
            return torch.mean(d) if size_average else torch.sum(d)
        else:
            return lambd_optimized(input, target)


def spatial_discounting_mask(mask_width, mask_height, discounting_gamma):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = discounting_gamma
    shape = [1, 1, mask_width, mask_height]
    if True:
        print('Use spatial discounting l1 loss.')
        mask_values = np.ones((mask_width, mask_height))
        for i in range(mask_width):
            for j in range(mask_height):
                mask_values[i, j] = max(
                    gamma ** min(i, mask_width - i),
                    gamma ** min(j, mask_height - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 1)
        mask_values = mask_values
    else:
        mask_values = np.ones(shape)
    # it will be extended along the batch dimension suitably
    mask_values = torch.from_numpy(mask_values).float()
    return to_var(mask_values)


def down_sample(x, size=None, scale_factor=None, mode='nearest'):
    # define size if user has specified scale_factor
    if size is None: size = (int(scale_factor * x.size(2)), int(scale_factor * x.size(3)))
    # create coordinates
    h = torch.arange(0, size[0]) / (size[0] - 1) * 2 - 1
    w = torch.arange(0, size[1]) / (size[1] - 1) * 2 - 1
    # create grid
    grid = torch.zeros(size[0], size[1], 2)
    grid[:, :, 0] = w.unsqueeze(0).repeat(size[0], 1)
    grid[:, :, 1] = h.unsqueeze(0).repeat(size[1], 1).transpose(0, 1)
    # expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
    if x.is_cuda: grid = Variable(grid).cuda()
    # do sampling
    return F.grid_sample(x, grid, mode=mode)


def reduce_mean(x):
    for i in range(4):
        if i == 1: continue
        x = torch.mean(x, dim=i, keepdim=True)
    return x


def l2_norm(x):
    def reduce_sum(x):
        for i in range(4):
            if i == 1: continue
            x = torch.sum(x, dim=i, keepdim=True)
        return x

    x = x ** 2
    x = reduce_sum(x)
    return torch.sqrt(x)


def show_image(real, masked, stage_1, stage_2, fake, offset_flow):
    batch_size = real.shape[0]

    (real, masked, stage_1, stage_2, fake, offset_flow) = (
        var_to_numpy(real),
        var_to_numpy(masked),
        var_to_numpy(stage_1),
        var_to_numpy(stage_2),
        var_to_numpy(fake),
        var_to_numpy(offset_flow)
    )
    # offset_flow = (offset_flow*2).astype(int) -1
    for x in range(batch_size):
        if x > 5:
            break
        fig, axs = plt.subplots(ncols=5, figsize=(15, 3))
        axs[0].set_title('real image')
        axs[0].imshow(real[x])
        axs[0].axis('off')

        axs[1].set_title('masked image')
        axs[1].imshow(masked[x])
        axs[1].axis('off')

        axs[2].set_title('stage_1 image')
        axs[2].imshow(stage_1[x])
        axs[2].axis('off')

        axs[3].set_title('stage_2 image')
        axs[3].imshow(stage_2[x])
        axs[3].axis('off')

        axs[4].set_title('fake_image')
        axs[4].imshow(fake[x])
        axs[4].axis('off')

        # axs[5].set_title('C_Attn')
        # axs[5].imshow(offset_flow[x])
        # axs[5].axis('off')

        plt.show()


def var_to_numpy(obj, for_vis=True):
    if for_vis:
        obj = obj.permute(0, 2, 3, 1)
        obj = (obj + 1) / 2
    return obj.data.cpu().numpy()


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()