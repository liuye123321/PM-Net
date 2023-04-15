import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmseg.core import add_prefix
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

class Attention_fussion(nn.Module):
    # Not using location
    def __init__(self, indim, dim):
        super(Attention_fussion, self).__init__()
        # self.Key_r = nn.Conv2d(indim, dim, kernel_size=2, padding=0, stride=2)
        # self.Query_r = nn.Conv2d(indim, dim, kernel_size=2, padding=0, stride=2)
        # self.Value_r = nn.Conv2d(indim, dim, kernel_size=2, padding=0, stride=2)
        self.Key = nn.Conv2d(indim, dim, kernel_size=3, padding=1, stride=1)
        self.Query = nn.Conv2d(indim, dim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, dim, kernel_size=3, padding=1, stride=1)
    def forward(self, x):
        key = self.Key(x)
        value = self.Value(x)
        query = self.Query(x)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        out = torch.matmul(p_attn, value)
        # out = F.upsample(torch.matmul(p_attn, value), size=x.size()[2:], mode='bilinear',
        #            align_corners=True)
        return out


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        # self.Key = nn.Linear(indim, keydim)
        # self.Value = nn.Linear(indim, valdim)
        self.keydim = keydim
        self.valdim = valdim
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        k4 = self.Key(x)
        v4 = self.Value(x)
        return k4, v4

class Key(nn.Module):
    # Not using location
    def __init__(self, indim, keydim):
        super(Key, self).__init__()
        # self.Key = nn.Linear(indim, keydim)
        # self.Value = nn.Linear(indim, valdim)
        self.keydim = keydim
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        k4 = self.Key(x)
        return k4

# class Memory(nn.Module):
#     def __init__(self):
#         super(Memory, self).__init__()
#
#     def forward(self, m_in, m_out, q_in, q_out, p_m_in, p_q_in):
#         _, _, H, W = q_in.size()
#         no, centers, C = m_in.size()
#         _, _, vd = m_out.shape
#
#         qi = q_in.view(-1, C, H * W)
#         p = torch.bmm(m_in, qi)  # no x centers x hw
#         p = p / math.sqrt(C)
#         p = torch.softmax(p, dim=1)  # no x centers x hw
#
#         _, _, p_H, p_W = q_in.size()
#         p_no, p_centers, p_C = p_m_in.size()
#         p_qi = p_q_in.view(-1, p_C, p_H * p_W)
#         p_p = torch.bmm(p_m_in, p_qi)  # no x centers x hw
#         p_p = p_p / math.sqrt(p_C)
#         p_p = torch.softmax(p_p, dim=1)  # no x centers x hw
#
#
#         mo = m_out.permute(0, 2, 1)  # no x c x centers
#         mem = torch.bmm(mo, torch.softmax(p * p_p, dim=1))
#         mem = mem.view(no, vd, H, W)
#
#         mem_out = torch.cat([mem, q_out], dim=1)
#         return mem_out

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, m_out, q_in, q_out, p_m_in, p_q_in):
        _, _, H, W = q_in.size()
        no, centers, C = m_in.size()
        _, _, vd = m_out.shape

        qi = q_in.view(-1, C, H * W)
        p = torch.bmm(m_in, qi)  # no x centers x hw
        p = p / math.sqrt(C)
        p = torch.softmax(p, dim=1)  # no x centers x hw

        _, _, p_H, p_W = p_q_in.size()
        p_no, p_centers, p_C = p_m_in.size()
        p_qi = p_q_in.view(-1, p_C, p_H * p_W)
        p_p = torch.bmm(p_m_in, p_qi)  # no x centers x hw
        p_p = p_p / math.sqrt(p_C)
        p_p = torch.softmax(p_p, dim=1)  # no x centers x hw


        mo = m_out.permute(0, 2, 1)  # no x c x centers
        mem = torch.bmm(mo, p)
        mem = mem.view(no, vd, H, W)
        mem_p = torch.bmm(mo, p_p)
        mem_p = mem_p.view(no, vd, H, W)
        ori = mem + mem_p
        mem_out = torch.cat([ori, q_out], dim=1)

        return mem_out

def attention_pool(tensor, pool, norm=None):
    if pool is None:
        return tensor
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")
    tensor = pool(tensor)

    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor

class Memory_pooling(nn.Module):
    def __init__(self, keydim, valdim):
        super(Memory_pooling, self).__init__()
        self.keydim = keydim
        self.valdim = valdim
        self.kernel = 2
        self.stride = 2
        self.pool_q_key = (
            nn.Conv2d(
                self.keydim,
                self.keydim,
                kernel_size=self.kernel,
                stride=self.stride,
                bias=False,
            )
        )
        self.pool_m_key = (
            nn.Conv2d(
                2 * self.keydim,
                2 * self.keydim,
                kernel_size=self.kernel,
                stride=self.stride,
                bias=False,
            )
        )
        self.pool_q_val = (
            nn.Conv2d(
                self.valdim,
                self.valdim,
                kernel_size=self.kernel,
                stride=self.stride,
                bias=False,
            )
        )
        self.pool_m_val = (
            nn.Conv2d(
                2 * self.valdim,
                2 * self.valdim,
                kernel_size=self.kernel,
                stride=self.stride,
                bias=False,
            )
        )
        self.pool_val = (
            nn.Conv2d(
                self.valdim,
                self.valdim,
                kernel_size=self.kernel,
                stride=self.stride,
                bias=False,
            )
        )
        self.fusion = Memory_multimode(self.keydim, self.valdim)
        self.reduce_dim = nn.Conv2d(
                2 * self.valdim,
                self.valdim,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1
            )

    def forward(self, m_key, m_val, q_key, q_val, p_m_key, p_q_key):
        # pooling
        q_key_pool = attention_pool(q_key, pool=self.pool_q_key)
        m_key_pool = attention_pool(m_key, pool=self.pool_m_key)
        m_val_pool = attention_pool(m_val, pool=self.pool_m_val)
        p_m_key_pool = attention_pool(p_m_key, pool=self.pool_m_key)
        p_q_key_pool = attention_pool(p_q_key, pool=self.pool_q_key)
        q_val_pool = attention_pool(q_val, pool=self.pool_q_val)
        mem_out = self.fusion(m_key_pool, m_val_pool, q_key_pool, q_val_pool, p_m_key_pool, p_q_key_pool)
        mem_out = self.reduce_dim(F.upsample(mem_out, scale_factor=2))

        return mem_out

class Memory_multimode(nn.Module):
    def __init__(self, keydim, valdim):
        super(Memory_multimode, self).__init__()
        self.keydim = keydim
        self.valdim = valdim
    def forward(self, m_key, m_val, q_key, q_val, p_m_key, p_q_key):
        B, C, H, W = m_val.size()
        ## haze && haze
        B_m, C_m, H_m, W_m = m_key.size()
        m_val = m_val.unsqueeze(1).view(B_m, -1, 2 * H * W).permute(0, 2, 1)
        ii = torch.bmm(q_key.view(B_m, H_m*W_m, -1), m_key.view(B_m, -1, 2*H*W))
        ii = torch.softmax(ii, dim=1)
        iio = torch.bmm(ii, m_val).view(B_m, -1, H_m, W_m)
        ## phase && phase
        B_p, C_p, H_p, W_p = p_m_key.size()
        p_m_key = p_m_key.unsqueeze(1).reshape(B_p, 2, -1, H_p, W_p)
        pp = torch.bmm(p_q_key.view(B_p, H_p * W_p, -1), p_m_key.view(B_p, -1, 2 * H_p * W_p))
        pp = torch.softmax(pp, dim=1)
        ppo = torch.bmm(pp, m_val).view(B_p, -1, H_p, W_p)
        ## haze && phase
        m_key = m_key.unsqueeze(1).reshape(B_m, 2, -1, H_m, W_m)
        ip = torch.bmm(p_q_key.view(B_m, H_m * W_m, -1), m_key.view(B_m, -1, 2 * H_m * W_m))
        ip = torch.softmax(ip, dim=1)
        ipo = torch.bmm(ip, m_val).view(B_m, -1, H, W)

        ## phase && hase
        pi = torch.bmm(q_key.view(B_p, H_p * W_p, -1), p_m_key.view(B_p, -1, 2 * H_p * W_p))
        pi = torch.softmax(pi, dim=1)
        pio = torch.bmm(pi, m_val).view(B_p, -1, H_p, W_p)

        mem_out = torch.cat([iio+ppo+ipo+pio, q_val], dim=1)
        return mem_out

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        super(EncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        pretrained_P = None
        self.init_weights(pretrained=pretrained, pretrained_p=pretrained_P)
        self.conv = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.keydim_rgb = 256
        self.valdim_rgb = 512
        self.keydim_p = 256
        self.KV_M_r4 = KeyValue(512, keydim=self.keydim_rgb, valdim=self.valdim_rgb)
        self.KV_Q_r4 = KeyValue(512, keydim=self.keydim_rgb, valdim=self.valdim_rgb)
        self.K_M_r4_p = Key(80, keydim=self.keydim_p)
        self.K_Q_r4_p = Key(80, keydim=self.keydim_p)
        self.reduce_dim = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)  #1536，2560
        self.memory_phase_guided_1 = Memory_multimode(keydim=self.keydim_rgb, valdim =self.valdim_rgb)
        # self.memory_phase_guided_1 = Memory_pooling(keydim=self.keydim_rgb, valdim=self.valdim_rgb)
        # self.memory_phase_guided_2 = Memory_pooling(keydim=self.keydim_rgb, valdim=self.valdim_rgb)
        # self.memory_phase_guided_3 = Memory_pooling(keydim=self.keydim_rgb, valdim=self.valdim_rgb)
        # self.memory_phase_guided_4 = Memory_pooling(keydim=self.keydim_rgb, valdim=self.valdim_rgb)
        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None, pretrained_p = None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained, pretrained_p)
        self.backbone[0].init_weights(pretrained=pretrained)
        self.backbone[1].init_weights(pretrained=pretrained)
        self.backbone[2].init_weights(pretrained=pretrained_p)
        self.backbone[3].init_weights(pretrained=pretrained_p)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()


    def extract_feat(self, haze, ref, ref_gt):
        input_rgb_haze = haze
        out_rgb_haze = self.backbone[0](input_rgb_haze)
        if self.with_neck:
            out_rgb_haze = self.neck(out_rgb_haze)

        input_rgb_ref0 = ref[0]
        input_rgb_refgt0 = ref_gt[0]
        out_rgb_ref0 = self.backbone[1](self.conv(torch.cat([input_rgb_ref0, input_rgb_refgt0], dim=1)))
        if self.with_neck:
            out_rgb_ref0 = self.neck(out_rgb_ref0)

        # input_rgb_ref1 = ref[1]
        # input_rgb_refgt1 = ref_gt[1]
        # out_rgb_ref1 = self.backbone[1](self.conv(torch.cat([input_rgb_ref1, input_rgb_refgt1], dim=1)))
        # if self.with_neck:
        #     out_rgb_ref1 = self.neck(out_rgb_ref1)

        haze_ = abs(torch.angle(torch.fft.fftn(haze, dim=(-2, -1))))
        ref0_ = abs(torch.angle(torch.fft.fftn(ref[0], dim=(-2, -1))))
        # ref1_ = abs(torch.angle(torch.fft.fftn(ref[1], dim=(-2, -1))))

        input_P_haze = haze_
        out_P_haze = self.backbone[2](input_P_haze)
        if self.with_neck:
            out_P_haze = self.neck(out_P_haze)

        input_P_ref0 = ref0_
        out_P_ref0 = self.backbone[3](input_P_ref0)
        if self.with_neck:
            out_P_ref0 = self.neck(out_P_ref0)

        # input_P_ref1 = ref1_
        # out_P_ref1 = self.backbone[3](input_P_ref1)
        # if self.with_neck:
        #     out_P_ref1 = self.neck(out_P_ref1)
        rgb_haze_k4, rgb_haze_v4 = self.KV_Q_r4(out_rgb_haze[-1])
        rgb_ref0_k4, rgb_ref0_v4 = self.KV_M_r4(out_rgb_ref0[-1])
        # rgb_ref1_k4, rgb_ref1_v4 = self.KV_M_r4(out_rgb_ref1[-1])

        P_haze_k4 = self.K_Q_r4_p(out_P_haze[-1])
        P_ref0_k4 = self.K_M_r4_p(out_P_ref0[-1])
        # P_ref1_k4 = self.K_M_r4_p(out_P_ref1[-1])
        memory_phase_guided_1 = self.memory_phase_guided_1(
            torch.cat([rgb_ref0_k4, rgb_ref0_k4], dim=1),
            torch.cat([rgb_ref0_v4, rgb_ref0_v4], dim=1),
            rgb_haze_k4, rgb_haze_v4,
            torch.cat([P_ref0_k4, P_ref0_k4], dim=1), P_haze_k4)
        # memory_phase_guided_1 = self.memory_phase_guided_1(
        #     torch.cat([rgb_ref0_k4, rgb_ref1_k4], dim=1),
        #     torch.cat([rgb_ref0_v4, rgb_ref1_v4], dim=1),
        #     rgb_haze_k4, rgb_haze_v4,
        #     torch.cat([P_ref0_k4, P_ref1_k4], dim=1), P_haze_k4)
        out_rgb_haze[-1] = self.reduce_dim(memory_phase_guided_1) + out_rgb_haze[-1]
        # out_rgb_haze[-1] = memory_phase_guided_4 + out_rgb_haze[-1]
        return out_rgb_haze

    # 测试
    def encode_decode(self, img, ref, gt, img_metas, refgt):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img, ref, refgt)
        out = self._decode_head_forward_test(x, img_metas, img)
        return out['haze']

    def _decode_head_forward_train(self, x, img_metas, gt, haze):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,    #调用mmseg/models/decode_heads/decoder_head里面的forward_train
                                                     gt, haze,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas, img):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img, img_metas)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, img, img, None, img)

        return seg_logit

    def forward_train(self, img, img_metas, ref, gt, refgt):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img, ref, refgt)
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas, gt, img)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas)
            losses.update(loss_aux)

        return losses

    def slide_inference(self, img, img_meta, ref, gt):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                ref_img = []
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_gt = gt[:, :, y1:y2, x1:x2]
                size_end_h = h_stride + (h_crop - h_stride) // 2
                size_end_w = w_stride + (w_crop - w_stride) // 2
                size_head_h = (h_crop - h_stride) // 2
                size_head_w = (w_crop - w_stride) // 2
                for i in range(len(ref)):
                    ref_img.append(ref[i][:, :, y1:y2, x1:x2])
                crop_seg_logit = self.encode_decode(crop_img, ref_img, crop_gt, img_meta)
                # if int(x1) == 0:
                #     crop_seg_logit[:, :, :, size_end_w:-1] = 0
                # if int(y1) == 0:
                #     crop_seg_logit[:, :, size_end_h:-1, :] = 0
                # if int(preds.shape[3] - x2) == 0:
                #     crop_seg_logit[:, :, :, 0:size_head_w - 1] = 0
                # if int(preds.shape[2] - y2) == 0:
                #     crop_seg_logit[:, :, 0:size_head_h - 1, :] = 0
                # else:
                #     crop_seg_logit[:, :, 0:size_head_h-1, 0:size_head_w-1] = 0
                #     crop_seg_logit[:, :, size_end_h:-1, size_end_w:-1] = 0
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        return preds, gt

    def slide_inference_(self, img, img_meta, ref, gt):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        batch_size, _, h_img, w_img = img.size()
        patch1, patch2, patch3, patch4 = self.test_cfg.test_size
        h, w = self.test_cfg.drop_size
        ##patch1
        crop_img = img[:, :, patch1[0]:patch1[1], patch1[2]:patch1[3]]
        crop_gt = gt[:, :, patch1[0]:patch1[1], patch1[2]:patch1[3]]
        ref_img = []
        for j in range(len(ref)):
            ref_img.append(ref[j][:, :, patch1[0]:patch1[1], patch1[2]:patch1[3]])
        crop_seg_logit = self.encode_decode(crop_img, ref_img, crop_gt, img_meta)
        preds[:, :, patch1[0]:patch1[1]-h, patch1[2]:patch1[3]-w] = crop_seg_logit[:, :, 0:-h, 0:-w]
        ##patch2
        crop_img = img[:, :, patch2[0]:patch2[1], patch2[2]:patch2[3]]
        crop_gt = gt[:, :, patch2[0]:patch2[1], patch2[2]:patch2[3]]
        ref_img = []
        for j in range(len(ref)):
            ref_img.append(ref[j][:, :, patch2[0]:patch2[1], patch2[2]:patch2[3]])
        crop_seg_logit = self.encode_decode(crop_img, ref_img, crop_gt, img_meta)
        preds[:, :, patch2[0]:patch2[1]-h, patch2[2]+w+1:patch2[3]] = crop_seg_logit[:, :, 0:-h, w:-1]

        ##patch3
        crop_img = img[:, :, patch3[0]:patch3[1], patch3[2]:patch3[3]]
        crop_gt = gt[:, :, patch3[0]:patch3[1], patch3[2]:patch3[3]]
        ref_img = []
        for j in range(len(ref)):
            ref_img.append(ref[j][:, :, patch3[0]:patch3[1], patch3[2]:patch3[3]])
        crop_seg_logit = self.encode_decode(crop_img, ref_img, crop_gt, img_meta)
        preds[:, :, patch3[0]+h+1:patch3[1], patch3[2]:patch3[3] - w] = crop_seg_logit[:, :, h:-1, 0:-w]
        ##patch4
        crop_img = img[:, :, patch4[0]:patch4[1], patch4[2]:patch4[3]]
        crop_gt = gt[:, :, patch4[0]:patch4[1], patch4[2]:patch4[3]]
        ref_img = []
        for j in range(len(ref)):
            ref_img.append(ref[j][:, :, patch4[0]:patch4[1], patch4[2]:patch4[3]])
        crop_seg_logit = self.encode_decode(crop_img, ref_img, crop_gt, img_meta)
        preds[:, :, patch4[0] + h+1:patch4[1], patch4[2] + w + 1:patch4[3]] = crop_seg_logit[:, :, h:-1, w:-1]

        return preds, gt

    def whole_inference(self, img, img_meta, ref, gt, refgt):
        """Inference with full image."""
        seg_logit = self.encode_decode(img, ref, gt, img_meta, refgt)
        return seg_logit, gt

    def inference(self, img, img_meta, ref, gt, refgt):
        """Inference with slide/whole style.
        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.
        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, ref[0], gt[0])
        else:
            seg_logit = self.whole_inference(img, img_meta, ref[0], gt[0], refgt[0])

        return seg_logit

    def simple_test(self, img, img_meta, ref, gt, refgt):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, ref, gt, refgt)
        return seg_logit


    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
