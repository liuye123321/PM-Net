import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES
import os

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        results['ref'] = []
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        # input
        filename = results['img_info']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        #ref
        for i in range(len(results['ref_info'])):
            refname = results['ref_info'][i]
            refname = self.file_client.get(refname)
            refimg = mmcv.imfrombytes(
                refname, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                refimg = refimg.astype(np.float32)
            results['ref'].append(refimg)
        # gt
        gtname = results['gt_info']
        gt_bytes = self.file_client.get(gtname)
        gt = mmcv.imfrombytes(
            gt_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            gt = gt.astype(np.float32)
        #ref_gt
        if 'refgt_info' in results:
            results['refgt'] = []
            for i in range(len(results['refgt_info'])):
                refname = results['refgt_info'][i]
                if os.path.exists(refname):
                    refname = self.file_client.get(refname)
                    refimg = mmcv.imfrombytes(
                        refname, flag=self.color_type, backend=self.imdecode_backend)
                    if self.to_float32:
                        refimg = refimg.astype(np.float32)
                    results['refgt'].append(refimg)
        # depth
        # depthname = results['depth_info']
        # if depthname.split('/')[6] == 'Train':
        #     depth_bytes = self.file_client.get(depthname)
        #     depth = mmcv.imfrombytes(
        #         depth_bytes, flag='grayscale', backend=self.imdecode_backend)
        #     depth = depth/255
        #     if self.to_float32:
        #         depth = depth.astype(np.float32)
        #     results['depth'] = depth

        results['filename'] = filename
        results['img'] = img
        results['gt'] = gt

        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


