import os.path as osp
from functools import reduce
import os
import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset
import torch
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation.

    An example of file structure is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 ref_frame,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.ref_frame = ref_frame
        self.size = 460
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.img_list = []
        self.reflist_list = []
        self.gt_list = []
        self.depth_list = []
        self.reflist_gt_list = []
        self.mode = '.JPG'
        self.number = 5

        # load annotations
        if self.test_mode==0:
            self.img_infos = self.load_annotations(img_dir, 'train', self.ref_frame)
        else:
            self.img_infos = self.load_annotations(img_dir, 'test', self.ref_frame)


    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)
    def load_annotations(self, img_dir, image_set, ref_frame):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        # image_set = 'train'
        listfile = os.path.join(img_dir, "list", "{}_gt.txt".format(image_set))

        if image_set == 'train':
            with open(listfile) as f:
                for line in f:
                    line = line.strip()
                    l = line.split(" ")
                    index = int(os.path.basename(l[0]).split('.')[0])
                    dir_path = os.path.dirname(l[0])
                    p = []
                    exist = True
                    for i in range(ref_frame):
                        p.append(os.path.join(self.img_dir, dir_path,str(int(os.path.basename(l[0]).split('.')[0]) + i + 1).zfill(self.number) + self.mode))
                        exist = os.path.exists(p[i]) and exist
                    # if index > ref_frame and exist:
                    if index >= ref_frame and exist:
                        ref_list = []
                        ref_gt_list = []
                        self.img_list.append(os.path.join(self.img_dir, l[0]))   # l[0][1:]  get rid of the first '/' so as for os.path.join
                        for j in range(ref_frame):
                            ref = os.path.join(self.img_dir, dir_path, str(int(os.path.basename(l[0]).split('.')[0]) - (ref_frame - j)).zfill(self.number)+self.mode)
                            if ref == os.path.join(self.img_dir, l[0]):
                                continue
                            ref_list.append(ref)
                        for k in range(ref_frame):
                            ref_gt = os.path.join(self.img_dir, l[1][0:-9], str(int(os.path.basename(l[1]).split('.')[0]) - (ref_frame - k)).zfill(self.number)+self.mode)
                            if ref_gt == os.path.join(self.img_dir, l[1]):
                                continue
                            ref_gt_list.append(ref_gt)
                        self.reflist_list.append(ref_list)
                        self.reflist_gt_list.append(ref_gt_list)
                        self.gt_list.append(os.path.join(self.img_dir, l[1]))
                        self.depth_list.append(os.path.join(self.img_dir, l[1]).replace('gt', 'depth').replace('JPG', 'png'))
                    else:
                        continue
        if image_set == 'test':
            with open(listfile) as f:
                for line in f:
                    line = line.strip()
                    l = line.split(" ")
                    index = int(os.path.basename(l[0]).split('.')[0])
                    dir_path = os.path.dirname(l[0])
                    p = []
                    exist = True
                    for i in range(ref_frame):
                        p.append(os.path.join(self.img_dir, dir_path,
                                              str(int(os.path.basename(l[0]).split('.')[0]) + i + 1).zfill(self.number) + self.mode))
                        exist = os.path.exists(p[i]) and exist
                    # if index > ref_frame and exist:
                    if index >= ref_frame and exist:  ##indoor
                        ref_list = []
                        ref_gt_list = []
                        self.img_list.append(os.path.join(self.img_dir, l[0]))  # l[0][1:]  get rid of the first '/' so as for os.path.join
                        for j in range(ref_frame):
                            ref = os.path.join(self.img_dir, dir_path,
                                               str(int(os.path.basename(l[0]).split('.')[0]) - (ref_frame - j)).zfill(
                                                   self.number) + self.mode)
                            if ref == os.path.join(self.img_dir, l[0]):
                                continue
                            ref_list.append(ref)

                        for k in range(ref_frame):
                            ref_gt = os.path.join(self.img_dir, l[1][0:-9], str(int(os.path.basename(l[1]).split('.')[0]) - (ref_frame - k)).zfill(self.number)+self.mode).replace('gt', 'test_result')
                            if ref_gt == os.path.join(self.img_dir, l[1]):
                                continue
                            ref_gt_list.append(ref_gt)
                        self.reflist_list.append(ref_list)
                        self.reflist_gt_list.append(ref_gt_list)
                        self.gt_list.append(os.path.join(self.img_dir, l[1]))
                        self.depth_list.append(os.path.join(self.img_dir, l[1]).replace('gt', 'depth').replace('JPG', 'png'))

                    else:
                        continue
        print_log(f'Loaded {len(self.img_list)} images', logger=get_root_logger())
        return self.img_list

    def get_ref_info(self, idx):
        return self.reflist_list[idx]

    def get_depth_info(self, idx):
        return self.depth_list[idx]

    def get_clean_img_info(self, idx):
        return self.gt_list[idx]

    def get_refgt_img_info(self, idx):
        return self.reflist_gt_list[idx]

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        # results['scale']=(self.size, self.size)
        results['seg_fields'] = []
        # results['img_prefix'] = self.img_list
        # results['seg_prefix'] = self.segLabel_list
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.img_infos[idx]
        ref_info = self.get_ref_info(idx)
        depth_info = self.get_depth_info(idx)
        clean_img_info = self.get_clean_img_info(idx)
        refgt_info = self.get_refgt_img_info(idx)
        results = dict(img_info=img_info, ref_info=ref_info, depth_info=depth_info, gt_info=clean_img_info, refgt_info=refgt_info)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        img_info = self.img_infos[idx]
        ref_info = self.get_ref_info(idx)
        depth_info = self.get_depth_info(idx)
        clean_img_info = self.get_clean_img_info(idx)
        refgt_info = self.get_refgt_img_info(idx)
        results = dict(img_info=img_info, ref_info=ref_info, depth_info=depth_info, gt_info=clean_img_info, refgt_info=refgt_info)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass



