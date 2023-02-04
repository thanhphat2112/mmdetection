import os
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset
from .xml_style import XMLDataset


@DATASETS.register_module()
class YOLODataset(XMLDataset):
    """YOLO dataset for detection.
    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
        img_subdir (str): Subdir where images are stored. Default: images.
        ann_subdir (str): Subdir where annotations are. Default: labels.
    """

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.
        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []

        self.img_ids = mmcv.list_from_file(ann_file)
        for img_id in self.img_ids:
            # get all annotation in img
            # [["label, x, y, w, h"],
            #  ["label, x, y, w, h"],...]
            ann_file = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.txt')
            assert osp.exists(
                self.ann_file), f"Don't exists path {self.ann_file}"

            ann_infors = mmcv.list_from_file(ann_file)
            # split field ["label x y w h"] => [label, x, y, w, h]
            ann_infors = [infor.split() for infor in ann_infors]

            filename = osp.join(self.img_subdir, f'{img_id}.jpg')
            img_path = osp.join(self.img_prefix, f"images/{img_id}.jpg")
            img = Image.open(img_path)
            width, height = img.size

            bboxes = []
            labels = []
            # check annotation in image
            if len(ann_infors):
                for ann_infor in ann_infors:
                    labels.append(int(ann_infor[0]))
                    
                    bbox = []
                    for idx in range(len(ann_infor[1:])):
                        if idx % 2:
                            ann = float(ann_infor[idx + 1]) * width
                        else:
                            ann = float(ann_infor[idx + 1]) + height
                        bbox.append(ann)
                    bboxes.append(bbox)
                    # bboxes.append([float(ann) for ann in ann_infor[1:]])
            else:
                labels.append(1) # labels = 1 is background
                bboxes.append([0, 0, 0, 0])


            data_infos.append(
                dict(filename=filename,  # mention hardcode
                     width=width,
                     height=height,
                     ann=dict(
                         bboxes=np.array(bboxes).astype(np.float32),  # key
                         labels=np.array(labels).astype(
                             np.int64)
                     )
                )
            )


        return data_infos

    def get_ann_info(self, idx):
        """Get annotation from txt file by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        """Get category ids by index.
        Args:
            idx (int): Index of data.
        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels']
    
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                if len(img_info['ann']['bboxes']) != 0:
                    continue
            else:
                valid_inds.append(i)
        
        return valid_inds

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            # print(data)
            return data