# encoding: utf-8
"""Quick mixed human detector config: CrowdHuman + MOT17 + MOT20 -> MOT20 val subset."""
import os
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.train_ann = os.getenv("MIX_DET_TRAIN_ANN", "train_quick.json")
        self.val_ann = os.getenv("MIX_DET_VAL_ANN", "val_mot20_quick.json")
        input_h = int(os.getenv("MIX_DET_INPUT_H", "896"))
        input_w = int(os.getenv("MIX_DET_INPUT_W", "1600"))
        test_h = int(os.getenv("MIX_DET_TEST_H", str(input_h)))
        test_w = int(os.getenv("MIX_DET_TEST_W", str(input_w)))
        self.input_size = (input_h, input_w)
        self.test_size = (test_h, test_w)
        self.random_size = (max(10, input_h // 32), max(10, input_w // 32))
        self.max_epoch = int(os.getenv("MIX_DET_MAX_EPOCH", "3"))
        self.print_interval = 20
        self.eval_interval = max(1, int(os.getenv("MIX_DET_EVAL_INTERVAL", "1")))
        self.no_aug_epochs = min(1, self.max_epoch)
        self.basic_lr_per_img = float(os.getenv("MIX_DET_LR_PER_IMG", "0.00025")) / 64.0
        self.warmup_epochs = 1
        self.test_conf = float(os.getenv("MIX_DET_TEST_CONF", "0.001"))
        self.nmsthre = float(os.getenv("MIX_DET_NMS", "0.7"))
        self.data_num_workers = int(os.getenv("MIX_DET_WORKERS", "4"))
        self.max_labels = int(os.getenv("MIX_DET_MAX_LABELS", "1600"))

    def _maybe_truncate(self, dataset, env_name):
        n = int(os.getenv(env_name, "0") or "0")
        if n > 0:
            dataset.ids = dataset.ids[:n]
            dataset.annotations = dataset.annotations[:n]
        return dataset

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import MOTDataset, TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler, MosaicDetection
        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "MIX_CH_MOT17_MOT20"),
            json_file=self.train_ann,
            name='',
            img_size=self.input_size,
            preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_labels=self.max_labels),
        )
        dataset = self._maybe_truncate(dataset, "MIX_DET_SMOKE_TRAIN_IMAGES")
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_labels=self.max_labels),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )
        self.dataset = dataset
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, input_dimension=self.input_size, mosaic=not no_aug)
        kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "batch_sampler": batch_sampler}
        return DataLoader(self.dataset, **kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform
        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "MIX_CH_MOT17_MOT20"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='',
            preproc=ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
        valdataset = self._maybe_truncate(valdataset, "MIX_DET_SMOKE_VAL_IMAGES")
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler, "batch_size": batch_size}
        return torch.utils.data.DataLoader(valdataset, **kwargs)

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators.coco_evaluator import COCOEvaluator
        return COCOEvaluator(self.get_eval_loader(batch_size, is_distributed, testdev=testdev), self.test_size, self.test_conf, self.nmsthre, self.num_classes, testdev=testdev)
