class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            # return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
            return '/home/lulu/Dataset/VOCdevkit/VOC2012'
            # return 'E:\Dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
        elif dataset == 'sbd':
            # return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
            return '/home/lulu/Dataset/benchmark_RELEASE/'  # folder that contains dataset/.
            # return r'E:\Dataset\benchmark\benchmark_RELEASE'
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
