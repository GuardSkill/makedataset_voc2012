import json
import os
import random
import numpy as np
import copy
import torch.utils.data as data


class CombineDBs(data.Dataset):
    NUM_CLASSES = 21

    def __init__(self, dataloaders, excluded=None):
        self.dataloaders = dataloaders
        self.excluded = excluded
        self.im_ids = []

        # Combine object lists
        for dl in dataloaders:
            for elem in dl.im_ids:
                if elem not in self.im_ids:
                    self.im_ids.append(elem)

        # Exclude validation dataset
        if excluded:
            for dl in excluded:
                for elem in dl.im_ids:
                    if elem in self.im_ids:
                        self.im_ids.remove(elem)

        # Get object pointers
        self.cat_list = []
        self.im_list = []
        new_im_ids = []
        num_images = 0
        for ii, dl in enumerate(dataloaders):
            for jj, curr_im_id in enumerate(dl.im_ids):
                if (curr_im_id in self.im_ids) and (
                        curr_im_id not in new_im_ids):  # in combined daatase and not in new list
                    num_images += 1
                    new_im_ids.append(curr_im_id)
                    self.cat_list.append({'db_ii': ii, 'cat_ii': jj})

        self.im_ids = new_im_ids
        print('Combined number of images: {:d}'.format(num_images))

    def __getitem__(self, index):

        _db_ii = self.cat_list[index]["db_ii"]
        _cat_ii = self.cat_list[index]['cat_ii']
        sample = self.dataloaders[_db_ii].__getitem__(_cat_ii)

        if 'meta' in sample.keys():
            sample['meta']['db'] = str(self.dataloaders[_db_ii])

        return sample

    def __len__(self):
        return len(self.cat_list)

    def __str__(self):
        include_db = [str(db) for db in self.dataloaders]
        exclude_db = [str(db) for db in self.excluded]
        return 'Included datasets:' + str(include_db) + '\n' + 'Excluded datasets:' + str(exclude_db)


def divide_combineddataset(comibne_dataset, test_number=1000):
    train_id_file = 'train_id.txt'
    test_id_file = 'test_id.txt'
    data_len = comibne_dataset.__len__()
    index = list(range(data_len))
    random.shuffle(index)
    train_dataset = copy.deepcopy(comibne_dataset)
    test_dataset = copy.deepcopy(comibne_dataset)
    train_index = index[:(data_len - test_number)]
    test_index = index[(data_len - test_number):]
    new_imgs = []
    new_list = []
    for i in train_index:
        new_imgs.append(comibne_dataset.im_ids[i])
        new_list.append(comibne_dataset.cat_list[i])
    train_dataset.im_ids = new_imgs
    train_dataset.cat_list = new_list
    np.savetxt(train_id_file, train_dataset.im_ids, fmt='%s')
    print('Saved the divided training images: {:d} to {}'.format(train_dataset.__len__(), train_id_file))

    new_imgs = []
    new_list = []
    for i in test_index:
        new_imgs.append(comibne_dataset.im_ids[i])
        new_list.append(comibne_dataset.cat_list[i])
    test_dataset.im_ids = new_imgs
    test_dataset.cat_list = new_list
    np.savetxt(test_id_file, test_dataset.im_ids, fmt='%s')
    print('Saving the divided test images: {:d} to {}'.format(test_dataset.__len__(), test_id_file))

    return train_dataset, test_dataset


def load_combined_dataset(comibne_dataset):
    train_id_file = 'train_id.txt'
    test_id_file = 'test_id.txt'
    if os.path.isfile(train_id_file) & os.path.isfile(test_id_file):
        try:
            train_list = np.genfromtxt(train_id_file, dtype=np.str, encoding='utf-8')
            test_list = np.genfromtxt(test_id_file, dtype=np.str, encoding='utf-8')

        except:
            print("Error: Can't read divide information!!!!")
    else:
        print("Error: Can't read divide file!!!!")
    train_dataset = copy.deepcopy(comibne_dataset)
    test_dataset = copy.deepcopy(comibne_dataset)

    new_im_ids = []
    num_images = 0
    train_dataset.cat_list=[]
    for ii, dl in enumerate(comibne_dataset.dataloaders):
        for jj, curr_im_id in enumerate(dl.im_ids):
            if (curr_im_id in train_list) and (
                    curr_im_id not in new_im_ids):  # in combined daatase and not in new list
                num_images += 1
                new_im_ids.append(curr_im_id)
                train_dataset.cat_list.append({'db_ii': ii, 'cat_ii': jj})
    train_dataset.im_ids = new_im_ids
    print('The divided number of training images: {:d}'.format(num_images))

    new_im_ids = []
    num_images = 0
    test_dataset.cat_list = []
    for ii, dl in enumerate(comibne_dataset.dataloaders):
        for jj, curr_im_id in enumerate(dl.im_ids):
            if (curr_im_id in test_list) and (
                    curr_im_id not in new_im_ids):  # in combined daatase and not in new list
                num_images += 1
                new_im_ids.append(curr_im_id)
                test_dataset.cat_list.append({'db_ii': ii, 'cat_ii': jj})
    test_dataset.im_ids = new_im_ids
    print('The divided number of test images: {:d}'.format(num_images))

    return train_dataset, test_dataset


def make_dataset_val(train_dataset,val_dataset,dir='/home/lulu/Dataset/Voc2012_aug'):
    train_imgs_dir=os.path.join(dir, 'JPEGImages', 'train')
    test_imgs_dir=os.path.join(dir, 'JPEGImages', 'test')
    labels_dir=os.path.join(dir, 'SegmentationClassAug')
    if not os.path.isdir(train_imgs_dir):
        os.makedirs(train_imgs_dir)
    if not os.path.isdir(test_imgs_dir):
        os.makedirs(test_imgs_dir)
    if not os.path.isdir(labels_dir):
        os.makedirs(labels_dir)
    import shutil
    for index in range(train_dataset.im_ids.__len__()):
        _db_ii = train_dataset.cat_list[index]["db_ii"]
        _cat_ii = train_dataset.cat_list[index]['cat_ii']
        sample = train_dataset.dataloaders[_db_ii].images[_cat_ii]
        label = train_dataset.dataloaders[_db_ii].categories[_cat_ii]
        # print(os.path.basename(sample))
        shutil.copyfile(sample, os.path.join(train_imgs_dir,os.path.basename(sample)))
        shutil.copyfile(label, os.path.join(labels_dir,os.path.basename(label)))
    print("Save {} training samples".format(train_dataset.im_ids.__len__()))
    for index in range(val_dataset.images.__len__()):
        sample = val_dataset.images[index]
        label = val_dataset.categories[index]
        shutil.copyfile(sample, os.path.join(test_imgs_dir,os.path.basename(sample)))
        shutil.copyfile(label, os.path.join(labels_dir,os.path.basename(label)))
    print("Save {} test samples".format(val_dataset.images.__len__()))
    return None

def make_dataset(train_dataset,test_dataset,dir='/home/lulu/Dataset/Voc2012_aug'):
    train_imgs_dir=os.path.join(dir, 'JPEGImages', 'train')
    test_imgs_dir=os.path.join(dir, 'JPEGImages', 'test')
    labels_dir=os.path.join(dir, 'SegmentationClassAug')
    if not os.path.isdir(train_imgs_dir):
        os.makedirs(train_imgs_dir)
    if not os.path.isdir(test_imgs_dir):
        os.makedirs(test_imgs_dir)
    if not os.path.isdir(labels_dir):
        os.makedirs(labels_dir)
    import shutil
    for index in range(train_dataset.im_ids.__len__()):
        _db_ii = train_dataset.cat_list[index]["db_ii"]
        _cat_ii = train_dataset.cat_list[index]['cat_ii']
        sample = train_dataset.dataloaders[_db_ii].images[_cat_ii]
        label = train_dataset.dataloaders[_db_ii].categories[_cat_ii]
        # print(os.path.basename(sample))
        shutil.copyfile(sample, os.path.join(train_imgs_dir,os.path.basename(sample)))
        shutil.copyfile(label, os.path.join(labels_dir,os.path.basename(label)))
    print("Save {} training samples".format(train_dataset.im_ids.__len__()))
    for index in range(test_dataset.im_ids.__len__()):
        _db_ii = test_dataset.cat_list[index]["db_ii"]
        _cat_ii = test_dataset.cat_list[index]['cat_ii']
        sample = test_dataset.dataloaders[_db_ii].images[_cat_ii]
        label = test_dataset.dataloaders[_db_ii].categories[_cat_ii]
        shutil.copyfile(sample, os.path.join(test_imgs_dir,os.path.basename(sample)))
        shutil.copyfile(label, os.path.join(labels_dir,os.path.basename(label)))
    print("Save {} test samples".format(test_dataset.im_ids.__len__()))
    return None

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloaders.datasets import pascal, sbd
    from dataloaders import sbd
    import torch
    import numpy as np
    from dataloaders.utils import decode_segmap
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    pascal_voc_val = pascal.VOCSegmentation(args, split='val')
    sbd = sbd.SBDSegmentation(args, split=['train', 'val'])
    pascal_voc_train = pascal.VOCSegmentation(args, split='train')

    dataset = CombineDBs([pascal_voc_train, sbd], excluded=[pascal_voc_val])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break
    plt.show(block=True)
