import os
import sys
import json
import pickle
import random

import torch
# from torch.utils.tensorboard.summary import image
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import torchvision.transforms.functional as F
from torchvision.transforms import  RandomCrop as RandomCrop

import cv2
import albumentations
from PIL import Image
class Random_Dropout:
    def __init__(self,probility):
        self.probility=probility
    def __call__(self,image):
        if random.uniform(0,1) > self.probility:
            return image
        w, h = image.size
        return Image.fromarray(cv2.cvtColor(albumentations.CoarseDropout(max_holes=2,max_height=h/2, max_width=w/2,p=self.probility)(image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))['image'],cv2.COLOR_BGR2RGB))

class CLAHE:
    def __init__(self,prob):
        self.probility = prob
    def __call__(self,image):
        if random.uniform(0,1) > self.probility:
            return image
        return Image.fromarray(cv2.cvtColor(albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=self.probility)(image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))['image'],cv2.COLOR_BGR2RGB))

class Random_blur:
    def __init__(self,prob):
        self.probility = prob
    def __call__(self,image):
        if random.uniform(0,1) > self.probility:
            return image
        return Image.fromarray(cv2.cvtColor(albumentations.Blur(blur_limit = 5,always_apply = False,p = self.probility)(image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))['image'],cv2.COLOR_BGR2RGB))

class random_paste:
    def __init__(self,image_size,prob):
        self.prob = prob
        g = os.walk(r'background_image')
        self.background_image_path = []
        self.image_size = image_size
        for path,dir_list,file_list in g:
            for file_name in file_list:  
                self.background_image_path.append(os.path.join(path,file_name))
    def __call__(self,image):
        if random.uniform(0,1) > self.prob:
            return image
#         select_image_path = background_image_path[random.randint(0,len(self.background_image_path)-1)]
        background_image = Image.open(self.background_image_path[random.randint(0,len(self.background_image_path)-1)])
        background_image = Image.fromarray(np.array(background_image)).resize((self.image_size,self.image_size),0)
        w, h = image.size
        max_wh = np.max([w, h])
        ratial = (max_wh/self.image_size) * random.uniform(1,2)
        new_w = int(w/ratial)
        new_h = int(h/ratial)
        image = image.resize((new_w,new_h),0)
        new_x = int(random.uniform(0,self.image_size - new_w))
        new_y = int(random.uniform(0,self.image_size - new_h))
        background_image.paste(image, (new_x,new_y))
        return background_image

class letterbox:
    def __init__(self,color):
        self.color = color

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        self.color2 = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        if self.color == 0:
            self.color2 = (0,0,0)
        return F.pad(image, padding, self.color2, 'constant')

class cropping_image_ramdomly:
    def __init__(self,size,p):
        self.size = size
        self.prob = p
        self.cropping_fucntion =  RandomCrop(size=(self.size,self.size))
    def __call__(self,image):
        # p = 30
        # size = img_size[num_model][0]
        if random.randint(0,99) <= self.prob:
            # crop_func = RandomCrop(size=(self.size,self.size))
            return self.cropping_fucntion(image)
        else:
            return image

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * torch.nn.functional.nll_loss(log_preds, target, reduction=self.reduction)
def train_siamese_network_one_epoch(model1,model2,optimizer,dataloader,device,epoch,label_smothing_eps = 0.1):
    model1.eval()
    model2.efficientnet_module.load_state_dict(model1.state_dict(),strict=False)
    for name,para in model2.efficientnet_module.named_parameters():
        if "head" not in name:
            para.requires_grad_(False)
    model2.train()
    if label_smothing_eps != 0:
        loss_function = LabelSmoothingCrossEntropy(eps=label_smothing_eps)
    else:
        loss_function=torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    dataloader = tqdm(dataloader)
    for step,data in enumerate(dataloader):
        images,labels = data
        sample_num += images.shape[0]
        pred = model2(images.to(device))
        pred_classes = torch.max(pred,dim=1)[1]
        accu_num += torch.eq(pred_classes,labels.to(device)).sum()
        loss = loss_function(pred,labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        dataloader.desc = "[train siamese_network epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        optimizer.step()
        optimizer.zero_grad()
    model2.eval()
    return accu_loss.item() / (step + 1),accu_num.item() / sample_num

def val_the_dataset(model2,dataloader,device):
    model2.eval()
    prob_of_classes_list = []
    prediction_cls_list = []
    lables_list = []
    output_list=[]
    for step,data in enumerate(dataloader):
        images,labels = data
        with torch.no_grad():
            output = torch.squeeze(model2(images.to(device)))
            prediction = torch.softmax(output,dim=1)
            prob_of_classes = torch.max(prediction,dim=1)[0] * 100
            prediction_cls = torch.max(prediction,dim=1)[1]
        prob_of_classes_list += prob_of_classes.cpu().numpy().tolist()
        prediction_cls_list += prediction_cls.cpu().numpy().tolist()
        lables_list += labels.cpu().numpy().tolist()
    output_list.append([prediction_cls_list[i],prob_of_classes_list[i],lables_list[i]] for i in range(len(lables_list)))
    return np.mean(prob_of_classes_list),np.var(prob_of_classes_list),np.std(prob_of_classes_list),np.sum(list(map(lambda funct:funct<=80, prob_of_classes_list))),output_list

class vector_loss(torch.nn.Module):
    def __init__(self,th=20):
        super(vector_loss, self).__init__()
        self.thresh = th
#         self.eps = eps
#         self.reduction = reduction

    def forward(self, output, target):
        if torch.equal(target[0],target[1]):
            return torch.nn.functional.pdist(output,p=2)
        else:
            dis = torch.nn.functional.pdist(output,p=2)
            if self.thresh - dis <= 0:
                return dis * (10 ** -10)
            else:
                return self.thresh - dis
    
def train_one_epoch_vector_loss(model,optimizer, data_loader,data_loader_shuffle, device, epoch,label_smothing_eps = 0.1):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    feature_map = {}
    def forward_hook(module, inp, outp):
        feature_map['feature'] = outp
    extract_feature_layers = list(model.children())[-1][3]
    extract_feature_layers.register_forward_hook(forward_hook)
    if label_smothing_eps != 0:
        loss_func = vector_loss()
    else:
        loss_func = vector_loss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    pos_sample = 0
    nag_sample = 0
    data_loader = tqdm(data_loader)
    data_loader_shuffle = tqdm(data_loader_shuffle)
    for step, data in enumerate(data_loader):
        images, labels = data
#         print(np.array(labels.size()) != 2)
        if np.array(labels.size()) != 2:
            break
        # print(images.size())
        
        if torch.equal(labels[0],labels[1]):
            pos_sample+=1
        else:
            nag_sample += 1
        sample_num += images.shape[0]
        pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_func(feature_map['feature'],labels.to(device))
        # loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        
        data_loader.desc = "[train epoch {}] loss: {:.3f}, pos_sample: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               pos_sample/(step + 1))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
        
    for step, data in enumerate(data_loader_shuffle):
        images, labels = data
        if np.array(labels.size()) != 2:
            break
        # print(images.size())
        if torch.equal(labels[0],labels[1]):
            pos_sample+=1
        else:
            nag_sample += 1
        sample_num += images.shape[0]
        pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_func(feature_map['feature'],labels.to(device))
        # loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        
        data_loader_shuffle.desc = "[train shuffle epoch {}] loss: {:.3f}, pos_sample: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               pos_sample/(step + 1))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
        
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def train_one_epoch(model,optimizer, data_loader, device, epoch,label_smothing_eps = 0.1):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    if label_smothing_eps != 0:
        loss_func = LabelSmoothingCrossEntropy(eps=label_smothing_eps)
    else:
        loss_func = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        # print(images.size())
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_func(pred,labels.to(device))
        # loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
def compare_feature_vectors(total_feature_list,total_label_list):
    total_featuremap_in_classes = []
    mean_vector_dis = []
    for i in set(total_label_list):
        feature_map_in_cls = []
        for j in [index for index,x in enumerate(total_label_list) if x == i]:
            feature_map_in_cls.append(total_feature_list[j])
        total_featuremap_in_classes.append(feature_map_in_cls)
    for item in total_featuremap_in_classes:
        dis = []
        for index1 in range(len(item)-1):
            for index2 in range(index1+1,len(item)):
                dis.append(np.linalg.norm(np.array(item[index1]) - np.array(item[index2])))
        mean_vector_dis.append(np.mean(dis))
    return mean_vector_dis
        

@torch.no_grad()
def evaluate_feature(model, data_loader, device, epoch):
#     loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    
    feature_map = {}
    def forward_hook(module, inp, outp):
        feature_map['feature'] = outp
    extract_feature_layers = list(model.children())[-1][3]
    extract_feature_layers.register_forward_hook(forward_hook)
    
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
    total_feature_list = []
    total_label_list = []
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
#         sample_num += images.shape[0]
        model(images.to(device))
        feature = feature_map['feature'].cpu().numpy().tolist()
        total_feature_list += feature
        total_label_list += labels.cpu().numpy().tolist()
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#         loss = loss_function(pred, labels.to(device))
#         accu_loss += los
        data_loader.desc = "[valid epoch {}] step: {:.3f}".format(epoch,step)
#     compare_feature_vectors()
    return compare_feature_vectors(total_feature_list,total_label_list)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def small_obj_dataset_reading(train_csv_file,val_csv_file):
    import csv
    train_csv_file_reader = csv.reader(open(train_csv_file,'r'))
    val_csv_file_reader = csv.reader(open(val_csv_file,'r'))
    train_image_path_list = []
    train_image_class_list = []
    val_image_path_list = []
    val_image_class_list = []
    train_every_class_count = {}
    root_path = os.path.split(train_csv_file)[0]
    
    for item in train_csv_file_reader:
        if train_csv_file_reader.line_num == 1:
            continue
        if len(item[0].split()) == 1:
            train_image_path_list.append(os.path.join(root_path,item[0]))
        else:
#             print(item[0])
            train_image_path_list.append(os.path.join(root_path,item[0].split()[0])+' '+os.path.join(root_path,item[0].split()[1]))
        train_image_class_list.append(int(item[1])-1)
    
    for item in val_csv_file_reader:
        if val_csv_file_reader.line_num == 1:
            continue
        val_image_path_list.append(os.path.join(root_path,item[0]))
        val_image_class_list.append(int(item[1])-1)
    from collections import Counter
    intotal_class = train_image_class_list + val_image_class_list
    # intotal_class.append(val_image_class_list)
    # print(intotal_class)
    train_every_class_count = Counter(intotal_class)
    print(train_every_class_count)
    return train_image_path_list, train_image_class_list, val_image_path_list, val_image_class_list
    
# def plot_figure(nums,name):
#     fig = plt.figure(figsize = (10,10))       #figsize是图片的大小`
#     ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
#     pl.plot(nums,'g-',label=u'Dense_Unet(block layer=5)')
#     plt.title(name)
#     fig.savefig(name + '.png')
#     plt.close(fig)

def plot_figure(nums,name):
    x1,y1= [],[]
    for i,j in enumerate(nums):
      x1.append(i)
      y1.append(j)
    plt.plot(x1, y1,  color='r',markerfacecolor='blue',marker='.')  
    for a, b in zip(x1,y1):  
      plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)  
    plt.title(name)
    plt.savefig(name + '.png')
    plt.close()