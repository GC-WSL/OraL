import torch.nn as nn
import math
import numpy as np
import copy

# padding函数，先在数据集的上下分别加patch/2行数据(最近的那一行重复加)，再在左右分别加patch/2列数据
def padding(dataset, patch_size):
    # X Y C
    padding_dataset = copy.deepcopy(dataset)
    one_side_patch_size = patch_size//2
    up_row = np.repeat([padding_dataset[0, :, :]], one_side_patch_size, axis=0)
    down_row = np.repeat([padding_dataset[-1, :, :]], one_side_patch_size, axis=0)
    padding_dataset = np.concatenate((up_row, padding_dataset, down_row), axis=0)
    # 按列重复完要转置一下，不然拼接不上
    left_column = np.repeat([padding_dataset[:, 0, :]], one_side_patch_size, axis=0).transpose((1, 0, 2))
    right_column = np.repeat([padding_dataset[:, -1, :]], one_side_patch_size, axis=0).transpose((1, 0, 2))
    padding_dataset = np.concatenate((left_column, padding_dataset, right_column), axis=1)
    return padding_dataset

# 制作训练集切片的函数
def make_train_patch(dataset, coordinate, patch_size):
    current_class_train_sample = list()
    # 得到一个个patch_size*patch_size*bands的patch
    for row in coordinate:
        current_class_train_sample.append(dataset[row[0]:row[0]+patch_size, row[1]:row[1]+patch_size, :])
    return np.array(current_class_train_sample)

# 制作训练集的函数
def make_train_set_diff(dataset1, dataset2, idx, img):
    H, W, C = img.shape
    label_coordinate = list()
    for i in idx:  # np.max(ground_truth)=1表示有多少类别，此for循环时获得label=0和label=1对应的坐标，存储在label_coordinate中
        row = int(np.ceil((i+1)/W)-1)        # 列
        col = int(i-row*W)        # 行
        label_coordinate.append(np.array([row, col]).T)  # label_coordinate中记录的是ground_truth == i的坐标。
    # 开始在数据集里面取对应坐标的数据，注意数据集是padding过的
    train_sample1, train_sample2 = np.zeros((1, 7, 7, dataset1.shape[2])), np.zeros((1, 7, 7, dataset1.shape[2]))
    num_train_sample = np.array(label_coordinate).shape[0]
    train_sample1 = np.concatenate((train_sample1, make_train_patch(dataset1, label_coordinate, 7)),axis=0)
    train_sample2 = np.concatenate((train_sample2, make_train_patch(dataset2, label_coordinate, 7)),axis=0)
    train_sample1, train_sample2 = train_sample1[1:], train_sample2[1:]

    # 数据增强，竖直、水平、对角线翻转数据集，使其变为原来的4倍
    vertical1 = np.flip(train_sample1, 1)
    horizontal1 = np.flip(train_sample1, 2)
    crosswise1 = np.flip(horizontal1, 2)

    vertical2 = np.flip(train_sample2, 1)
    horizontal2 = np.flip(train_sample2, 2)
    crosswise2 = np.flip(horizontal2, 2)
    train_sample1 = np.concatenate((vertical1, horizontal1, crosswise1, train_sample1), axis=0)
    train_sample2 = np.concatenate((vertical2, horizontal2, crosswise2, train_sample2), axis=0)
    
    # [N,H,W,C]转[N,C,H,W]，不然Torch读取会出问题
    train_sample1 = train_sample1.transpose((0, 3, 1, 2))
    train_sample2 = train_sample2.transpose((0, 3, 1, 2))
    idx = np.random.permutation(train_sample1.shape[0])  # 对train_sample.shape[0]打乱
    train_sample1 = train_sample1[idx]
    train_sample2 = train_sample2[idx]
    return train_sample1, train_sample2



# 制作训练集的函数
def make_train_set_nl(dataset1, dataset2, clean_index, clean_gt, img):
    H, W, C = img.shape
    label_coordinate = list()
    for i in clean_index:  # np.max(ground_truth)=1表示有多少类别，此for循环时获得label=0和label=1对应的坐标，存储在label_coordinate中
        row = int(np.ceil((i+1)/W)-1)        # 列
        col = int(i-row*W)        # 行
        label_coordinate.append(np.array([row, col]).T)  # label_coordinate中记录的是ground_truth == i的坐标。
    # 开始在数据集里面取对应坐标的数据，注意数据集是padding过的
    train_sample1, train_sample2, train_label = np.zeros((1, 7, 7, dataset1.shape[2])), np.zeros((1, 7, 7, dataset1.shape[2])), np.zeros(1)
    num_train_sample = np.array(label_coordinate).shape[0]
    train_sample1 = np.concatenate((train_sample1, make_train_patch(dataset1, label_coordinate, 7)),axis=0)
    train_sample2 = np.concatenate((train_sample2, make_train_patch(dataset2, label_coordinate, 7)),axis=0)
    train_label = np.concatenate((train_label, np.array(clean_gt)), axis=0)
    train_sample1, train_sample2, train_label = train_sample1[1:], train_sample2[1:], train_label[1:]

    # 数据增强，竖直、水平、对角线翻转数据集，使其变为原来的4倍
    vertical1 = np.flip(train_sample1, 1)
    horizontal1 = np.flip(train_sample1, 2)
    crosswise1 = np.flip(horizontal1, 2)

    vertical2 = np.flip(train_sample2, 1)
    horizontal2 = np.flip(train_sample2, 2)
    crosswise2 = np.flip(horizontal2, 2)
    train_sample1 = np.concatenate((vertical1, horizontal1, crosswise1, train_sample1), axis=0)
    train_sample2 = np.concatenate((vertical2, horizontal2, crosswise2, train_sample2), axis=0)
    train_label = np.concatenate((train_label, train_label, train_label, train_label))
    
    # [N,H,W,C]转[N,C,H,W]，不然Torch读取会出问题
    train_sample1 = train_sample1.transpose((0, 3, 1, 2))
    train_sample2 = train_sample2.transpose((0, 3, 1, 2))
    idx = np.random.permutation(train_sample1.shape[0])  # 对train_sample.shape[0]打乱
    train_sample1 = train_sample1[idx]
    train_sample2 = train_sample2[idx]
    train_label = train_label[idx]
    return train_sample1, train_sample2, train_label
# 制作训练集的函数
def make_train_set(dataset1, dataset2, ground_truth):
    Fraction = 0.05 # 训练像素占总像素的比例
    # 搜寻每类标签的坐标,随机打乱，取每类前70%个作为训练集
    # patch_size是输入网络patch的尺寸
    print("总的像素数：" + str(round(ground_truth.shape[0]*ground_truth.shape[1])))
    patch_size = dataset1.shape[0] - ground_truth.shape[0] + 1
    pos_num = int(np.sum(ground_truth == 1) * Fraction)  # 改变的像素数，总的变化样本的70%
    neg_num = int(np.sum(ground_truth == 0) * Fraction)  # 未改变的像素数，总的没有变化的70%
    pos_num1 = int(np.sum(ground_truth == 1))
    pos_num2 = int(np.sum(ground_truth == 0))
    print("改变的像素数： " + str(round(pos_num1)))
    print("未改变的像素数： " + str(round(pos_num2)))
    label_coordinate = list()
    for i in range(np.max(ground_truth) + 1):  # np.max(ground_truth)=1表示有多少类别，此for循环时获得label=0和label=1对应的坐标，存储在label_coordinate中
        label_coordinate.append(np.array(np.where(ground_truth == i)).T)  # label_coordinate中记录的是ground_truth == i的坐标。

    train_sample_coordinate = list()
    test_sample_coordinare = list()
    # 此for循环是将用于训练的像素坐标放入train_sample_coordinate中
    for i in range(0, len(label_coordinate)):
        np.random.shuffle(label_coordinate[i])
        pos_num = int(np.sum(ground_truth == i) * Fraction)  # 改变的像素数，总的变化样本的70%
        print("用于训练 pixel: " + str(round(pos_num)),"比例: " + str(round(Fraction,4)),"change: " + str(round(i)))
        Fraction = Fraction 
        train_sample_coordinate.append(label_coordinate[i][0:pos_num, :])  # 得到用于训练的像素的坐标
        test_sample_coordinare.append(label_coordinate[i][pos_num:np.sum(ground_truth == i), :])  # 得到用于test像素的坐标

    # 开始在数据集里面取对应坐标的数据，注意数据集是padding过的
    train_sample1, train_sample2, train_label = np.zeros((1, patch_size, patch_size, dataset1.shape[2])), np.zeros((1, patch_size, patch_size, dataset1.shape[2])), np.zeros(1)
    for i in range(len(train_sample_coordinate)):
        num_train_sample = np.array(train_sample_coordinate[i]).shape[0]
        train_sample1 = np.concatenate((train_sample1, make_train_patch(dataset1, train_sample_coordinate[i], 
                                                                      patch_size)),axis=0)
        train_sample2 = np.concatenate((train_sample2, make_train_patch(dataset2, train_sample_coordinate[i], 
                                                                      patch_size)),axis=0)
        train_label = np.concatenate((train_label, np.ones(num_train_sample) * i), axis=0)
        # train_label = np.concatenate((train_label, np.ones(sample_num_per_class) * i), axis=0)
    train_sample1, train_sample2, train_label = train_sample1[1:], train_sample2[1:], train_label[1:]

    # 开始在数据集里面取对应坐标的数据制作testdata，注意数据集是padding过的
    
    # test_sample1, test_sample2, test_label = np.zeros((1, patch_size, patch_size, dataset1.shape[2])), np.zeros((1, patch_size, patch_size, dataset1.shape[2])), np.zeros(1)
    # for i in range(len(test_sample_coordinare)):
    #     num_test_sample = np.array(test_sample_coordinare[i]).shape[0]
    #     test_sample1 = np.concatenate((test_sample1, make_train_patch(dataset1, test_sample_coordinare[i], patch_size)),
    #                                  axis=0)
    #     test_sample2 = np.concatenate((test_sample2, make_train_patch(dataset2, test_sample_coordinare[i], patch_size)),
    #                                  axis=0)
    #     test_label = np.concatenate((test_label, np.ones(num_test_sample) * i), axis=0)
    # test_sample1, test_sample2, test_label = test_sample1[1:], test_sample2[1:], test_label[1:]
    

    # 数据增强，竖直、水平、对角线翻转数据集，使其变为原来的4倍
    # d = np.rot90(a, 1,(2,3))  # 对a在第二维和第三维进行旋转，1表示逆时针旋转90°。
    #d = np.flip(a,1)  表示沿第一维i而进行对称
    vertical1 = np.flip(train_sample1, 1)
    horizontal1 = np.flip(train_sample1, 2)
    crosswise1 = np.flip(horizontal1, 2)

    vertical2 = np.flip(train_sample2, 1)
    horizontal2 = np.flip(train_sample2, 2)
    crosswise2 = np.flip(horizontal2, 2)
    # train_sample = np.concatenate((vertical, horizontal, crosswise, train_sample,xuanzhuan90,xuanzhuan180,xuanzhuan270), axis=0)
    # train_label = np.concatenate((train_label, train_label, train_label, train_label,train_label,train_label,train_label))
    
    train_sample1 = np.concatenate((vertical1, horizontal1, crosswise1, train_sample1), axis=0)
    train_sample2 = np.concatenate((vertical2, horizontal2, crosswise2, train_sample2), axis=0)
    train_label = np.concatenate((train_label, train_label, train_label, train_label))
    
    # [N,H,W,C]转[N,C,H,W]，不然Torch读取会出问题
    train_sample1 = train_sample1.transpose((0, 3, 1, 2))
    train_sample2 = train_sample2.transpose((0, 3, 1, 2))
    idx = np.random.permutation(train_sample1.shape[0])  # 对train_sample.shape[0]打乱
    train_sample1 = train_sample1[idx]
    train_sample2 = train_sample2[idx]
    train_label = train_label[idx]

    # # 对测试集进行翻转 [N,H,W,C]转[N,C,H,W]，不然Torch读取会出问题
    # test_sample1 = test_sample1.transpose((0, 3, 1, 2))
    # test_sample2 = test_sample2.transpose((0, 3, 1, 2))
    return train_sample1, train_sample2, train_label, test_sample_coordinare ,train_sample_coordinate#, test_sample1, test_sample2, test_label



# 制作测试集的函数，把整张图都做成测试集，要是不想这么做的话就用上文的办法做就可以了
def get_set_row(dataset1, dataset2, ground_truth, index):
    patch_size = dataset1.shape[0] - ground_truth.shape[0] + 1
    test_sample1 = list()
    test_sample2 = list()
    test_label = list()
    for j in range(ground_truth.shape[1]):
        test_sample1.append(dataset1[index:index+patch_size, j:j+patch_size])
        test_sample2.append(dataset2[index:index+patch_size, j:j+patch_size])
        test_label.append(ground_truth[index, j] - 1)
    test_sample1, test_sample2, test_label = np.array(test_sample1), np.array(test_sample2), np.array(test_label)
    test_sample1 = test_sample1.transpose((0, 3, 1, 2))
    test_sample2 = test_sample2.transpose((0, 3, 1, 2))
    return test_sample1, test_sample2, test_label


def get_sample_patch(img_3d, idx_sample, patch_size):
    img_3d = img_3d.transpose(2, 0, 1)
    # img_3d : img_channel, img_height, img_width
    k = patch_size//2
    img_channel, img_height, img_width = img_3d.shape     # 162,307,307
    patch_set = []
    pixel_set = []

    expand_img_3d = img_3d     # 162,307,307

    temp1 = img_3d[:, :, :k]
    temp2 = img_3d[:, :, -k:]
    expand_img_3d = np.concatenate((temp1, expand_img_3d), axis=2)
    expand_img_3d = np.concatenate((expand_img_3d, temp2), axis=2)

    temp3 = img_3d[:, :k, :]
    temp4 = img_3d[:, -k:, :]
    temp3 = np.concatenate((temp1[:, :k, :],temp3), axis=2)
    temp3 = np.concatenate((temp3, temp2[:, :k, :]), axis=2)
    temp4 = np.concatenate((temp1[:, -k:, :],temp4), axis=2)
    temp4 = np.concatenate((temp4, temp2[:, -k:, :]), axis=2)
    expand_img_3d = np.concatenate((temp3, expand_img_3d), axis=1)
    expand_img_3d = np.concatenate((expand_img_3d, temp4), axis=1)
    for idx in idx_sample:
        # row = int(np.ceil((idx+1)/img_height)-1)        # 列
        # col = int(idx-row*img_height)        # 行
        col = int(np.ceil((idx+1)/img_width)-1)  # 行
        row = int(idx-col*img_width)  # 列
        patch_set.append(expand_img_3d[:, col:(col+7), row:(row+7)])
        pixel_set.append(expand_img_3d[:, (col+3), (row+3)])

    patch_set = np.asarray(patch_set)
    pixel_set = np.asarray(pixel_set)

    return patch_set, pixel_set, expand_img_3d



def initNetParams_v2(net):
    # Init net parameters
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # print('Init FC')
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # print('Init Conv1D')
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.weight, 1)
        #     nn.init.constant_(m.bias, 0)
        #     print('Init LayerNorm')

def adjust_learning_rate(optimizer, init_lr, epoch, total_epoch):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epoch))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def two_cls_access(reference,result):
    # for Hermiston dataset
    # reference:change_value=1;unchange_value=0
    # result: predicted map:change_value=1;unchange_value=0
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)
    label_0 = np.where(reference == 0)
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)
    predict_1 = np.where(result == 1)
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # False Positive
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp) + 1e-6)
    recall = len(tp) / (len(tp) + len(fn) + 1e-6)

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    F1 = round(F1, 4)
    # print('F1=   ' + str(F1))
    # print('recall=   ' + str(recall))
    # print('precision=   ' + str(precision))

    oa = (len(tp)+len(tn))/m/n      # Overall precision
    pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/m/n/m/n
    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)

    #print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    return oa_kappa
def two_cls_access_for_Bay_Barbara(reference,result):
    # for Bay & Barbra datasets
    # reference:change_value=1;unchange_value=2
    # result: predicted map:change_value=1;unchange_value=0
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W), change=1; unchanged=2;uncertain=0
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    # m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)

    label_0 = np.where(reference == 2)  # Unchanged
    label_1 = np.where(reference == 1)  # Changed
    predict_0 = np.where(result == 0)  # Unchanged
    predict_1 = np.where(result == 1)  # Changed
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # True Negative
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp)+ 1e-6)  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)
    recall = len(tp) / (len(tp) + len(fn))  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)
    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall+ 1e-6)
    F1 = round(F1, 4)
    # print('F1=   ' + str(F1))
    # print('recall=   ' + str(recall))
    # print('precision=   ' + str(precision))
    total_num = len(label_0) +len(label_1)
    oa = (len(tp) + len(tn)) / total_num  # Overall precision
    pe = ((len(tp)+len(fn))*(len(tp)+len(fp)) +(len(fp)+len(tn))*(len(fn)+len(tn)))/ total_num / total_num


    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)

    #print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    return oa_kappa

def cal_cos(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)