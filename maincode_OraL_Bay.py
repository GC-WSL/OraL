import random
import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from NL_model import HICDetector
from get_datasets import get_args_Bay
from utils import initNetParams_v2, adjust_learning_rate, two_cls_access, two_cls_access_for_Bay_Barbara

torch.set_num_threads(2)


@torch.no_grad()
def instance_selection(cls_prob, num, instance_weights1, instance_weights2):
    instance_weights = instance_weights1 * instance_weights2
    values, indices = instance_weights.topk(np.min([int(num), instance_weights.shape[0]]), dim=0)
    instance_weights = torch.zeros_like(instance_weights)
    instance_weights[indices.squeeze()] = values
    return instance_weights


@torch.no_grad()
def temporal_consistency(temporal_dict, pixel_index, epoch, startepoch=5, step=3):
    if epoch < startepoch:
        instance_weights = torch.ones((pixel_index.shape[0], 1)).cuda()
    else:
        instance_weights = torch.zeros((pixel_index.shape[0], 1)).cuda()
        for i in range(step - 1):
            temp_prob1 = temporal_dict[pixel_index, epoch - step + i, :]
            temp_prob2 = temporal_dict[pixel_index, epoch - step + i + 1, :]
            instance_weights += 1 / step * 0.5 * (
                torch.sum(F.kl_div(temp_prob1.log_softmax(dim=1), (temp_prob2.softmax(dim=1) + temp_prob1.softmax(dim=1)) / 2, reduction='none'), dim=-1) +
                torch.sum(F.kl_div(temp_prob2.log_softmax(dim=1), (temp_prob2.softmax(dim=1) + temp_prob1.softmax(dim=1)) / 2, reduction='none'), dim=-1)
            ).reshape(-1, 1)
        instance_weights = torch.exp(-2 * instance_weights)
    return instance_weights


@torch.no_grad()
def spectral_consistency(model, train_img1, train_img2, epoch, startepoch, rates=[0.5, 0.3, 0.1]):
    if epoch < startepoch:
        instance_weights = torch.ones((train_img1.shape[0], 1)).cuda()
    else:
        step = len(rates)
        instance_weights = torch.zeros((train_img1.shape[0], 1)).cuda()
        temp_cls_prob = torch.zeros((train_img1.shape[0], step, 2)).cuda()
        for idx, rate in enumerate(rates):
            train_img1, train_img2 = F.dropout(train_img1, rate), F.dropout(train_img2, rate)
            temp_cls_prob[:, idx, :] = model(train_img1.unsqueeze(1), train_img2.unsqueeze(1))

        for i in range(step - 1):
            temp_prob1 = temp_cls_prob[:, i, :]
            temp_prob2 = temp_cls_prob[:, i + 1, :]
            instance_weights += 1 / step * 0.5 * (
                torch.sum(F.kl_div(temp_prob1.log_softmax(dim=1), (temp_prob2.softmax(dim=1) + temp_prob1.softmax(dim=1)) / 2, reduction='none'), dim=-1) +
                torch.sum(F.kl_div(temp_prob2.log_softmax(dim=1), (temp_prob2.softmax(dim=1) + temp_prob1.softmax(dim=1)) / 2, reduction='none'), dim=-1)
            ).reshape(-1, 1)
        instance_weights = torch.exp(-2 * instance_weights)
    return instance_weights


def train_STEP(args, C, startepoch, temporal_windows, sigma):
    print('---------------------------func: train_STEP---------------------------')
    print('\n')
    head_num = 12
    model = HICDetector(
                        seq_len=C * 2, patch_size=1, num_classes=2,
                        dim=C * 2, depth=2, heads=12,
                        mlp_dim=C * 2, channels=1)
    model.apply(initNetParams_v2)

    # ---------------------------- Data loading ----------------------------
    data = sio.loadmat(args.data_name)
    print('input data for test:', args.data_name, sep='\n')
    img_1 = data['img_1']
    img_2 = data['img_2']  # H, W, C
    H, W, C = img_1.shape
    real_gt = data['GT']
    pseudo_gt = sio.loadmat('./result_NL/pseudo_gt_{}.mat'.format(args.data_name.split('/')[-1].split('.')[0]))

    if not args.EX_num == 'USA':
        gt, _, _, ig_idx = pseudo_gt['GT'], pseudo_gt['unchanged_idx'], pseudo_gt['changed_idx'], pseudo_gt['ignored_idx']
    else:
        gt, _, _ = pseudo_gt['GT'], pseudo_gt['unchanged_idx'], pseudo_gt['changed_idx']

    X1 = torch.tensor(img_1.reshape(-1, C), dtype=torch.float32).cuda() / 255.0
    X2 = torch.tensor(img_2.reshape(-1, C), dtype=torch.float32).cuda() / 255.0

    gt_label = torch.tensor(gt.reshape(-1), dtype=torch.float32).cuda()
    ori_X1, ori_X2 = X1, X2
    gt_label_bak = gt_label
    if not args.EX_num == 'USA':
        contained_idx = np.where(gt.reshape(-1) != 0)
        X1 = X1[contained_idx]
        X2 = X2[contained_idx]
        gt_label = gt_label[contained_idx]
        idxes = contained_idx[0]
        gt[0, contained_idx] = 3 - gt[0, contained_idx]
        pseudo_gt = gt.reshape(H, W)
        cal_gt = gt_label - 1
    else:
        pseudo_gt = gt.reshape(H, W)
        idxes = np.arange(X1.shape[0])
        cal_gt = gt_label
    print('img_1 and img_2 is input for test')
    print('input.shape:', X1.shape)

    X = torch.concat([X1, X2], dim=-1)
    changed_mean = X[cal_gt == 0].mean(0)
    unchanged_mean = X[cal_gt == 1].mean(0)
    MEAN = torch.cat([changed_mean.unsqueeze(0), unchanged_mean.unsqueeze(0)], dim=0)
    del img_1, img_2, data

    if not args.test:
        # ---------------------------- Optimizer ----------------------------
        init_lr = args.lr
        model.cuda()
        optim_params = model.parameters()
        optimizer = torch.optim.AdamW(optim_params, lr=init_lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        cudnn.benchmark = True
        num_sample = X1.shape[0]
        patch_label = torch.zeros((num_sample, 2))
        if not args.EX_num == 'USA':
            patch_label[gt_label == 1, 0] = 1  # Changed
            patch_label[gt_label == 2, 1] = 1  # UnChanged
        else:
            patch_label[gt_label == 0, 0] = 1  # Changed
            patch_label[gt_label == 1, 1] = 1  # UnChanged

        train_dataset = TensorDataset(torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32), torch.tensor(patch_label, dtype=torch.float32), torch.tensor(idxes, dtype=torch.long))
        num_batch_size = 1024
        data_loader = DataLoader(train_dataset, batch_size=num_batch_size, shuffle=True, drop_last=True)

        # ---------------------------- Training ----------------------------
        print('training begins----------------------------')
        ori_acc = []
        ours_acc = []
        gt_label_bak = gt_label_bak.cpu().numpy()

        if args.EX_num != 'USA':
            init_gt = (gt_label_bak[contained_idx[0]] - 1)
            real_gt_eval = (real_gt.reshape(-1)[contained_idx])
            print('Initial_Pseudo_Acc', (2 - init_gt == real_gt_eval).sum() / contained_idx[0].shape[0])
        else:
            init_gt = (gt_label_bak)
            real_gt_eval = (real_gt.reshape(-1))
            print('Initial_Pseudo_Acc', (init_gt == real_gt_eval).sum() / real_gt_eval.shape[0])

        select_num = num_batch_size * 0.05
        clean_index = []

        start = time.time()
        for phase in range(args.phases):
            memory_bank = torch.zeros(ori_X1.shape[0], args.epochs[phase], 2).cuda()
            if phase == 1:
                # Rebuild the model (with dropout slots) and refresh the optimizer
                del optimizer, data_loader
                torch.cuda.empty_cache()
                dropout_rate = 0.0
                if args.EX_num == 'USA':
                    new_model = HICDetector(
                        seq_len=C * 2, patch_size=1, num_classes=2,
                        dim=C * 2, depth=2, heads=head_num,
                        mlp_dim=C * 2, channels=1,
                        dropout=0.0, emb_dropout=dropout_rate)
                else:
                    new_model = HICDetector(
                        seq_len=C * 2, patch_size=1, num_classes=2,
                        dim=C * 2, depth=2, heads=12,
                        mlp_dim=C * 2, channels=1,
                        dropout=dropout_rate, emb_dropout=dropout_rate)
                new_model.apply(initNetParams_v2)
                model = new_model.cuda()
                optim_params = model.parameters()
                optimizer = torch.optim.AdamW(optim_params, lr=init_lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

                # Update pseudo labels with the selected clean samples
                temp = torch.tensor([], dtype=torch.int32)
                for t in clean_index:
                    temp = torch.cat([temp, t])
                clean_index = torch.unique(temp).reshape(-1).long()
                if not args.EX_num == 'USA':
                    clean_gt = 1 - (gt[0][clean_index] - 1)
                    print('Pseudo_Acc', (clean_gt == (2 - real_gt.reshape(-1)[clean_index])).sum() / clean_index.shape[0])
                else:
                    clean_gt = gt[0][clean_index]
                    print('Pseudo_Acc', (clean_gt == real_gt.reshape(-1)[clean_index]).sum() / clean_index.shape[0])
                num_sample = clean_index.shape[0]
                print('New num_sample:', num_sample, 'Pos num:', (clean_gt == 0).sum(), 'Neg num:', (clean_gt == 1).sum())
                clean_patch_label = torch.zeros((num_sample, 2))
                if not args.EX_num == 'USA':
                    clean_patch_label[clean_gt == 0, 0] = 1  # Changed
                    clean_patch_label[clean_gt == 1, 1] = 1  # UnChanged
                else:
                    clean_patch_label[clean_gt == 0, 0] = 1  # Changed
                    clean_patch_label[clean_gt == 1, 1] = 1  # UnChanged
                clean_X1, clean_X2, clean_patch_label, clean_idxes = ori_X1[clean_index], ori_X2[clean_index], clean_patch_label, clean_index
                train_dataset = TensorDataset(torch.tensor(clean_X1, dtype=torch.float32), torch.tensor(clean_X2, dtype=torch.float32), torch.tensor(clean_patch_label, dtype=torch.float32), torch.tensor(clean_idxes, dtype=torch.long))
                data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            for epoch in range(args.epochs[phase]):
                if phase == 0:
                    adjust_learning_rate(optimizer, init_lr, epoch, args.epochs[phase])
                for i, data in enumerate(data_loader):
                    train_img1, train_img2, pixel_label, pixel_index = data
                    train_img1, train_img2, pixel_label, pixel_index = train_img1.cuda(), train_img2.cuda(), pixel_label.cuda(), pixel_index.cuda()
                    if phase == 0:
                        cls_prob = model(train_img1.unsqueeze(1), train_img2.unsqueeze(1))
                        temporal_weight_ = temporal_consistency(memory_bank, pixel_index, epoch, startepoch, temporal_windows)
                        spectral_weight = spectral_consistency(model, train_img1, train_img2, epoch, startepoch)
                        temporal_weight = instance_selection(cls_prob, select_num, temporal_weight_, spectral_weight)
                        loss = F.binary_cross_entropy_with_logits(cls_prob, pixel_label)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if epoch >= startepoch + 1:
                            s_index = pixel_index[(temporal_weight != 0).reshape(-1)].cpu()
                            clean_index.append(s_index)
                            if args.EX_num != 'USA':
                                gt_labels = 2 - real_gt.reshape(-1)[s_index]
                                ori_gt = 2 - real_gt.reshape(-1)[pixel_index.reshape(-1).cpu()]
                            else:
                                gt_labels = real_gt.reshape(-1)[s_index]
                                ori_gt = real_gt.reshape(-1)[pixel_index.reshape(-1).cpu()]
                            pre_labels = (pixel_label[(temporal_weight != 0).reshape(-1), :].argmax(dim=1)).cpu().numpy()
                            ori_labels = pixel_label.argmax(dim=1).cpu().numpy()
                            ori_acc.append((ori_gt == ori_labels).sum() / ori_gt.shape[0])
                            ours_acc.append((gt_labels == pre_labels).sum() / gt_labels.shape[0])
                    else:
                        mul_rate = 0.5
                        img = torch.concat([train_img1, train_img2], dim=-1)
                        factor = torch.exp(-(torch.norm(img - MEAN[pixel_label.argmax(1)], p=2, dim=1)) / (2 * sigma)).detach()
                        factor = factor.repeat(C).reshape(-1, C)

                        drop_mask1, drop_mask2 = torch.ones_like(train_img1), torch.ones_like(train_img2)
                        drop_mask1[torch.rand_like(train_img1) < mul_rate * factor] = 0
                        drop_mask2[torch.rand_like(train_img2) < mul_rate * factor] = 0
                        train_img1, train_img2 = train_img1 * drop_mask1, train_img2 * drop_mask2

                        add_rate = 0.05
                        spectral_jitter1 = add_rate * torch.randn_like(train_img1) * factor
                        spectral_jitter2 = add_rate * torch.randn_like(train_img2) * factor
                        train_img1 += spectral_jitter1
                        train_img2 += spectral_jitter2

                        cls_prob = model(train_img1.unsqueeze(1), train_img2.unsqueeze(1))
                        loss = F.binary_cross_entropy_with_logits(cls_prob, pixel_label)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                print('epoch [{}/{}], train:{:.4f}'.format(epoch, args.epochs[phase], loss.item()))

                with torch.no_grad():
                    step = 2000
                    for i in range(0, X1.shape[0], step):
                        input_data1 = X1[i:i + step, :].reshape(-1, C)
                        input_data2 = X2[i:i + step, :].reshape(-1, C)
                        idx = idxes[i:i + step]
                        memory_bank[idx, epoch, :] = model(input_data1.unsqueeze(1), input_data2.unsqueeze(1))

        print('--------------STEP: training is successfully done--------------')
        print('--------------model save_path:', args.save_model_path_NL, sep='\n')
        print(time.time() - start)
        plt.figure()
        plt.plot(np.arange(len(ours_acc)), np.asarray(ours_acc), 'r', label="Ours")
        plt.plot(np.arange(len(ori_acc)), np.asarray(ori_acc), 'g', label="Init")
        plt.legend()
        plt.savefig('./result_NL/{}/pseudo_acc_{}.jpg'.format(args.EX_num, args.sess))
        plt.close()

        # ---------------------------- Testing ----------------------------
        cluster_labels = torch.zeros((H * W))
        step = 500
        for i in tqdm(range(0, H * W, step)):
            input_data1 = ori_X1[i:i + step, :].reshape(-1, C)
            input_data2 = ori_X2[i:i + step, :].reshape(-1, C)
            temp_cluster_labels = model(input_data1.unsqueeze(1), input_data2.unsqueeze(1))
            cluster_labels[i:i + step] = torch.argmax(temp_cluster_labels.squeeze().cpu().detach(), dim=1)

        # ------------------------- Two-class assessment -------------------------
        print('---Two-class assessment---')
        Two_Chge_label = real_gt
        fig_result = (1 - cluster_labels) + 1
        if not args.EX_num == 'USA':
            fig_result[ig_idx] = 0
        plt.imshow(1 - fig_result.reshape(H, W))
        plt.savefig('./result_NL/{}/{}_pseudo_labels.png'.format(args.EX_num, args.sess))
        if args.EX_num == 'USA':
            bi_oa_kappa1 = two_cls_access(Two_Chge_label, cluster_labels.reshape(H, W))
        else:
            bi_oa_kappa1 = two_cls_access_for_Bay_Barbara(Two_Chge_label, cluster_labels.reshape(H, W))
        print('\n {},'.format(bi_oa_kappa1[1]), '{},'.format(bi_oa_kappa1[3]),
              '{},'.format(bi_oa_kappa1[5]), '{},'.format(bi_oa_kappa1[9]),
              '{}'.format(bi_oa_kappa1[7]))
        print('\n OA:{}\n'.format(bi_oa_kappa1[1]), 'Kappa:{}\n'.format(bi_oa_kappa1[3]),
              'F1:{}\n'.format(bi_oa_kappa1[5]), 'Precision:{}\n'.format(bi_oa_kappa1[9]),
              'Recall:{}\n'.format(bi_oa_kappa1[7]))
        sio.savemat('./result_NL/{}/{}.mat'.format(args.EX_num, args.sess),
                    {'detection_results': fig_result.numpy().reshape(H, W),
                     'ori_pl': np.asarray(ori_acc),
                     'our_pl': np.asarray(ours_acc)})


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def zz(seed):
    print('---------------everything will be ok-------------')
    print('current seed:', seed)


if __name__ == "__main__":
    seed = 0
    warmup_step = 2
    temporal_windows = 2

    # ------------------------------------------------------------------
    # Select one dataset to run. `sigma` is the CVA-based noise scale;
    # alternative sigmas for the compared methods are listed for
    # reproducibility.
    #   USA (Hermiston): CVA 2.9983 | PCA 3.0510 | HyperNet 2.9682 | DiffuCD 2.6560 | BCGNet 2.6670
    #   Bay:             CVA 0.2174 | PCA 0.1599 | HyperNet 0.2040 | DiffuCD 0.1164
    #   Barbara:         CVA 6.0596 | PCA 0.6092 | HyperNet 0.7698 | DiffuCD 0.6457
    # ------------------------------------------------------------------

    # --- Dataset: USA (Hermiston) ---
    # sigma = 2.9983  # CVA
    # C = 154
    # args = get_args_USA(seed)
    # setup_seed(args.seed)
    # zz(seed)
    # train_STEP(args, C, warmup_step, temporal_windows, sigma)

    # --- Dataset: Bay ---
    sigma = 0.2174  # CVA
    C = 224
    args = get_args_Bay(seed)
    setup_seed(args.seed)
    zz(seed)
    train_STEP(args, C, warmup_step, temporal_windows, sigma)

    # --- Dataset: Barbara ---
    # sigma = 6.0596  # CVA
    # args = get_args_Barbara(seed)
    # setup_seed(args.seed)
    # zz(seed)
    # train_STEP(args, warmup_step, temporal_windows, sigma)
