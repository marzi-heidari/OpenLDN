import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from models.resnet_cifar import SimpleDiscriminator
from losses.losses import entropy
from utils.utils import AverageMeter, accuracy, set_seed, save_checkpoint
from datasets.datasets import get_dataset
from models.build_model import build_model
from models.transformer import Gpt
from utils import utils_u
from utils.diffusion_u import create_diffusion, ema
from utils.diffusion_u.timestep_sampler import UniformSampler
from utils.evaluate_utils import hungarian_evaluate
from utils.pseudo_labeling_utils import pseudo_labeling


def main():
    parser = argparse.ArgumentParser(description='Base Training')
    parser.add_argument('--data-root', default=f'data', help='directory to store data')
    parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'aircraft',
                                 'stanfordcars', 'imagenet100', 'herbarium'], help='dataset name')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--pl-percent', type=int, default=10, help='percent of selected pseudo-labels data')
    parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=512, type=int, help='train batchsize')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate, default 1e-3')
    parser.add_argument('--lr-simnet', default=1e-4, type=float, help='earning rate for simnet, default 1e-4')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1, help="random seed (-1: don't use random seed)")
    parser.add_argument('--mu', default=1, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--no-class', default=10, type=int, help='total classes')
    parser.add_argument('--gpu', default=0, type=int, help='total classes')
    parser.add_argument('--threshold', default=0.5, type=float, help='pseudo-label threshold, default 0.50')
    parser.add_argument('--split-id', default='split_0', type=str, help='random data split number')
    parser.add_argument('--ssl-indexes', default='', type=str, help='path to random data split')
    parser.add_argument('--pre', default='pretrained/simclr_cifar_10.pth.tar', type=str,
                        help='closed-world SSL method to use')

    args = parser.parse_args()
    best_acc = 0
    print(' | '.join(f'{k}={v}' for k, v in vars(args).items()))

    writer = SummaryWriter(logdir=args.out)

    with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
        ofile.write('************************************************************************\n\n')
        ofile.write(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
        ofile.write('\n\n************************************************************************\n')

    args.n_gpu = torch.cuda.device_count()
    args.dtype = torch.float32
    if args.seed != -1:
        set_seed(args)

    # set dataset specific parameters
    if args.dataset == 'cifar10':
        args.no_class = 10
    elif args.dataset == 'cifar100':
        args.no_class = 100
    elif args.dataset == 'svhn':
        args.no_class = 10
    elif args.dataset == 'tinyimagenet':
        args.no_class = 200
    elif args.dataset == 'aircraft':
        args.no_class = 100
    elif args.dataset == 'stanfordcars':
        args.no_class = 196
    elif args.dataset == 'oxfordpets':
        args.no_class = 37
    elif args.dataset == 'imagenet100':
        args.no_class = 100
    elif args.dataset == 'herbarium':
        args.no_class = 682

    args.data_root = os.path.join(args.data_root, args.dataset)
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.split_root, exist_ok=True)

    # load dataset
    args.no_known = args.no_class - int((args.novel_percent * args.no_class) / 100)
    lbl_dataset, unlbl_dataset, pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all = get_dataset(args)

    # create dataloaders
    unlbl_batchsize = int((float(args.batch_size) * len(unlbl_dataset)) / (len(lbl_dataset) + len(unlbl_dataset)))
    lbl_batchsize = args.batch_size - unlbl_batchsize
    args.iteration = (len(lbl_dataset) + len(unlbl_dataset)) // args.batch_size

    train_sampler = RandomSampler
    lbl_loader = DataLoader(lbl_dataset, sampler=train_sampler(lbl_dataset), batch_size=lbl_batchsize,
                            num_workers=args.num_workers, drop_last=True)
    unlbl_loader = DataLoader(unlbl_dataset, sampler=train_sampler(unlbl_dataset), batch_size=unlbl_batchsize,
                              num_workers=args.num_workers, drop_last=True)
    pl_loader = DataLoader(pl_dataset, sampler=SequentialSampler(pl_dataset), batch_size=args.batch_size,
                           num_workers=args.num_workers, drop_last=False)
    test_loader_known = DataLoader(test_dataset_known, sampler=SequentialSampler(test_dataset_known),
                                   batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_novel = DataLoader(test_dataset_novel, sampler=SequentialSampler(test_dataset_novel),
                                   batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_all = DataLoader(test_dataset_all, sampler=SequentialSampler(test_dataset_all),
                                 batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    # create model
    model, simnet = build_model(args)

    diffusion_model = Gpt(prototype_sizes=512,
                          predict_xstart=False,
                          max_freq_log2=14,
                          num_frequencies=128,
                          n_embd=512,
                          encoder_depth=1,
                          n_layer=1,
                          n_head=1,
                          len_input=3,
                          attn_pdrop=0.0,
                          resid_pdrop=0.0,
                          embd_pdrop=0.0).cuda(args.gpu)
    diffusion = create_diffusion(learn_sigma=False, predict_xstart=False,
                                 noise_schedule='linear', steps=100)
    timestep_sampler = UniformSampler(diffusion)

    ema_helper = ema.EMAHelper(mu=0.9999)
    ema_helper.register(diffusion_model)
    optimizer_diffusion, lr_scheduler_diffusion = utils_u.make_optimizer(
        diffusion_model.parameters(),
        'sgd', lr=1.e-3, weight_decay=5.e-4, milestones=[30, 50, 80])
    model = model.cuda(args.gpu)
    diffusion_model.cuda(args.gpu)
    discriminator = SimpleDiscriminator(512, args.no_class)
    discriminator = discriminator.cuda(args.gpu)
    optimizer_d = optim.SGD(discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9)
    # simnet = simnet.cuda(args.gpu)

    # optimizer
    if torch.cuda.device_count() > 1:
        optimizer = optim.SGD(model.module.params(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
    # optimizer_diffusion = torch.optim.Adam(simnet.module.params(), lr=args.lr_simnet)
    else:
        optimizer = optim.SGD(model.params(), lr=1e-2, weight_decay=5e-4, momentum=0.9)
    # optimizer_diffusion = torch.optim.Adam(simnet.params(), lr=args.lr_simnet)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    start_epoch = 0
    args.out = os.path.dirname(args.pre)
    checkpoint = torch.load(args.pre)
    # best_acc = checkpoint['best_acc']
    # start_epoch = checkpoint['epoch']
    # print(checkpoint)
    new_state_dict = {}
    for key, value in checkpoint.items():
        if 'contrastive_head' not in key:
            new_key = f'module.{key}'  # Add 'module.' prefix
            new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)
    # Freeze the earlier filters
    for name, param in model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False
    if args.resume:
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        diffusion_model.load_state_dict(checkpoint['state_dict_diffusion'])
        optimizer_diffusion.load_state_dict(checkpoint['optimizer_diffusion'])

    test_accs = []
    model.zero_grad()
    diffusion_model.zero_grad()
    discriminator.zero_grad()
    prototypes = torch.zeros((args.no_class, 512)).cuda(args.gpu)
    for epoch in range(start_epoch, args.epochs):
        train_loss = train(args, lbl_loader, unlbl_loader, model, optimizer, diffusion_model, optimizer_diffusion,
                           epoch, diffusion, timestep_sampler, ema_helper, lr_scheduler_diffusion, discriminator,
                           optimizer_d, prototypes)
        scheduler.step()

        test_acc_known = test_known(args, test_loader_known, model, epoch,
                                    diffusion_model=diffusion_model, timestep_sampler=timestep_sampler,
                                    diffusion=diffusion)
        novel_cluster_results = test_cluster(args, test_loader_novel, model, epoch, offset=args.no_known,
                                             diffusion_model=diffusion_model, timestep_sampler=timestep_sampler,
                                             diffusion=diffusion)
        all_cluster_results = test_cluster(args, test_loader_all, model, epoch,
                                           diffusion_model=diffusion_model, timestep_sampler=timestep_sampler,
                                           diffusion=diffusion)

        test_acc = all_cluster_results

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        print(f'epoch: {epoch}, acc-known: {test_acc_known}')
        print(f'epoch: {epoch}, acc-novel: {novel_cluster_results}, nmi-novel: {novel_cluster_results}')
        print(
            f'epoch: {epoch}, acc-all: {all_cluster_results}, nmi-all: {all_cluster_results}, best-acc: {best_acc}')

        writer.add_scalar('train/1.train_loss', train_loss, epoch)
        writer.add_scalar('test/1.acc_known', test_acc_known, epoch)
        writer.add_scalar('test/2.acc_novel', novel_cluster_results, epoch)
        writer.add_scalar('test/3.nmi_novel', novel_cluster_results, epoch)
        writer.add_scalar('test/4.acc_all', all_cluster_results, epoch)
        writer.add_scalar('test/5.nmi_all', all_cluster_results, epoch)

        model_to_save = model.module if hasattr(model, "module") else model
        simnet_to_save = simnet.module if hasattr(simnet, "module") else simnet
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'state_dict_diffusion': simnet_to_save.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'optimizer_diffusion': optimizer_diffusion.state_dict(),
        }, is_best, args.out, tag='base')

        test_accs.append(test_acc)

        with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
            ofile.write(f'epoch: {epoch}, acc-known: {test_acc_known}\n')
            ofile.write(
                f'epoch: {epoch}, acc-novel: {novel_cluster_results}, nmi-novel: {novel_cluster_results}\n')
            ofile.write(
                f'epoch: {epoch}, acc-all: {all_cluster_results}, nmi-all: {all_cluster_results}, best-acc: {best_acc}\n')

        # close writer
        writer.close()

    # pseudo-label generation and selection
    model.zero_grad()
    optimizer_diffusion.zero_grad()
    lbl_unlbl_dict = pickle.load(
        open(f'{args.split_root}/{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl', 'rb'))
    total_samples = len(lbl_unlbl_dict['labeled_idx']) + len(lbl_unlbl_dict['unlabeled_idx'])
    # no_pl_perclass = int((args.pl_percent*total_samples)/(args.no_class*100))
    no_pl_perclass = int((args.lbl_percent * total_samples) / (args.no_class * 100))
    pl_dict, pl_acc, pl_no = pseudo_labeling(args, pl_loader, model, list(range(args.no_known, args.no_class)),
                                             no_pl_perclass)
    with open(os.path.join(args.out, 'pseudo_labels_base.pkl'), "wb") as f:
        pickle.dump(pl_dict, f)

    with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
        ofile.write(f'acc-pl: {pl_acc}, total-selected: {pl_no}\n')


def compute_unsup_loss(denoised_soft_labels, logits_x_ulb_s, gpu, num_classes=10, unseen_classes=[]):
    # Convert logits to probabilities
    probs = F.softmax(logits_x_ulb_s, dim=1)

    # Find the pseudo-labels and their confidence
    pseudo_labels = torch.argmax(denoised_soft_labels, dim=1)
    max_probs, _ = torch.max(probs, dim=1)

    # Select samples where the confidence is above the threshold
    confident_samples = max_probs > 0.0
    confident_labels = pseudo_labels[confident_samples]
    confident_logits = logits_x_ulb_s[confident_samples]

    # Balance the set
    unique_labels, counts = confident_labels.unique(return_counts=True)
    if len(unique_labels) < num_classes:
        return torch.tensor(0.0).cuda(gpu)
    #
    min_samples = counts.min()
    #
    balanced_indices = torch.cat(
        [confident_samples.nonzero()[confident_labels == label][-min_samples:] for label in unique_labels])

    # # Compute loss only on the balanced subset
    balanced_logits = logits_x_ulb_s[balanced_indices.squeeze(1)]
    balanced_labels = pseudo_labels[balanced_indices.squeeze(1)]
    # balanced_labels = torch.where(balanced_labels < 5, balanced_labels + 5, balanced_labels)

    # Assuming a standard cross-entropy loss for simplicity
    loss = F.cross_entropy(confident_logits, confident_labels)
    return loss


def compute_context_vector(x_u, pseudo_labels, k, tau):
    # Assuming x is of shape (batch_size, num_features)
    # and pseudo_labels is of shape (batch_size, num_classes)
    batch_size, num_classes = pseudo_labels.shape

    # Normalize x to have unit norm, which is required for cosine similarity
    x_normalized = F.normalize(x_u, p=2, dim=1)

    # Compute cosine similarity, shape: (batch_size, batch_size)
    similarities = torch.mm(x_normalized, x_normalized.t())

    # Get indices of k nearest neighbors (excluding self), shape: (batch_size, k)
    _, neighbors_indices = torch.topk(similarities, k + 1, largest=True, sorted=False)
    neighbors_indices = neighbors_indices[:, 1:]  # Exclude self

    # Gather k nearest pseudo labels, shape: (batch_size, k, num_classes)
    neighbors_pseudo_labels = pseudo_labels[neighbors_indices.reshape(-1)].reshape(-1, k, num_classes)

    # Compute weights using cosine similarity, shape: (batch_size, k)
    similarities = similarities.gather(1, neighbors_indices)
    weights = F.softmax(similarities / tau, dim=1)

    # Expand dimensions for proper matrix multiplication
    weights = weights.unsqueeze(2)  # shape: (batch_size, k, 1)

    # Compute context vectors, shape: (batch_size, num_classes)
    y_c = torch.sum(weights * neighbors_pseudo_labels, dim=1)
    # y_c=F.softmax(y_c, dim=1)

    return y_c


def calc_bce_loss(feat, pred_l, pred_u, target):
    bce = torch.nn.BCELoss()

    prob = F.softmax(pred_l, dim=1)
    prob2 = F.softmax(pred_u, dim=1)

    feat_detach = feat.detach()
    feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
    cosine_dist = torch.mm(feat_norm, feat_norm.t())
    pos_pairs = []
    target_np = target.cpu().numpy()
    # label part
    labeled_len = len(pred_l) // 2
    for i in range(labeled_len):
        target_i = target_np[i]
        idxs = np.where(target_np == target_i)[0]
        if len(idxs) == 1:
            pos_pairs.append(idxs[0])
        else:
            selec_idx = np.random.choice(idxs, 1)
            while selec_idx == i:
                selec_idx = np.random.choice(idxs, 1)
            pos_pairs.append(int(selec_idx))

    # unlabel part
    unlabel_cosine_dist = cosine_dist[labeled_len:, :]
    vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
    pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
    pos_pairs.extend(pos_idx)

    pos_prob = prob2[pos_pairs, :]

    pos_sim = torch.bmm(prob.view(2 * labeled_len, 1, -1), pos_prob.view(2 * labeled_len, -1, 1)).squeeze()
    ones = torch.ones_like(pos_sim)
    bce_loss = bce(pos_sim, ones)
    return bce_loss


def compute_prototype(feat, targets, num_classes, is_unseen=False, tau=0.0, gpu=0):
    """
    Compute prototypes for given data.

    Args:
    feat (Tensor): Batch of data (either features or inputs).
    targets (Tensor): Corresponding targets or predicted labels.
    is_unseen (bool): Flag to indicate if the data is for unseen classes.
    tau (float, optional): Threshold for filtering predictions in unseen classes.

    Returns:
    Tensor: Prototype vector.
    """

    if is_unseen and tau is not None:
        # For unseen classes in unlabeled data
        valid_indices = torch.max(targets, dim=1).values > tau
        feat = feat[valid_indices]
        targets = targets[valid_indices]

    # Compute prototypes
    target_index = torch.argmax(targets, dim=1)
    clasees = torch.unique(target_index)

    # assert len(clasees) == num_classes
    prototypes = torch.zeros((num_classes, feat.size(1))).cuda(gpu)

    for idx, cls in enumerate(clasees):
        cls_mask = target_index == cls
        cls_data = feat[cls_mask]
        prototype = torch.mean(cls_data, dim=0)
        prototypes[cls] = prototype.detach()

    return prototypes


def kmeans(X, num_clusters, num_iterations=10):
    """
    Performs K-Means clustering on PyTorch tensors.

    Args:
    X: PyTorch tensor of data points, shape (num_samples, num_features).
    num_clusters: Number of clusters.
    num_iterations: Number of iterations to run the algorithm.

    Returns:
    centroids: Final cluster centroids.
    labels: Index of the cluster each sample belongs to.
    """
    with torch.no_grad():
        # Step 1: Initialize centroids randomly
        indices = torch.randperm(X.size(0))[:num_clusters]
        centroids = X[indices]

        for _ in range(num_iterations):
            # Step 2: Assign each point to the nearest centroid
            distances = torch.cdist(X, centroids, p=2)  # L2 distance
            labels = torch.argmin(distances, dim=1)

            # Step 3: Update centroids
            new_centroids = torch.stack([X[labels == k].mean(dim=0) for k in range(num_clusters)])

            # Check for convergence (if centroids do not change)
            if torch.all(new_centroids == centroids):
                break

            centroids = new_centroids

    return centroids, labels


def update_prototype(old_proto, prototypes, alpha=0.0):
    for i in range(old_proto.shape[0]):
        if old_proto[i].sum() == 0:
            old_proto[i] = prototypes[i]

    prototypes = alpha * old_proto + (1 - alpha) * prototypes
    return prototypes


def calculate_dis_loss(discriminator, diffusion_model, prototypes, feat, diffusion, gpu, context_labels, feat_labels):
    criterion = nn.BCELoss()

    # Forward pass for real data
    real_outputs = discriminator(feat.detach(), feat_labels)
    real_labels = torch.ones_like(real_outputs)  # Real data label is 1
    real_loss = criterion(real_outputs, real_labels)
    with torch.no_grad():
        z_0 = diffusion.p_sample_loop(diffusion_model, prototypes, prototypes.shape, device=f'cuda:{gpu}')
    # Forward pass for fake data
    fake_outputs = discriminator(z_0.detach(), context_labels)
    fake_labels = torch.zeros_like(fake_outputs)  # Fake data label is 0
    fake_loss = criterion(fake_outputs, fake_labels)

    # Total loss
    return real_loss + fake_loss


def calculate_gen_loss(discriminator, diffusion_model, prototypes, diffusion, gpu, context_labels):
    criterion = nn.BCELoss()

    z_0 = diffusion.p_sample_loop(diffusion_model, prototypes.detach(), prototypes.shape, device=f'cuda:{gpu}')
    # Forward pass for fake data
    with torch.no_grad():
        fake_outputs = discriminator(z_0, context_labels)
    fake_labels = torch.ones_like(fake_outputs)  # Fake data label is 1
    fake_loss = criterion(fake_outputs, fake_labels)

    # Total loss
    return fake_loss


def train(args, lbl_loader, unlbl_loader, model, optimizer, diffusion_model, optimizer_diffusion, epoch, diffusion=None,
          timestep_sampler=None, ema_helper=None, lr_scheduler_diffusion=None, discriminator=None, optimizer_d=None,
          prototypes=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_pair = AverageMeter()
    losses_reg = AverageMeter()

    end = time.time()
    diffusion_model.train()
    discriminator.train()
    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = zip(lbl_loader, unlbl_loader)
    for batch_idx, (data_lbl, data_unlbl) in enumerate(train_loader):
        inputs_l, targets_l, _ = data_lbl
        (inputs_u_w, inputs_u_s), _, _ = data_unlbl
        inputs_l=inputs_l[0]


        feats_x_lb, logits_x_lb = model(inputs_l.cuda(args.gpu))
        feats_x_lb2, logits_x_lb2 = model(inputs_l.cuda(args.gpu))
        feats_x_ulb_w, logits_x_ulb_w = model(inputs_u_w.cuda(args.gpu))
        feat_u_s, logits_x_ulb_s = model(inputs_u_s.cuda(args.gpu))
        output = torch.cat((logits_x_lb, logits_x_ulb_w), dim=0)
        feat = torch.cat((feats_x_lb, feats_x_ulb_w), dim=0)
        output2 = torch.cat((logits_x_lb2, logits_x_ulb_s), dim=0)
        # feat_ = torch.cat((feats_x_lb2, feat_u_s), dim=0)
        target = targets_l.cuda(args.gpu)

        labeled_len = len(target)
        # Compute context vectors for unlabeled data

        one_hot_labels = F.one_hot(target, args.no_class)
        p1 = compute_prototype(feat[:labeled_len, ], one_hot_labels, args.no_class // 2, gpu=args.gpu)
        pseudo_labels = F.softmax(output2[labeled_len:], dim=1)
        if epoch == 0:
            p2 = kmeans(feat[labeled_len:, :], args.no_class, )[0]
        else:
            # pseudo_labels = calc_pseudo_lables(feat, labeled_len, prototypes)
            p2 = compute_prototype(feat[labeled_len:, :], pseudo_labels, args.no_class, gpu=args.gpu, is_unseen=True, )
        proto = torch.cat((p1, p2[args.no_class // 2:, :].cuda(args.gpu)), dim=0)
        prototypes = update_prototype(prototypes, proto)

        t, vlb_weights = timestep_sampler.sample(args.batch_size, args.gpu)
        t = t.cuda(args.gpu)
        context_vector = torch.zeros((args.batch_size, 512)).cuda(args.gpu)
        feat_labels = torch.zeros((args.batch_size, args.no_class)).cuda(args.gpu)
        context_labels = torch.zeros((args.batch_size, args.no_class)).cuda(args.gpu)
        # for i in range(labeled_len):
        #     context_vector[i] = prototypes[target[i]]
        # for i in range(labeled_len, args.batch_size):
        #     context_vector[i] = prototypes[torch.argmax(pseudo_labels[i - labeled_len])]
        for i in range(args.batch_size):
            context_vector[i] = prototypes[torch.argmax(output[i])]
            context_labels[i] = F.one_hot(torch.argmax(output[i]), args.no_class).to(torch.float32)
        for i in range(labeled_len):
            feat_labels[i] = one_hot_labels[i].to(torch.float32)
        for i in range(labeled_len, args.batch_size):
            feat_labels[i] = F.one_hot(torch.argmax(pseudo_labels[i - labeled_len]), args.no_class).to(torch.float32)

        diffusion_loss_, denoised_labdeld_ = diffusion.training_losses(diffusion_model,
                                                                       feat,
                                                                       context_vector, t)

        discriminator_loss = calculate_dis_loss(discriminator, diffusion_model, context_vector, feat,

                                                diffusion, args.gpu, context_labels, feat_labels)

        generator_loss = calculate_gen_loss(discriminator, diffusion_model, context_vector.detach(),

                                            diffusion, args.gpu, context_labels)

        # logits_ub_s = output[labeled_len:, :]
        # unsup_loss = compute_unsup_loss(pseudo_labels, logits_ub_s, args.gpu, num_classes=args.no_class)
        # bce_loss = calculate_bce_loss(args, output2, labeled_len, F.softmax(output, dim=1), feat, target)
        # query_to_proto_distance = euclidean_dist(feat[:labeled_len], prototypes[:args.no_class // 2])
        # aug_loss = calulate_aug_loss(diffusion_model, model, prototypes, diffusion, args.gpu)
        ce_loss = F.cross_entropy(output[:labeled_len], target)
        # distances = euclidean_dist(feat[labeled_len:], prototypes)
        # distances2 = euclidean_dist(feat_[labeled_len:], prototypes)
        unsup_loss = compute_unsup_loss(pseudo_labels, output[labeled_len:], args.gpu)
        #
        # prob = F.softmax(-distances, dim=1)
        # # print(prob[0])
        # # exit()
        # entropy_loss = entropy(torch.mean(prob, 0))
        if epoch < 2:
            a = 0
            loss = ce_loss
        else:
            a = 1
            loss = ce_loss + 0.5 * unsup_loss + diffusion_loss_.mean() + discriminator_loss + generator_loss

        # inputs_l, y_lb, _ = data_lbl
        # (inputs_u_w, inputs_u_s), _, _ = data_unlbl

        # inputs = interleave(
        #     torch.cat((inputs_l, inputs_u_w, inputs_u_s)), 2*args.mu+1).cuda(args.gpu)
        # y_lb = y_lb.cuda(args.gpu)
        # batch_l = inputs_l.shape
        #
        # model.train()
        # # create intermediate model for bi-level optimization
        # model_, _ = build_model(args)
        # model_ = model_.cuda(args.gpu)
        # model_.load_state_dict(model.state_dict())
        #
        # feat, logits = model_(inputs)
        # logits = de_interleave(logits, 2*args.mu+1)
        # logits_x_lb = logits[:batch_l[0]]
        # logits_x_ulb_w, logits_x_ulb_s = logits[batch_l[0]:].chunk(2)
        #
        # feat = de_interleave(feat, 2*args.mu+1)
        # feats_x_lb = feat[Îƒ:batch_l[0]]
        # feats_x_ulb_w, _ = feat[batch_l[0]:].chunk(2)
        # feats = torch.cat((feats_x_lb, feats_x_ulb_w), 0)
        #
        # pseudo_label = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
        # max_probs_pl, targets_u_pl = torch.max(pseudo_label, dim=-1)
        # mask_pl = max_probs_pl.ge(args.threshold).float()
        #
        # feats = feats.unsqueeze(0).repeat(feats.shape[0],1,1)
        # feats_t = torch.transpose(feats,0,1)
        # feat_pairs = torch.cat((feats, feats_t),2).view(-1, feats.shape[2]*2)
        # sim_feat = simnet(feat_pairs).view(-1,feats.shape[0])
        #
        # class_logit = torch.cat((clean_logits_u, clean_logits_l), 0)
        # sim_prob = sim_matrix(F.softmax(class_logit, dim=1), F.softmax(class_logit, dim=1), args)
        #
        # loss_pair = symmetric_mse_loss(sim_prob.view(-1), sim_prob.view(-1)) / sim_prob.view(-1).view(-1).shape[0]
        # loss_reg = entropy(torch.mean(F.softmax(class_logit, dim=1), 0), input_as_probabilities=True)
        # loss_ce_supervised = F.cross_entropy(class_logit[:batch_l[0]], y_lb)
        # loss_ce_pseudo = (F.cross_entropy(logits_x_ulb_s, targets_u_pl, reduction='none') * mask_pl).mean()
        # loss_ce = loss_ce_supervised + loss_ce_pseudo

        # loss = total_loss+loss_pair

        # model_.zero_grad()
        # # compute gradients for the intermediate update
        # if torch.cuda.device_count() > 1:
        #     grads = torch.autograd.grad(loss, (model_.module.params()), create_graph=True)
        # else:
        #     grads = torch.autograd.grad(loss, (model_.params()), create_graph=True)
        #
        # # update the model parameters
        # if torch.cuda.device_count() > 1:
        #     model_.module.update_params(lr_inner=1e-3, source_params=grads)
        # else:
        #     model_.update_params(lr_inner=1e-3, source_params=grads)
        #
        # del grads

        # update simnet parameters
        # feat, logits = model_(inputs)
        # logits = de_interleave(logits, 2*args.mu+1)
        # logits_x_lb = logits[:batch_l[0]]
        # loss_ce_supervised = F.cross_entropy(logits_x_lb, y_lb)

        # optimizer_diffusion.zero_grad()
        # # loss_ce_supervised.backward()
        # optimizer_diffusion.step()
        #
        # # update main model parameters with updated simnet
        # feat, logits = model(inputs)
        # logits = de_interleave(logits, 2*args.mu+1)
        # logits_x_lb = logits[:batch_l[0]]
        # logits_x_ulb_w, logits_x_ulb_s = logits[batch_l[0]:].chunk(2)
        #
        # feat = de_interleave(feat, 2*args.mu+1)
        # feats_x_lb = feat[:batch_l[0]]
        # feats_x_ulb_w, _ = feat[batch_l[0]:].chunk(2)
        # feats = torch.cat((feats_x_lb, feats_x_ulb_w), 0)
        #
        # pseudo_label = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
        # max_probs_pl, targets_u_pl = torch.max(pseudo_label, dim=-1)
        # mask_pl = max_probs_pl.ge(args.threshold).float()
        #
        # feats = feats.unsqueeze(0).repeat(feats.shape[0],1,1)
        # feats_t = torch.transpose(feats,0,1)
        # feat_pairs = torch.cat((feats, feats_t),2).view(-1, feats.shape[2]*2)
        #
        # # no gradients for the simnet parameters
        # # with torch.no_grad():
        # #     sim_feat = simnet(feat_pairs).view(-1,feats.shape[0])
        # class_logit = torch.cat((logits_x_lb, logits_x_ulb_w), 0)
        # sim_prob = sim_matrix(F.softmax(class_logit,dim=1), F.softmax(class_logit,dim=1), args)
        #
        # # loss_pair = symmetric_mse_loss(sim_prob.view(-1), sim_feat.view(-1))/sim_feat.view(-1).shape[0]
        # loss_reg = entropy(torch.mean(F.softmax(class_logit, dim=1), 0), input_as_probabilities = True)
        # loss_ce_supervised = F.cross_entropy(class_logit[:batch_l[0]], y_lb)
        # loss_ce_pseudo = (F.cross_entropy(logits_x_ulb_s, targets_u_pl, reduction='none') * mask_pl).mean()
        # loss_ce = loss_ce_supervised + loss_ce_pseudo
        #
        # final_loss = loss_pair - loss_reg + loss_ce
        final_loss = loss
        losses.update(final_loss.item(), inputs_l.size(0))
        losses_ce.update(ce_loss.item(), inputs_l.size(0))
        losses_pair.update(unsup_loss.mean().item(), inputs_l.size(0))
        losses_reg.update(unsup_loss.item(), inputs_l.size(0))

        optimizer.zero_grad()
        optimizer_d.zero_grad()
        optimizer_diffusion.zero_grad()
        final_loss.backward()
        optimizer.step()
        optimizer_d.step()
        optimizer_diffusion.step()
        lr_scheduler_diffusion.step()
        ema_helper.update(diffusion_model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not args.no_progress:
            p_bar.set_description(
                "train epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. loss: {loss:.4f}. l_ce: {loss_ce:.4f}. l_pair: {loss_pair:.4f}. l_reg: {loss_reg:.4f}".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.iteration,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_ce=losses_ce.avg,
                    loss_pair=losses_pair.avg,
                    loss_reg=losses_reg.avg,
                ))
            p_bar.update()

    if not args.no_progress:
        p_bar.close()

    return losses.avg


def test_known(args, test_loader, model, epoch, diffusion_model=None, diffusion=None,
               timestep_sampler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(args.gpu)
            targets = targets.cuda(args.gpu)
            _, outputs = model(inputs)

            # outputs = denoised_labeld["pred_xstart"]
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()

    return top1.avg


def test_cluster(args, test_loader, model, epoch, offset=0, diffusion_model=None, diffusion=None,
                 timestep_sampler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    gt_targets = []
    predictions = []
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda(args.gpu)
            targets = targets.cuda(args.gpu)

            _, outputs = model(inputs)
            # # y_c_u = sampling.generalized_steps(diffusion_model, F.softmax(outputs, dim=1),
            # #                                    100)
            # t = torch.ones(outputs.shape[0], dtype=torch.long).cuda(args.gpu)
            # t = t.cuda(args.gpu)
            # denoised_labeld = diffusion.p_sample(diffusion_model,
            #                                      outputs, F.softmax(outputs, dim=1), t)

        # outputs = denoised_labeld["pred_xstart"]
        _, max_idx = torch.max(outputs, dim=1)
        predictions.extend(max_idx.cpu().numpy().tolist())
        gt_targets.extend(targets.cpu().numpy().tolist())
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_progress:
            test_loader.set_description(
                "test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    y_pred = np.array(predictions)
    y_true = np.array(gt_targets)

    """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

    return eval_output


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
