from __future__ import division
from __future__ import print_function

import time
import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Output directory
out_dir = r"E:\桌面文档\pygcn-master\pygcn"
os.makedirs(out_dir, exist_ok=True)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# History
train_losses, train_accs = [], []
val_losses, val_accs = [], []

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    train_losses.append(loss_train.item())
    train_accs.append(acc_train.item())
    val_losses.append(loss_val.item())
    val_accs.append(acc_val.item())

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test.item(), acc_test.item()

def save_curves():
    plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='train_acc')
    plt.plot(val_accs, label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'acc_curve.png'), dpi=150)
    plt.close()


def save_embeddings():
    model.eval()
    with torch.no_grad():
        hidden = F.relu(model.gc1(features, adj))
    hidden_cpu = hidden.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=args.seed)
    emb_2d = tsne.fit_transform(hidden_cpu)

    plt.figure()
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_cpu, s=10, cmap='tab20')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'embedding_tsne.png'), dpi=150)
    plt.close()


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test_loss, test_acc = test()

# Save outputs
save_curves()
save_embeddings()
with open(os.path.join(out_dir, 'test_results.txt'), 'w', encoding='utf-8') as f:
    f.write(f"loss={test_loss:.4f}, accuracy={test_acc:.4f}\n")
