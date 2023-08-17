import numpy as np
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  
import argparse
import time
import torch.backends.cudnn
from sklearn import metrics
from scipy.stats import pearsonr
import torch.optim as optim
from sklearn.model_selection import KFold
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from encoder_gcn import GCNNet
from utils import TestbedDataset

SEED = 2022
np.random.seed(SEED)
torch.manual_seed(SEED)  
torch.cuda.manual_seed(SEED)  
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
torch.backends.cudnn.deterministic = True


# train for one epoch to learn unique features
def train_one_epoch(train_net, data_loader, train_optimizer):
    train_net.train()
    print('Training on {} samples...'.format(len(data_loader.dataset)))
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for tem in train_bar:
        out = train_net(tem.cuda())
        out = nn.Sigmoid()(out)
        # out = nn.Softmax()(out)
        loss = loss_fn(out.squeeze(), tem.y)

        total_num += len(tem)
        total_loss += loss.item()
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))
        train_optimizer.zero_grad()
        loss.backward()
        # train_net.l.backward()
        train_optimizer.step()

    return total_loss / total_num


def evaluate(test_net, data_loader):
    test_net.eval()
    t_loss = 0.0
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(data_loader.dataset)))
    with torch.no_grad():
        for data in data_loader:
            pred = test_net(data.cuda())
            pred = nn.Sigmoid()(pred)
            v_loss = loss_fn(pred.squeeze(), data.y)
            t_loss = v_loss.item() + t_loss
            pred = pred.to('cpu')
            y_ = data.y.to('cpu')
            total_preds = torch.cat((total_preds, pred), 0)
            total_labels = torch.cat((total_labels, y_), 0)

    return t_loss, total_labels.numpy().flatten(), total_preds.numpy().flatten()


def analysis(y_true, y_pred):
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    binary_true = [1 if true >= 0.5 else 0 for true in y_true]

    # continous evaluate

    r2 = metrics.r2_score(y_true, y_pred)

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    auc = metrics.roc_auc_score(binary_true, y_pred)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)

    result = {

        'r2': round(r2, 3),
        'binary_acc': round(binary_acc, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3),
        'auc': round(auc, 3),
        'mcc': round(mcc, 3),
        'sensitivity': round(sensitivity, 3),
        'specificity': round(specificity, 3),
    }
    return result


def get_min(results):
    tmp = 0
    test_Min = {

        'r2': 0,
        'binary_acc': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'auc': 0,
        'mcc': 0,
        'sensitivity': 0,
        'specificity': 0,
    }
    for key in test_Min.keys():

        for j in range(5):
            tmp = results[j][key] + tmp
        test_Min[key] = round(tmp / 5, 3)
        tmp = 0
    return test_Min


def get_test_results(test_model, test_data_loader_, name):
    loss_test_l, y_true, y_pred = evaluate(test_model, test_data_loader_)
    # print(analysis(y_true, y_pred))
    test_res = analysis(y_true, y_pred)
    # y_all = np.concatenate([y_true, y_pred], axis=0)
    np.save(name + 'true.npy', y_true)
    np.save(name + 'pred.npy', y_pred)
    rmse = metrics.mean_squared_error(y_true, y_pred) ** 0.5
    print(name + ' ' + str(round(rmse, 3)))
    return test_res, loss_test_l


if __name__ == '__main__':

    print('开始运行')

    parser = argparse.ArgumentParser(description='Train GraphSol')
    parser.add_argument('--train_datafile', default='novel_train', help='orginal data for input train now')
    parser.add_argument('--test_datafile', default='novel_test', help='orginal data for input test now')
    parser.add_argument('--s_test_datafile', default='scerevisiae_test', help='orginal data for input test now')
    parser.add_argument('--path', default='Data', help='processed data for input')
    parser.add_argument('--hidden_dim', default=2048, type=int, help='Feature dim for latent vector')
    parser.add_argument('--output_dim', default=256, type=int, help='Feature dim for latent vector')
    parser.add_argument('--pre_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--dropout', default=0.3, type=float, help='Feature dim for latent vector')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--p', default=0.49, type=float, help='Number of sweeps over the dataset to train')
    parser.add_argument('--PATH', default='1', type=str, help='orginal data for input train now')

    # args parse
    args = parser.parse_args()
    print(args)
    hidden_dim, output_dim, pre_dim = args.hidden_dim, args.output_dim, args.pre_dim
    batch_size, epochs = args.batch_size, args.epochs
    train_datafile, test_datafile, p, s_test_datafile = args.train_datafile, args.test_datafile, args.p, args.s_test_datafile
    PATH = args.PATH
    if not os.path.exists('./results/' + PATH):
        # 文件夹不存在，创建文件夹
        os.makedirs('./results/' + PATH)
        print("文件夹已创建！")
    else:
        print("文件夹已存在！")

    train_data = TestbedDataset(root=args.path, dataset=train_datafile, patt='_train', p=p)
    test_data = TestbedDataset(root=args.path, dataset=test_datafile, patt='_test', p=p)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    s_test_data = TestbedDataset(root=args.path, dataset=s_test_datafile, patt='_test', p=p)
    s_test_data_loader = DataLoader(s_test_data, batch_size=batch_size, shuffle=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    test_results = []
    s_test_results = []
    loss_test_list = []
    loss_s_test_list = []
    cv_result = []
    best_cv_result = {}
    train_true_list = []
    train_pred_list = []
    for train_index, test_index in kf.split(train_data):
        print('现在是fold{}'.format(fold))
        train = train_data.__getitem__(train_index)
        valid = train_data.__getitem__(test_index)
        train_data_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)

        print('use GAT encoder')
        # model_encoder1 = GATNet()
        # model_encoder1 = GCNNet()
        model_encoder1 = TAGCNNet(hidden_dim=hidden_dim, output_dim=output_dim, pre_dim=pre_dim, dropout=args.dropout)

        loss_fn = torch.nn.MSELoss()
        # loss_fn = torch.nn.BCELoss()
        model = model_encoder1.cuda()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        big_loss = 10000

        stopping_monitor = 0
        for epoch in range(0, epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(train_net=model, data_loader=train_data_loader, train_optimizer=optimizer)
            # torch.save(model_encoder.state_dict(), 'results/model/down_task/' + str(epoch) + 'down_task' + '.pkl')
            _, y_true, y_pred = evaluate(model, valid_data_loader)
            val_loss = metrics.mean_squared_error(y_true, y_pred) ** 0.5
            print(val_loss)

            result = analysis(y_true=y_true, y_pred=y_pred)
            if val_loss < big_loss:
                big_loss = val_loss
                stopping_monitor = 0
                print('result_better:', result)
                best_cv_result = result
                torch.save(model.state_dict(), os.path.join('./Model/', 'Fold' + str(fold) + '_best_model.pkl'))
            else:
                stopping_monitor += 1
            if stopping_monitor > 0:
                print('stopping_monitor:', stopping_monitor)
            if stopping_monitor > 5:
                cv_result.append(best_cv_result)
                break
        model.load_state_dict(torch.load('./Model/' + 'Fold' + str(fold) + '_best_model.pkl'))
      
        _, _ = get_test_results(model, valid_data_loader, './results/' + PATH + '/train' + str(fold))

        a, b = get_test_results(model, test_data_loader, './results/' + PATH + '/test' + str(fold))
        loss_test_list.append(b)
        test_results.append(a)

        m, n = get_test_results(model, s_test_data_loader, './results/' + PATH + '/s_test' + str(fold))
        loss_s_test_list.append(n)
        s_test_results.append(m)

        fold += 1

    for i in test_results:
        print('test:', i)
    print(loss_test_list)
    for i in cv_result:
        print('cv:', i)
    for i in s_test_results:
        print('s_test:', i)

    print('五折平均值', get_min(cv_result))

    print('test数据集平均值', get_min(test_results))
    print('s_test数据集平均值', get_min(s_test_results))


