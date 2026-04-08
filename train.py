
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import sys
import yaml
import csv
from timeit import default_timer as timer
from argparse import Namespace

from model import *
from utils import *



def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_epoch_fusion_vae(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=100, colour='red')

    for i, (wsi_data, exp_data, hovernet, label) in enumerate(train_loader):
        optimizer.zero_grad()
        wsi_data, exp_data, hovernet, label = wsi_data.to(device), exp_data.to(device), hovernet.to(device),label.to(device)
        label = label.view(-1,1).float()
        fused_prob, _, x_recon, prot_feature_vae, log_var= model(wsi_data, hovernet, exp_data)
        #
        loss1 = VAE_loss_function(exp_data.squeeze(0), x_recon, prot_feature_vae, log_var)
        loss3 = criterion(fused_prob, label)
        loss = loss3 +loss1
        print(f"Total Loss: {loss.item()}")
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    return train_loss

def val_one_epoch_fusion_vae(fold, model,epoch, val_loader, criterion, device, early_stopping, results_dir):
    model.eval()
    val_loss = 0.
    y_true = []
    y_score = []
    with torch.no_grad():
        for i, (wsi_data, exp_data, hovernet, label, _) in enumerate(val_loader):
            wsi_data, exp_data, hovernet, label = wsi_data.to(device), exp_data.to(device),hovernet.to(device), label.to(device)
            label = label.view(-1,1).float()
            fused_prob, _, x_recon, prot_feature_vae, log_var= model(wsi_data, hovernet,exp_data)
            loss1 = VAE_loss_function(exp_data.squeeze(0), x_recon, prot_feature_vae, log_var)
            loss3 = criterion(fused_prob, label)
            loss = loss3 +loss1
            val_loss += loss.item()
            y_true.append(label.item())
            y_score.append(fused_prob.item())

    val_loss /= len(val_loader)
    with open(f'{results_dir}/val_loss.csv', 'a') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([epoch + 1, val_loss])

    if early_stopping:
        assert results_dir
        early_stopping(epoch, y_true,y_score, model,
                       ckpt_name=os.path.join(results_dir, "fold_{}_minloss_checkpoint.pt".format(fold)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def train(fold: int, results_dir, args: Namespace):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(fold))
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    model = FusionModel(**config['FusionModel'])


    model = model.to(args.device)

    train_set = Prepare_Train_Datasets_FusionModel(fold, args.train_wsi_path)
    val_set = Prepare_Val_Datasets_FusionModel(fold, args.train_wsi_path)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    criterion = torch.nn.BCELoss()

    print(f'Using fold {fold}')
    print(f'train: {len(train_set)}')
    print(f'valid: {len(val_set)}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.early_stopping:
        early_stopping = EarlyStopping_auc(patience=10, stop_epoch=10)
    else:
        early_stopping = None

    with open(f'{results_dir}/train_loss.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'train loss'])

    with open(f'{results_dir}/val_loss.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'val loss'])

    with open(f'{results_dir}/metrics.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['fold', 'specificity', 'sensitivity', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr'])


    for epoch in range(args.max_epochs):
        train_loss = train_one_epoch_fusion_vae(model, train_loader, criterion, optimizer, args.device)
        with open(f'{results_dir}/train_loss.csv', 'a') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch + 1, train_loss])

        stop = val_one_epoch_fusion_vae(fold, model, epoch, val_loader, criterion, args.device, early_stopping, results_dir)
        if stop:
            print(f"Early stop criterion reached. Broke off training loop after epoch {epoch}.")
            break

    #
    model.load_state_dict(torch.load(os.path.join(results_dir, "fold_{}_minloss_checkpoint.pt".format(fold))))
    specificity, sensitivity, accuracy, precision, recall, f1, auc_roc, auc_pr, y_true, y_pred, y_score = cal_metrics(
        model, val_loader, args.device)
    with open(f'{results_dir}/metrics.csv', 'a') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([fold, specificity, sensitivity, accuracy, precision, recall, f1, auc_roc, auc_pr])
    pred = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'y_score': y_score})
    path = os.path.join(results_dir, 'val_predicted_result.csv')
    pred.to_csv(path, index=False, encoding='utf-8')




def get_args_parser():
    parser = argparse.ArgumentParser('Training script', add_help=False)
    parser.add_argument(
            "--train_wsi_path",
            type=str,
            help="Directory of CSV files containing paths to WSIs used for training and validation." #/train_test_split/5_cv_split
        )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation."
    )
    parser.add_argument(
        "--fold_w",
        type=int,
        default=0,
        help="Number of folds for finished cross-validation."
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use for training."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate."
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=1e-5,
        help="Weight decay."
    )
    parser.add_argument('--early_stopping',
                        action='store_true',
                        default=True,
                        help='Enable early stopping.')
    parser.add_argument('--max_epochs',
                        type=int,
                        default=100,
                        help='Maximum number of epochs to train (default: 100).')
    parser.add_argument(
        "--train_results_dir",
        type=str,
        required=True,
        help="Directory to output the training results."
    )
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed for reproducible experiment.')


    return parser

def main(args):
    if not os.path.isdir(args.train_results_dir):
        os.mkdir(args.train_results_dir)

    folds = range(args.fold_w, args.folds)

    ### Start -Fold CV Evaluation.
    for i in folds:
        start = timer()
        seed_torch(args.seed)
        results_dir = f'{args.train_results_dir}/fold_{i}/'
        os.makedirs(results_dir, exist_ok=True)
        train(i, results_dir, args)
        end = timer()
        print('Fold %d Time: %f mins' % (i, (end - start)/60))


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser('Training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    end = timer()
    print("finished train!")
    print('Train Time: %f mins' % ((end - start)/60))