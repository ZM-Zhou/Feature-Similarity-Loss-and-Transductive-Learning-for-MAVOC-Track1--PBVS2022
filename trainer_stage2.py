import os
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
import time
import argparse
import csv
from tqdm import tqdm

from dataset import EOSAR_Dataset
from swin_transformer import SwinTransformer, Classifier


seed = 310
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def _print(arg, *argg, **kargs):
    output_file = os.path.join(args.save_path, 'train_log.txt')
    with open(output_file, 'a+') as f:
        if kargs:
            print(arg, end=kargs['end'])
            f.write(arg + kargs['end'])
        else:
            print(arg)
            f.write(str(arg) + '\n')

def parse_args():
    """
    Parse input arguments
    Returns
    -------
    args : object
        Parsed args
    """
    h = {
        "program": "Simple Baselines training",
        "train_folder": "Path to training data folder.",
        "batch_size": "Number of images to load per batch. Set according to your PC GPU memory available. If you get "
                      "out-of-memory errors, lower the value. defaults to 64",
        "epochs": "How many epochs to train for. Once every training image has been shown to the CNN once, an epoch "
                  "has passed. Defaults to 15",
        "test_folder": "Path to test data folder",
        "num_workers": "Number of workers to load in batches of data. Change according to GPU usage",
        "test_only": "Set to true if you want to test a loaded model. Make sure to pass in model path",
        "model_path": "Path to your model",
        "learning_rate": "The learning rate of your model. Tune it if it's overfitting or not learning enough"}
    parser = argparse.ArgumentParser(description=h['program'], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', type=str, default='SwinB')
    parser.add_argument('--train_folder', help=h["train_folder"], type=str)
    parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=64)
    parser.add_argument('--epochs', help=h["epochs"], type=int, default=15)
    parser.add_argument('--num_workers', help=h["num_workers"], type=int, default=4)
    parser.add_argument('--learning_rate', help=h["learning_rate"], type=float, default=0.003)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='./train_log/debug')
    
    parser.add_argument('--test_folder', help=h["test_folder"], type=str)
    parser.add_argument('--test_only', help=h["test_only"], type=bool, default=False)

    parser.add_argument('--sample_num', type=int, default=62)
    parser.add_argument('--a_sim', type=float, default=1)
   
    args = parser.parse_args()

    return args


def load_train_data(train_data_path, batch_size, test_data_path):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(180,fill=(101,
                                                                        101,
                                                                        101)),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.2, 0.2, 0, 0),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.3984,
                                                          0.3984,
                                                          0.3984),
                                                         (0.1328,
                                                          0.1328,
                                                          0.1328))])
    train_data = EOSAR_Dataset(train_data_path,
                               'data_splits/train_list_raw.txt',
                               transform,
                               read_EO=False,
                               image_to_RGB=True,
                               uniform_sample=500,
                               test_path=test_data_path)
    train_data_loader = data.DataLoader(train_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers)

    return train_data_loader, train_data

def load_valid_data(valid_data_path, batch_size):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(180, fill=(101,
                                                                         101,
                                                                         101)),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.2, 0.2, 0, 0),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.3984,
                                                          0.3984,
                                                          0.3984),
                                                         (0.1328,
                                                          0.1328,
                                                          0.1328))])

    valid_data = EOSAR_Dataset(valid_data_path,
                               'data_splits/valid_list_divided.txt',
                               transform,
                               read_EO=False,
                               image_to_RGB=True)
    valid_data_loader = data.DataLoader(valid_data,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=args.num_workers)

    return valid_data_loader


def load_test_data(test_data_path, batch_size, aug=False):
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.RandomRotation([20,20], fill=(101,
                                                                            101,
                                                                            101)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.3984,
                                                            0.3984,
                                                            0.3984),
                                                            (0.1328,
                                                            0.1328,
                                                            0.1328))])
    for r, ds, fs in os.walk(os.path.join(test_data_path, '-1')):
        test_data_nums = len(fs)
        break
    if test_data_nums == 50:
        print('Do the test on 50 test samples')
        test_list_file = 'data_splits/test_list_temp.txt'
    elif test_data_nums == 826:
        print('Do the test on the full test set')
        test_list_file = 'data_splits/test_list_raw.txt'
    else:
        print('There are {} samples in the "test_images", please use the full test set or the given 50 samples'.format(test_data_nums))
        raise NotImplementedError
    test_data = EOSAR_Dataset(test_data_path,
                              test_list_file,
                              transform,
                              read_EO=False,
                              image_to_RGB=True)
    test_data_loader = data.DataLoader(test_data,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=args.num_workers)

    return test_data_loader, test_data_nums

def read_model(model_name):
    _print(model_name)
    if model_name == 'SwinB':
        backbone = SwinTransformer(img_size=224,
                               patch_size=4,
                               in_chans=3,
                               num_classes=10,
                               embed_dim=128,
                               depths=[2, 2, 18, 2],
                               num_heads=[4, 8, 16, 32],
                               window_size=7,
                               mlp_ratio=4.,
                               qkv_bias=True,
                               qk_scale=None,
                               drop_rate=0.,
                               attn_drop_rate=0.,
                               drop_path_rate=0.1,
                               norm_layer=nn.LayerNorm, 
                               ape=False,
                               patch_norm=True,
                               use_checkpoint=False)
        pretrained_path = 'pretrained_backbone/swin_base_patch4_window7_224_22k.pth'
        backbone.load_state_dict(torch.load(pretrained_path)['model'], strict=False)
        cls_head = Classifier(1024, num_class=10)
        return backbone, cls_head


def train():
    args = parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    _print('train_stage2')

    train_data, train_set = load_train_data(args.train_folder,
                                            args.batch_size,
                                            args.test_folder)
    valid_data = load_valid_data(args.train_folder, args.batch_size)
    test_data, _ = load_test_data(args.test_folder, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    if args.load_path is not None:
        backbone = torch.load(os.path.join(args.load_path, 'last_backbone.pth'))
        cls_head = torch.load(os.path.join(args.load_path, 'last_cls_head.pth'))
    else:
        backbone, cls_head = read_model(args.model_name)

    criterion = nn.CrossEntropyLoss()

    if 'Swin' in args.model_name:
        _print('Use AdamW')
        other_params = []
        woWD_params = []
        swin_group = backbone.named_parameters()
        for name, params in swin_group:
            if 'norm' in name or 'position' in name:
                woWD_params.append(params)
            else:
                other_params.append(params)

        group_woWD = {"params": woWD_params, 'weight_decay': 0.0}
        group_other = {"params": other_params}

        optimizerB = optim.AdamW([group_other, group_woWD], lr=0.00002)
        optimizerC = optim.AdamW(cls_head.parameters(), lr=0.00002)

    else:
        optimizerB = optim.Adam(backbone.parameters(), lr=0.0001)
        optimizerC = optim.Adam(cls_head.parameters(), lr=0.0001)
    backbone.to(device)
    cls_head.to(device)

    best_acc = 0
    select_list = None
    test_distrib = {0:0, 1:0, 2:0, 3:0, 4:0,
                    5:0, 6:0, 7:0, 8:0, 9:0}
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        backbone.eval()
        cls_head.eval()
        select_list = get_select_list_v1(backbone, cls_head, test_data, device, args.sample_num)
        train_set.update_data_list(extra_list = select_list)
        with tqdm(train_data, unit="batch") as tepoch:
            backbone.train()
            cls_head.train()
            for inputs in tepoch:
                # get the inputs
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)

                tepoch.set_description(f"Epoch {epoch}")

                # zero the parameter gradients
                optimizerB.zero_grad()
                optimizerC.zero_grad()

                # forward + get predictions + backward + optimize
                mid_outputs = backbone(inputs['SAR'])
                mid_outputs = mid_outputs.view(mid_outputs.shape[0], -1)
                outputs = cls_head(mid_outputs)

                a_sim = args.a_sim
                outputs_sim = (mid_outputs @ mid_outputs.permute(1, 0)) / (mid_outputs.shape[1] ** 0.5)
                one_hot_target = nn.functional.one_hot(inputs['Label'], 10).to(torch.float)
                target_sim =  one_hot_target @ one_hot_target.permute(1, 0)
                loss_sim1 = -outputs_sim * target_sim
                loss_sim2 = outputs_sim * (1-target_sim)
                # loss_sim1[loss_sim1 < -1] = -1
                loss_sim2[loss_sim2 < 0.1] = 0.1
                loss_sim = loss_sim1.mean() + loss_sim2.mean()
                loss = criterion(outputs, inputs['Label'])
                loss += a_sim * loss_sim

                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == inputs['Label']).sum().item()
                accuracy = correct / args.batch_size

                loss.backward()
                optimizerB.step()
                optimizerC.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            
            backbone.eval()
            cls_head.eval()
            # Evaluation
            with torch.no_grad():
                labels = []
                preds = []
                for inputs in valid_data:
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(device)
                    outputs = backbone(inputs['SAR'])
                    outputs = cls_head(outputs.view(outputs.shape[0], -1))

                    predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                    preds.append(predictions.unsqueeze(0))
                    labels.append(inputs['Label'].unsqueeze(0))
                
            preds = torch.cat(preds, dim=1).squeeze(0)
            labels = torch.cat(labels, dim=1).squeeze(0)
            correct = (preds == labels).sum()
            accuracy = correct / labels.shape[0]
            _print('\nEpoch: {}'.format(epoch))
            _print('acc: {}'.format(accuracy.item()))
            _print('Valid:')
            acc_line1 = 'cls |'
            acc_line2 = 'acc |'
            acc_line3 = 'num |'
            for class_idx in range(10):
                class_mask = labels == class_idx
                cls_preds = preds[class_mask]
                cls_labels = labels[class_mask]
                cls_correct = (cls_preds == cls_labels).sum()
                cls_accuracy = cls_correct / cls_labels.shape[0]
                acc_line1 += str(class_idx).center(10) + '|'
                acc_line2 += '{:.4f}'.format(cls_accuracy).center(10) + '|'

                pred_num = (preds == class_idx).sum()
                acc_line3 += str(int(pred_num)).center(10) + '|'
            _print(acc_line1)
            _print(acc_line2)
            _print(acc_line3)

            _print('Test:')
            with torch.no_grad():
                labels = []
                preds = []
                for inputs in test_data:
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(device)
                    outputs = backbone(inputs['SAR'])
                    outputs = cls_head(outputs.view(outputs.shape[0], -1))

                    predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                    preds.append(predictions.unsqueeze(0))
                
            preds = torch.cat(preds, dim=1).squeeze(0)
            acc_line1 = 'cls |'
            acc_line3 = 'num |'
            distrib_vary = 0
            for class_idx in range(10):
                acc_line1 += str(class_idx).center(10) + '|'
                pred_num = (preds == class_idx).sum()
                distrib_vary += abs(float(pred_num) - test_distrib[class_idx]) / (test_distrib[class_idx] + 1e-8)
                test_distrib[class_idx] = int(pred_num)
                acc_line3 += str(int(pred_num)).center(10) + '|'
            _print('prediction changing rate: {}'.format(distrib_vary / 10))
            _print(acc_line1)
            _print(acc_line3)

            
            if (epoch + 1) % args.save_freq == 0:
                torch.save(backbone, os.path.join(args.save_path, '{}_backbone.pth'.format(epoch+1)))
                torch.save(cls_head, os.path.join(args.save_path, '{}_cls_head.pth'.format(epoch+1)))

    _print('Finished Training')
    torch.save(backbone, os.path.join(args.save_path, 'last_backbone.pth'))
    torch.save(cls_head, os.path.join(args.save_path, 'last_cls_head.pth'))

def get_select_list_v1(backbone, cls_head, test_data, device, select_num=70):
    idxs = []
    probs = []
    with torch.no_grad():
        for inputs in test_data:
            # get the inputs
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            outputs = backbone(inputs['SAR'])
            outputs = cls_head(outputs.view(outputs.shape[0], -1))
            prob = torch.softmax(outputs, dim=1)
            probs.append(prob.unsqueeze(0))

            idxs += inputs['Idx']

    probs = torch.cat(probs, dim=1).squeeze(0)

    data_per_class = {0:[], 1:[], 2:[], 3:[], 4:[],
                      5:[], 6:[], 7:[], 8:[], 9:[]}
    selected_idxs = []
    top_probs, top_preds = torch.topk(probs, 10, -1)
    for i in range(10):
        top_probs_range = top_probs[:, i]
        top_preds_range = top_preds[:, i]
        top_probs_range, corr_idx = torch.topk(top_probs_range,
                                               top_probs_range.shape[0])
        sort_preds_range = torch.gather(top_preds_range, 0, corr_idx)
        for select_idx, pred, prob in zip(corr_idx, sort_preds_range, top_probs_range):
            idx = idxs[select_idx.item()]
            if (len(data_per_class[pred.item()]) >= select_num
                or idx in selected_idxs):
                continue
            data_per_class[pred.item()].append(idx)
            selected_idxs.append(idx)
    
    new_data = []
    for i in range(10):
        class_list = data_per_class[i]
        for data_idx in class_list:
            new_data.append('_{}'.format(i) + ' ' + data_idx)

    return new_data


def test():
    args = parse_args()
    test_data, test_num = load_test_data(args.test_folder, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print('Load trained model')
    backbone = torch.load(os.path.join(args.save_path, 'last_backbone.pth'))
    cls_head = torch.load(os.path.join(args.save_path, 'last_cls_head.pth'))
    backbone.eval()
    cls_head.eval()

    idxs = []
    preds = []
    # probs = []
    print('Start test')
    start_time = time.time()
    with torch.no_grad():
        for inputs in test_data:
            # get the inputs
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            outputs = backbone(inputs['SAR'])
            outputs = cls_head(outputs.view(outputs.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)
            preds.append(predicted.unsqueeze(0))
            # prob = torch.softmax(outputs, dim=1)
            # probs.append(prob.unsqueeze(0))
            idxs += inputs['Idx']
    pred_time = time.time() - start_time
    time_per_image = pred_time / test_num
    print('testing: {} s/image'.format(time_per_image))
    preds = torch.cat(preds, dim=1).squeeze(0)
    # probs = torch.cat(probs, dim=1).squeeze(0)
    # top_probs, top_preds = torch.topk(probs, 1, -1)
    with open('results/results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'class_id'])
        # for idx, pred, prob in zip(idxs, preds, probs):
        #     writer.writerow([idx, pred.item()] + prob.cpu().numpy().tolist())
        for idx, pred in zip(idxs, preds):
            writer.writerow([idx, pred.item()])


if __name__ == "__main__":
    args = parse_args()
    if args.test_only:
        test()
    else:
        train()
