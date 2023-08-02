import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           initialize_metrics, get_mean_metrics, load_model_DARNet,
                           set_metrics)
import os
import logging
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import warnings
import torchsummary
from thop import profile
import optuna

from convert import BinaryImageConverter

now=datetime.datetime.now().strftime('%m-%d_%H-%M_')
print(now)

warnings.filterwarnings("ignore")

resume_model = '' # Assign pretrained models or not

"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""
Set up environment: define paths, download data, and set device
"""

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=28)


train_loader, val_loader = get_loaders(opt)

"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))
device_ids = [4,5]
device_idx = [num for num in range(len(device_ids))]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(dev)

model = load_model_DARNet(opt, dev)
loss_weight = [0.2,0.2,0.2,0.2,0.2]
model_name = 'DARNet'+str(loss_weight)

torchsummary.summary(model, input_size=[(3, opt.patch_size, opt.patch_size),
                                       (3, opt.patch_size, opt.patch_size)], device='cuda')
input_shape = (1, 3, opt.patch_size, opt.patch_size)
input_data = (torch.randn(*input_shape).to(dev), torch.randn(*input_shape).to(dev))
flops, params = profile(model, inputs=input_data)
print(f"Total FLOPs: {flops / 1e9} G FLOPs")
print(f"Total Parameters: {params / 1e6} M parameters")

model.to(dev)

if resume_model:
    model = torch.load(resume_model)
    model = model.module
    print(f'Resume model:{resume_model}')
    model_name = 'resume'

if True:
    model = torch.nn.DataParallel(model,device_ids=device_idx)

'''
 Set starting values
'''


# def objective (trial):

#     criterion = get_criterion(opt)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) #step_size=8 # step6 gamma 0.5 #0.0002 50 0.5

#     lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log = True)
#     optimizer.param_groups[0]['lr'] = lr  # updates lr for every trial

#     best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
#     logging.info('STARTING training')
#     total_step = -1

#     for epoch in range(opt.epochs):
#         train_metrics = initialize_metrics()
#         val_metrics = initialize_metrics()
#         '''
#         Training
#         '''
#         model.train()
#         logging.info('Training')
#         batch_iter = 0
#         tbar = tqdm(train_loader,ncols=140)
#         trial_num = trial.number
#         for batch_img1, batch_img2, labels in tbar:
#             tbar.set_description("Trial: {} Epoch: {} info ".format(trial_num,epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
#             batch_iter = batch_iter+opt.batch_size
#             total_step += 1

#             batch_img1 = batch_img1.float().to(dev)
#             batch_img2 = batch_img2.float().to(dev)
#             labels = labels.long().to(dev)

#             optimizer.zero_grad()

#             cd_preds = model(batch_img1, batch_img2)

#             cd_loss = criterion(cd_preds, labels,loss_weight)
#             loss = cd_loss
#             loss.backward()
#             optimizer.step()

#             cd_preds = cd_preds[-1]
#             _, cd_preds = torch.max(cd_preds, 1)

#             cd_corrects = (100 *
#                         (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
#                         float(labels.size()[0] * (opt.patch_size**2)))

#             cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
#                                 cd_preds.data.cpu().numpy().flatten(),
#                                 average='binary',
#                                 pos_label=1)

#             train_metrics = set_metrics(train_metrics,
#                                         cd_loss,
#                                         cd_corrects,
#                                         cd_train_report,
#                                         scheduler.get_lr())

#             mean_train_metrics = get_mean_metrics(train_metrics)

#             for k, v in mean_train_metrics.items():
#                 writer.add_scalars(str(k), {'train': v}, total_step)

#             del batch_img1, batch_img2, labels

#         scheduler.step()
#         logging.info("EPOCH {} TRAIN METRICS".format(epoch)+str({k: round(v,6) for k,
#                                                             v in mean_train_metrics.items()}))

#         '''
#         Validation
#         '''
#         model.eval()
#         with torch.no_grad():
#             for batch_img1, batch_img2, labels in val_loader:
#                 # Set variables for training
#                 batch_img1 = batch_img1.float().to(dev)
#                 batch_img2 = batch_img2.float().to(dev)
#                 labels = labels.long().to(dev)

#                 # Get predictions and calculate loss
#                 cd_preds = model(batch_img1, batch_img2)
#                 cd_loss = criterion(cd_preds, labels, loss_weight)

#                 cd_preds = cd_preds[-1]
#                 _, cd_preds = torch.max(cd_preds, 1)

#                 # Calculate and log other batch metrics
#                 cd_corrects = (100 *
#                             (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
#                             float(labels.size()[0] * (opt.patch_size**2)))

#                 cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
#                                     cd_preds.data.cpu().numpy().flatten(),
#                                     average='binary',
#                                     pos_label=1)

#                 val_metrics = set_metrics(val_metrics,
#                                         cd_loss,
#                                         cd_corrects,
#                                         cd_val_report,
#                                         scheduler.get_lr())

#                 # log the batch mean metrics
#                 mean_val_metrics = get_mean_metrics(val_metrics)

#                 for k, v in mean_train_metrics.items():
#                     writer.add_scalars(str(k), {'val': v}, total_step)

#                 # clear batch variables from memory
#                 del batch_img1, batch_img2, labels

#             logging.info("EPOCH {} VALIDATION METRICS".format(epoch) + str({k: round(v,6) for k,
#                                                                     v in mean_val_metrics.items()}))

#             """
#             Store the weights of good epochs based on validation results
#             """
#             # if (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']):

#             #     # Insert training and epoch information to metadata dictionary
#             #     logging.info('update the model')
#             #     metadata['validation_metrics'] = mean_val_metrics

#             #     # Save model and log
#             #     if not os.path.exists('./tmp'+'/'+now+model_name):
#             #         os.makedirs('./tmp'+'/'+now+model_name)
#             #     with open('./tmp'+'/'+now+model_name+'/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
#             #         json.dump(metadata, fout)

#             #     torch.save(model, './tmp'+'/'+now+model_name+'/metadata_epoch_'+str(epoch)+'.pt')

#             #     # comet.log_asset(upload_metadata_file_path)
#             #     best_metrics = mean_val_metrics


#             print('An epoch finished.')
#     writer.close()  # close tensor board
#     print('Training Finished!')

#     return mean_val_metrics['cd_f1scores']


# """
# Setting up the Optuna study
# """
# study = optuna.create_study(direction='maximize', study_name = 'DARNet-CD Hyper-param Tuning')

# n_trials = 5
# study.optimize(objective, n_trials=n_trials)

# best_param = study.best_params
# print("Best Parameters: ", best_param)





"""
Testing with Best Optuna Parameter
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')
total_step = -1

criterion = get_criterion(opt)
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) #step_size=8 # step6 gamma 0.5 #0.0002 50 0.5
total_step = -1

# best_lr = best_param['learning_rate']
# optimizer.param_groups[0]['lr'] = best_lr
epochs = 70
for epoch in range(epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()
    '''
    Training
    '''
    model.train()
    logging.info('Training')
    batch_iter = 0
    tbar = tqdm(train_loader,ncols=140)

    for batch_img1, batch_img2, labels in tbar:
        tbar.set_description("Epoch: {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        optimizer.zero_grad()

        cd_preds = model(batch_img1, batch_img2)
        print(f'cd_preds size {cd_preds.size}')

        cd_loss = criterion(cd_preds, labels,loss_weight)
        loss = cd_loss
        loss.backward()
        optimizer.step()

        cd_preds = cd_preds[-1]
        print(f'cd_preds size2 {cd_preds.size}')

        _, cd_preds = torch.max(cd_preds, 1)
        print(f'cd_preds size3 {cd_preds.size}')

        predicted = cd_preds
        cd_corrects = (100 *
                    (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                    float(labels.size()[0] * (opt.patch_size**2)))

        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                            cd_preds.data.cpu().numpy().flatten(),
                            average='binary',
                            pos_label=1)

        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    scheduler.get_lr())

        mean_train_metrics = get_mean_metrics(train_metrics)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        del batch_img1, batch_img2, labels

    #print(predicted.shape)
    print(predicted)
    pred_img = BinaryImageConverter(predicted)
    pred_img.convert_and_save_images('/app/DARNet/Images/')
    scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch)+str({k: round(v,6) for k,
                                                        v in mean_train_metrics.items()}))

    '''
    Validation
    '''
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader:
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Get predictions and calculate loss
            cd_preds = model(batch_img1, batch_img2)

            cd_loss = criterion(cd_preds, labels, loss_weight)

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics
            cd_corrects = (100 *
                        (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                        float(labels.size()[0] * (opt.patch_size**2)))

            cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                cd_preds.data.cpu().numpy().flatten(),
                                average='binary',
                                pos_label=1)

            val_metrics = set_metrics(val_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_val_report,
                                    scheduler.get_lr())

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)

            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'val': v}, total_step)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        logging.info("EPOCH {} VALIDATION METRICS".format(epoch) + str({k: round(v,6) for k,
                                                                v in mean_val_metrics.items()}))

        """
        Store the weights of good epochs based on validation results
        """
        if (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']):

            # Insert training and epoch information to metadata dictionary
            logging.info('update the model')
            metadata['validation_metrics'] = mean_val_metrics

            # Save model and log
            if not os.path.exists('./tmp'+'/'+now+model_name):
                os.makedirs('./tmp'+'/'+now+model_name)
            with open('./tmp'+'/'+now+model_name+'/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(model, './tmp'+'/'+now+model_name+'/metadata_epoch_'+str(epoch)+'.pt')

            # comet.log_asset(upload_metadata_file_path)
            best_metrics = mean_val_metrics


        print('An epoch finished.')
writer.close()  # close tensor board
print('Training Finished!')