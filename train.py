import argparse
import torch
import utils
import datetime
import os
import pickle

import numpy as np
import logging

from torch.utils import data
import torch.nn.functional as F

import modules

import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--encoder', type=str, default='small',
                    help='Object extrator CNN size (e.g., `small`).')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=2,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
                    help='Apply same action to all object slots.')

parser.add_argument('--decoder', action='store_true', default=False,
                    help='Train model using decoder and pixel-based loss.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=1,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_train.h5',
                    help='Path to replay buffer.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')

parser.add_argument('--load_action_dim', action='store_true', default=False,
                    help='load action dim from file')

####argguments for quantization
parser.add_argument('--Quantization', action='store_true', default=False,
                    help='if quantize the edge information.')
parser.add_argument('--n_codebook_embedding', type=int, default=128)
parser.add_argument('--codebook_loss_weight', type=float, default=1)
parser.add_argument('--Quantization_method',type=str,default="None",help="method used for quantizaiton")
parser.add_argument('--n_quuantization_segments', type=int, default=1)
parser.add_argument('--Quantization_target',type=str,default="None",help="Target embedding for quantization")

###
parser.add_argument('--RatioDataForTraining', type=float, default=1.,
                    help='ratio of data for training')



args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

exp_counter = 0
save_folder = '{}/{}/'.format(args.save_folder, exp_name)



if not os.path.exists(save_folder):
    os.makedirs(save_folder)
meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')
log_file = os.path.join(save_folder, 'log.txt')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))
print = logger.info

pickle.dump({'args': args}, open(meta_file, "wb"))

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.StateTransitionsDataset(
    hdf5_file=args.dataset)

###split into validation and train set
dataset_size = len(dataset)
indices = list(range(dataset_size))
validation_split = .2
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

###only use certain percetnage of data for training (setting)
train_indices=train_indices[1:int(args.RatioDataForTraining*len(train_indices))]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, num_workers=4,sampler=train_sampler)


Train_set_size=int(dataset_size*(1-validation_split)*args.RatioDataForTraining)
Validation_set_size=int(dataset_size*validation_split)

print("total samples size:"+str(dataset_size))
#print("used for training"+str(len(train_loader.dataset)))
print(str(100*args.RatioDataForTraining)+"percent of total data used for training")
print("training set size:"+str(Train_set_size))
print("validation set size:"+str(Validation_set_size))

validation_loader =data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4,
                                     sampler=valid_sampler)


###load action dim if not provided 
if args.load_action_dim:
    print("loading action_dim from file")
    Vec=pd.read_csv("data/actionspace.csv",header=None)

    action_dim=Vec[Vec.iloc[:,1]==args.dataset].iloc[:,2].unique()[0]

    action_dim=int(action_dim)
else:
    action_dim=args.action_dim

print("action_dim:"+str(action_dim))

# Get data sample
obs = train_loader.__iter__().next()[0]
input_shape = obs[0].size()

model = modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=action_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    encoder=args.encoder,
    Quantization=args.Quantization,n_codebook_embedding=args.n_codebook_embedding,
    Quantization_method=args.Quantization_method,n_quuantization_segments=args.n_quuantization_segments,
    codebook_loss_weight=args.codebook_loss_weight,
    Quantization_target=args.Quantization_target).to(device)

model.apply(utils.weights_init)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate)

if args.decoder:
    if args.encoder == 'large':
        decoder = modules.DecoderCNNLarge(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    elif args.encoder == 'medium':
        decoder = modules.DecoderCNNMedium(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    elif args.encoder == 'small':
        decoder = modules.DecoderCNNSmall(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    decoder.apply(utils.weights_init)
    optimizer_dec = torch.optim.Adam(
        decoder.parameters(),
        lr=args.learning_rate)


# Train model.
print('Starting model training...')
step = 0
best_loss = 1e9

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0

    for batch_idx, data_batch in enumerate(train_loader):

            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()

            if args.decoder:
                optimizer_dec.zero_grad()
                obs, action, next_obs = data_batch
                objs = model.obj_extractor(obs)
                state = model.obj_encoder(objs)

                rec = torch.sigmoid(decoder(state))
                loss = F.binary_cross_entropy(
                    rec, obs, reduction='sum') / obs.size(0)

                next_state_pred = state + model.transition_model(state, action)
                next_rec = torch.sigmoid(decoder(next_state_pred))
                next_loss = F.binary_cross_entropy(
                    next_rec, next_obs,
                    reduction='sum') / obs.size(0)
                loss += next_loss
            else:
                loss = model.contrastive_loss(*data_batch)
                

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if args.decoder:
                optimizer_dec.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Epoch: {} [{}/{} ({:.0f}%)]\t training Loss: {:.6f}'.format(
                        epoch, batch_idx * len(data_batch[0]),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / len(data_batch[0])))

            step += 1

    avg_loss = train_loss / Train_set_size


    ###calculate validation loss

    with torch.no_grad():
        validation_loss=0
        for batch_idx, data_batch in enumerate(validation_loader):

                data_batch = [tensor.to(device) for tensor in data_batch]
      
                if args.decoder:
                    obs, action, next_obs = data_batch
                    objs = model.obj_extractor(obs)
                    state = model.obj_encoder(objs)
                    rec = torch.sigmoid(decoder(state))
                    loss = F.binary_cross_entropy(
                        rec, obs, reduction='sum') / obs.size(0)

                    next_state_pred = state + model.transition_model(state, action)
                    next_rec = torch.sigmoid(decoder(next_state_pred))
                    next_loss = F.binary_cross_entropy(
                        next_rec, next_obs,
                        reduction='sum') / obs.size(0)
                    loss += next_loss
                else:
                    loss = model.contrastive_loss(*data_batch)

                validation_loss += loss.item()

        avg_validation_loss = validation_loss / Validation_set_size

    ##pring out results
    print('====> Epoch: {} Average training loss: {:.6f},Average validation loss: {:.6f}'.format(
        epoch, avg_loss,avg_validation_loss))


    import csv   
    fields=[epoch, avg_loss,avg_validation_loss]

    with open(os.path.join(save_folder, 'Logs.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


    if avg_validation_loss < best_loss:
        best_loss = avg_validation_loss
        torch.save(model.state_dict(), model_file)

torch.save(model.state_dict(), os.path.join(save_folder, 'Final_model.pt'))



