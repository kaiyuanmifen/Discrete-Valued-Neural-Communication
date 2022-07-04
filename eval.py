import argparse
import torch
import utils
import os
import pickle


from torch.utils import data
import numpy as np
from collections import defaultdict

import modules

import pandas as pd

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str,
					default='checkpoints',
					help='Path to checkpoints.')
parser.add_argument('--num-steps', type=int, default=1,
					help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset', type=str,
					default='data/shapes_eval.h5',
					help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='Disable CUDA training.')



parser.add_argument('--name', type=str, default='none',
					help='Experiment name.')


parser.add_argument('--load_action_dim', action='store_true', default=False,
                    help='load action dim from file')
####argguments for quantization
parser.add_argument('--Quantization', action='store_true', default=False,
					help='if quantize the edge information.')
parser.add_argument('--n_codebook_embedding', type=int, default=512)
parser.add_argument('--codebook_loss_weight', type=float, default=0.25)
parser.add_argument('--Quantization_method',type=str,default="VQVAE",help="method used for quantizaiton")
parser.add_argument('--n_quuantization_segments', type=int, default=1)
parser.add_argument('--TargetTask',type=str,default="None",help="Target task for evaluation")
parser.add_argument('--Quantization_target',type=str,default="None",help="Target embedding for quantization")

parser.add_argument('--ExperimentName',type=str,default="None",help="Experiment name for the round of evaluation")

args_eval = parser.parse_args()#######Be very careful , this args_eval is different from args!
print("name",args_eval.name)


meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, 'model.pt')

args = pickle.load(open(meta_file, 'rb'))['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 10
args.dataset = args_eval.dataset
args.seed = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.PathDataset(
	hdf5_file=args.dataset, path_length=args_eval.num_steps)
eval_loader = data.DataLoader(
	dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# Get data sample
obs = eval_loader.__iter__().next()[0]
input_shape = obs[0][0].size()




###load action dim if not provided 
if args_eval.load_action_dim:
    print("loading action_dim from file")
    Vec=pd.read_csv("data/actionspace.csv",header=None)

    action_dim=Vec[Vec.iloc[:,1]==args.dataset].iloc[:,2].unique()[0]

    action_dim=int(action_dim)
else:
    action_dim=args.action_dim

print("action_dim:"+str(action_dim))



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
	Quantization=args_eval.Quantization,n_codebook_embedding=args_eval.n_codebook_embedding,
	Quantization_method=args_eval.Quantization_method,n_quuantization_segments=args_eval.n_quuantization_segments,
	codebook_loss_weight=args_eval.codebook_loss_weight,
	Quantization_target=args.Quantization_target).to(device)
# 
model.load_state_dict(torch.load(model_file))
model.eval()

# topk = [1, 5, 10]
topk = [1,2,3]
hits_at = defaultdict(int)
num_samples = 0
rr_sum = 0

pred_states = []
next_states = []

with torch.no_grad():
	All_CB_indexes=[]
	loss=0
	for batch_idx, data_batch in enumerate(eval_loader):



		data_batch = [[t.to(
			device) for t in tensor] for tensor in data_batch]
		observations, actions = data_batch

		if observations[0].size(0) != args.batch_size:
			continue

		obs = observations[0]
		next_obs = observations[-1]


		state = model.obj_encoder(model.obj_extractor(obs))
		next_state = model.obj_encoder(model.obj_extractor(next_obs))

		pred_state = state
		for i in range(args_eval.num_steps):
			if args.Quantization==False:
				pred_trans = model.transition_model(pred_state, actions[i])
				CB_diff=0
				CB_indexes=None
				All_CB_indexes="NA"
			if args.Quantization==True:
				pred_trans,CB_diff, CB_indexes= model.transition_model(pred_state, actions[i])

				All_CB_indexes.append(CB_indexes)
			pred_state = pred_state + pred_trans


			

		pred_states.append(pred_state.cpu())
		next_states.append(next_state.cpu())
	


	average_test_loss=loss/len(eval_loader.dataset)
	

	if args.Quantization==True:
		All_CB_indexes=torch.cat(All_CB_indexes,1)

	


	####calculate measurement metrics 
	pred_state_cat = torch.cat(pred_states, dim=0)
	next_state_cat = torch.cat(next_states, dim=0)

	full_size = pred_state_cat.size(0)

	# Flatten object/feature dimensions
	next_state_flat = next_state_cat.view(full_size, -1)
	pred_state_flat = pred_state_cat.view(full_size, -1)

	dist_matrix = utils.pairwise_distance_matrix(
		next_state_flat, pred_state_flat)
	dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
	dist_matrix_augmented = torch.cat(
		[dist_matrix_diag, dist_matrix], dim=1)

	# Workaround to get a stable sort in numpy.
	dist_np = dist_matrix_augmented.numpy()
	indices = []
	for row in dist_np:
		keys = (np.arange(len(row)), row)
		indices.append(np.lexsort(keys))
	indices = np.stack(indices, axis=0)
	indices = torch.from_numpy(indices).long()

	print('Processed {} batches of size {}'.format(
		batch_idx + 1, args.batch_size))

	labels = torch.zeros(
		indices.size(0), device=indices.device,
		dtype=torch.int64).unsqueeze(-1)

	num_samples += full_size
	print('Size of current topk evaluation batch: {}'.format(
		full_size))

	for k in topk:
		match = indices[:, :k] == labels
		num_matches = match.sum()
		hits_at[k] += num_matches.item()

	match = indices == labels
	_, ranks = match.max(1)

	reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
	rr_sum += reciprocal_ranks.sum()

	pred_states = []
	next_states = []

for k in topk:
	print('Hits @ {}: {}'.format(k, hits_at[k] / float(num_samples)))

print('MRR: {}'.format(rr_sum / float(num_samples)))







###A separate thread to get the test loss

dataset = utils.StateTransitionsDataset(hdf5_file=args.dataset)

eval_loader  =data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0,shuffle=True)


step = 0
model.eval()


###calculate test loss

with torch.no_grad():
	test_loss=0
	for batch_idx, data_batch in enumerate(eval_loader):

		data_batch = [tensor.to(device) for tensor in data_batch]

		###not using decoder ( same as in training), diectly calculate loss on laten state
		loss = model.contrastive_loss(*data_batch)
	

		test_loss += loss.item()

	average_test_loss = test_loss / len(eval_loader .dataset)










print('Test loss: {}'.format(average_test_loss))





##Save the evaluationresults

Eval_results_save_folder="Eval_results/"+args_eval.ExperimentName+"/"
if not os.path.exists(Eval_results_save_folder):
	os.makedirs(Eval_results_save_folder)


SavingFile=Eval_results_save_folder+"/"+args_eval.TargetTask+"_eval_results.csv"
k=1
allinfor=[args_eval.name,args.Quantization_target,average_test_loss,
k,hits_at[k] / float(num_samples),float(rr_sum) / float(num_samples),float(CB_diff),All_CB_indexes]
#save record
import csv  
with open (SavingFile,'a') as f:                            
	writer = csv.writer(f)
	writer.writerow(allinfor)

#save the codebooks for analysis
# import os
# if not os.path.exists(Eval_results_save_folder+"/codebooks/"+args_eval.TargetTask+"/"):
#     os.makedirs(Eval_results_save_folder+"/codebooks/"+args_eval.TargetTask+"/")
# torch.save(All_CB_indexes, Eval_results_save_folder+"/codebooks/"+args_eval.TargetTask+"/"+args_eval.name+"_AllCodebooks.pt")
