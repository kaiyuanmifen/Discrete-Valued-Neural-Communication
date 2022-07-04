import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import distributed as dist_fn

import torch
from torch import nn, einsum
import torch.nn.functional as F

from scipy.cluster.vq import kmeans2
import random
#from torchviz import make_dot



class Quantize(nn.Module):
	"""
	Neural Discrete Representation Learning, van den Oord et al. 2017
	https://arxiv.org/abs/1711.00937
	Follows the original DeepMind implementation
	https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
	https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
	"""
	def __init__(self, dim, n_embed, groups,Using_projection_layer=False):
		super().__init__()

		num_hiddens=dim
		embedding_dim=dim
		self.embedding_dim = embedding_dim
		self.n_embed = n_embed
		self.groups = groups

		self.kld_scale = 10.0


		self.Using_projection_layer=Using_projection_layer #if a projection layer should be used after quantization

		self.proj = nn.Linear(embedding_dim, embedding_dim)
		
		self.embed = nn.Embedding(n_embed, embedding_dim//groups)

		self.register_buffer('data_initialized', torch.zeros(1))

	def forward(self, z):
		#####input is batch size (B)X Number of units Xembedding size

		B, N,D= z.size()
		#W = 1

		# project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
		#z_e = self.proj(z)
		#z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
		#z_e = z.reshape((B, H, self.groups, self.embedding_dim//self.groups)).reshape((B, H*self.groups, self.embedding_dim//self.groups))
		
		z_e = z.reshape((B, N, self.groups, self.embedding_dim//self.groups)).reshape((B*N*self.groups, self.embedding_dim//self.groups))
		
		flatten = z_e.reshape(-1, self.embedding_dim//self.groups)

		#flatten = flatten.reshape((flatten.shape[0], self.groups, self.embedding_dim//self.groups)).reshape((flatten.shape[0] * self.groups, self.embedding_dim//self.groups))

		# DeepMind def does not do this but I find I have to... ;\
		if self.training and self.data_initialized.item() == 0:
			print('running kmeans!!') # data driven initialization for the embeddings
			rp = torch.randint(0,flatten.size(0),(20000,))###batch size is small in RIM, up sampling here for clustering 
			kd = kmeans2(flatten[rp].data.cpu().numpy(), self.n_embed, minit='points')
			self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
			self.data_initialized.fill_(1)
			# TODO: this won't work in multi-GPU setups

		
		dist = (
			flatten.pow(2).sum(1, keepdim=True)
			- 2 * flatten @ self.embed.weight.t()
			+ self.embed.weight.pow(2).sum(1, keepdim=True).t()
		)

		
		_, ind = (-dist).max(1)
		
		#ind = ind.view(B,self.groups)
		
		# vector quantization cost that trains the embedding vectors
		z_q = self.embed_code(ind) # (B, H, W, C)
		commitment_cost = 0.25
		diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
		diff *= self.kld_scale

		z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass

		z_q = z_q.reshape((B,N,self.groups, self.embedding_dim//self.groups)).reshape((B,N,self.embedding_dim))

		if self.Using_projection_layer:
			###linear projection of quantized embeddings after quantization
			z_q_projected=[]
			for i in range(N):
				z_q_projected.append(self.proj(z_q[:,i,:]))

			z_q=torch.stack(z_q_projected,1)


		ind = ind.reshape(B,N,self.groups).view(N,B*self.groups)
		
		if random.uniform(0,1) < 0.000001:
			print('encoded ind',ind.view(N,B,self.groups)[:,0,:])
			print('before', z[0])
			print('after', z_q[0])
			print('extra loss on layer', diff)
		
		return z_q, diff, ind.view(N,B,self.groups)


	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.weight)






class Quantize_separate(nn.Module):
	

	"""
	This version uses different codebook for each group ( segment)
	Neural Discrete Representation Learning, van den Oord et al. 2017
	https://arxiv.org/abs/1711.00937
	Follows the original DeepMind implementation
	https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
	https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
	"""
	def __init__(self, dim, n_embed, groups,Using_projection_layer=False):
		super().__init__()

		num_hiddens=dim
		embedding_dim=dim
		self.embedding_dim = embedding_dim
		self.n_embed = n_embed
		self.groups = groups

		self.kld_scale = 10.0



		self.Using_projection_layer=Using_projection_layer #if a projection layer should be used after quantization

		self.proj = nn.Linear(embedding_dim, embedding_dim)

		self.embed = nn.Embedding(n_embed*groups, embedding_dim//groups)

		self.register_buffer('data_initialized', torch.zeros(1))
		#print("using separated embedding for each group")

	def forward(self, z):
		#####input is batch size (B)X Number of units Xembedding size

		B, N,D= z.size()
		#W = 1

		# project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
		#z_e = self.proj(z)
		#z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
		#z_e = z.reshape((B, H, self.groups, self.embedding_dim//self.groups)).reshape((B, H*self.groups, self.embedding_dim//self.groups))
		
		z_e = z.reshape((B, N, self.groups, self.embedding_dim//self.groups)).reshape((B*N*self.groups, self.embedding_dim//self.groups))
		
		flatten = z_e.reshape(-1, self.embedding_dim//self.groups)
		
		#flatten = flatten.reshape((flatten.shape[0], self.groups, self.embedding_dim//self.groups)).reshape((flatten.shape[0] * self.groups, self.embedding_dim//self.groups))

		# DeepMind def does not do this but I find I have to... ;\
		if self.training and self.data_initialized.item() == 0:
			print('running kmeans!!') # data driven initialization for the embeddings
			rp = torch.randint(0,flatten.size(0),(20000,))###batch size is small in RIM, up sampling here for clustering 
			kd = kmeans2(flatten[rp].data.cpu().numpy(), self.n_embed*self.groups, minit='points')
			self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
			self.data_initialized.fill_(1)
			# TODO: this won't work in multi-GPU setups

		
		dist = (
			flatten.pow(2).sum(1, keepdim=True)
			- 2 * flatten @ self.embed.weight.t()
			+ self.embed.weight.pow(2).sum(1, keepdim=True).t()
		)##size B X N X (N_embedXGroups)

		dist=dist.reshape((B*N,self.groups,self.groups, self.n_embed))#each group use separate codebook
		ind=[]	
		for i in range(self.groups):
			_, ind_vec = (-dist[:,i,i,:]).max(1)
			ind.append(ind_vec)
		ind=torch.cat(ind)
		
		
		#ind = ind.view(B,self.groups)
		
		# vector quantization cost that trains the embedding vectors
		z_q = self.embed_code(ind) # (B, H, W, C)
		commitment_cost = 0.25
		diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
		diff *= self.kld_scale

		z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass

		z_q = z_q.reshape((B,N,self.groups, self.embedding_dim//self.groups)).reshape((B,N,self.embedding_dim))

		if self.Using_projection_layer:
			###linear projection of quantized embeddings after quantization
			z_q_projected=[]
			for i in range(N):
				z_q_projected.append(self.proj(z_q[:,i,:]))

			z_q=torch.stack(z_q_projected,1)



		ind = ind.reshape(B,N,self.groups).view(N,B*self.groups)
		
		if random.uniform(0,1) < 0.000001:
			print('encoded ind',ind.view(N,B,self.groups)[:,0,:])
			print('before', z[0])
			print('after', z_q[0])
			print('extra loss on layer', diff)

		return z_q, diff, ind.view(N,B,self.groups)


	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.weight)







class Quantize_conditional(nn.Module):
	"""
	this verison uses the first segment as indicator for interaciton
	Neural Discrete Representation Learning, van den Oord et al. 2017
	https://arxiv.org/abs/1711.00937
	Follows the original DeepMind implementation
	https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
	https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
	"""
	def __init__(self, dim, n_embed, groups,Using_projection_layer=False):
		super().__init__()

		num_hiddens=dim
		embedding_dim=dim
		self.embedding_dim = embedding_dim
		self.n_embed = n_embed
		self.groups = groups

		self.kld_scale = 10.0


		self.Using_projection_layer=Using_projection_layer #if a projection layer should be used after quantization

		self.proj = nn.Linear(embedding_dim, embedding_dim)

		self.interaction_classidier=nn.Linear(embedding_dim, 2)##classifier if  two nodes should interaction, will be connect to a straight-through gumble softmax
		
		self.embed = nn.Embedding(n_embed, embedding_dim//groups)

		self.register_buffer('data_initialized', torch.zeros(1))

	def forward(self, z):
		#####input is batch size (B)X Number of units Xembedding size

		B, N,D= z.size()
		#W = 1

		# project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
		#z_e = self.proj(z)
		#z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
		#z_e = z.reshape((B, H, self.groups, self.embedding_dim//self.groups)).reshape((B, H*self.groups, self.embedding_dim//self.groups))
		
		z_e = z.reshape((B, N, self.groups, self.embedding_dim//self.groups)).reshape((B*N*self.groups, self.embedding_dim//self.groups))
		
		flatten = z_e.reshape(-1, self.embedding_dim//self.groups)

		#flatten = flatten.reshape((flatten.shape[0], self.groups, self.embedding_dim//self.groups)).reshape((flatten.shape[0] * self.groups, self.embedding_dim//self.groups))

		# DeepMind def does not do this but I find I have to... ;\
		if self.training and self.data_initialized.item() == 0:
			print('running kmeans!!') # data driven initialization for the embeddings
			rp = torch.randint(0,flatten.size(0),(20000,))###batch size is small in RIM, up sampling here for clustering 
			kd = kmeans2(flatten[rp].data.cpu().numpy(), self.n_embed, minit='points')
			self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
			self.data_initialized.fill_(1)
			# TODO: this won't work in multi-GPU setups

		
		dist = (
			flatten.pow(2).sum(1, keepdim=True)
			- 2 * flatten @ self.embed.weight.t()
			+ self.embed.weight.pow(2).sum(1, keepdim=True).t()
		)

		
		_, ind = (-dist).max(1)
		
		#ind = ind.view(B,self.groups)
		
		# vector quantization cost that trains the embedding vectors
		z_q = self.embed_code(ind) # (B, H, W, C)
		commitment_cost = 0.25
		diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
		diff *= self.kld_scale

		z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass


		z_q = z_q.reshape((B,N,self.groups, self.embedding_dim//self.groups))
		
		z_q = z_q.reshape((B,N,self.embedding_dim))

		###this part tells whether two node should interact
		
		interaction_indicator=[]
		for i in range(N):
			Interaction_or_not=self.interaction_classidier(z_q[:,i,:])#use the first group/segment to determine existance of itneraction
			Interaction_or_not=torch.mean(Interaction_or_not,dim=0,keepdim=True)
			Interaction_or_not= F.gumbel_softmax(Interaction_or_not, dim=1, hard=True)#straight-through gumbel
			interaction_indicator.append(Interaction_or_not.flatten())

		interaction_indicator=torch.stack(interaction_indicator)
		interaction_indicator=interaction_indicator[:,0].to(torch.float)#use the first value as interaction indivator
		interaction_indicator=interaction_indicator.reshape((1,interaction_indicator.shape[0],1))



		if self.Using_projection_layer:
			###linear projection of quantized embeddings after quantization
			z_q_projected=[]
			for i in range(N):
				z_q_projected.append(self.proj(z_q[:,i,:]))

			z_q=torch.stack(z_q_projected,1)



		#convert all z_q values where coressponding interaction_indcator is zero to zero
		z_q = z_q*interaction_indicator




		ind = ind.reshape(B,N,self.groups).view(N,B*self.groups)
	

		if random.uniform(0,1) < 0.000001:
			print('encoded ind',ind.view(N,B,self.groups)[:,0,:])
			print('before', z[0])
			print('after', z_q[0])
			print('extra loss on layer', diff)

		return z_q, diff,  ind.view(N,B,self.groups)


	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.weight)






class Quantize_conditional_separate(nn.Module):
	

	"""
	This version use one of the segment of to determine if there should be an interaction
	This version uses different codebook for each group ( segment)
	Neural Discrete Representation Learning, van den Oord et al. 2017
	https://arxiv.org/abs/1711.00937
	Follows the original DeepMind implementation
	https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
	https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
	"""
	def __init__(self, dim, n_embed, groups,Using_projection_layer=False):
		super().__init__()

		num_hiddens=dim
		embedding_dim=dim
		self.embedding_dim = embedding_dim
		self.n_embed = n_embed
		self.groups = groups

		self.kld_scale = 10.0



		self.Using_projection_layer=Using_projection_layer #if a projection layer should be used after quantization

		self.proj = nn.Linear(embedding_dim, embedding_dim)

		self.interaction_classidier=nn.Linear(embedding_dim, 2)##classifier if  two nodes should interaction, will be connect to a straight-through gumble softmax

		self.embed = nn.Embedding(n_embed*groups, embedding_dim//groups)

		self.register_buffer('data_initialized', torch.zeros(1))
		#print("using separated embedding for each group")

	def forward(self, z):
		
		#####input is batch size (B)X Number of units Xembedding size

		B, N,D= z.size()
		#W = 1

		# project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
		#z_e = self.proj(z)
		#z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
		#z_e = z.reshape((B, H, self.groups, self.embedding_dim//self.groups)).reshape((B, H*self.groups, self.embedding_dim//self.groups))
		
		z_e = z.reshape((B, N, self.groups, self.embedding_dim//self.groups)).reshape((B*N*self.groups, self.embedding_dim//self.groups))
		
		flatten = z_e.reshape(-1, self.embedding_dim//self.groups)
		
		#flatten = flatten.reshape((flatten.shape[0], self.groups, self.embedding_dim//self.groups)).reshape((flatten.shape[0] * self.groups, self.embedding_dim//self.groups))

		# DeepMind def does not do this but I find I have to... ;\
		if self.training and self.data_initialized.item() == 0:
			print('running kmeans!!') # data driven initialization for the embeddings
			rp = torch.randint(0,flatten.size(0),(20000,))###batch size is small in RIM, up sampling here for clustering 
			kd = kmeans2(flatten[rp].data.cpu().numpy(), self.n_embed*self.groups, minit='points')
			self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
			self.data_initialized.fill_(1)
			# TODO: this won't work in multi-GPU setups

		
		dist = (
			flatten.pow(2).sum(1, keepdim=True)
			- 2 * flatten @ self.embed.weight.t()
			+ self.embed.weight.pow(2).sum(1, keepdim=True).t()
		)##size B X N X (N_embedXGroups)

		dist=dist.reshape((B*N,self.groups,self.groups, self.n_embed))#each group use separate codebook
		ind=[]	
		for i in range(self.groups):
			_, ind_vec = (-dist[:,i,i,:]).max(1)
			ind.append(ind_vec)
		ind=torch.cat(ind)
		
		
		#ind = ind.view(B,self.groups)
		
		# vector quantization cost that trains the embedding vectors
		z_q = self.embed_code(ind) # (B, H, W, C)
		
		commitment_cost = 0.25
		diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
		diff *= self.kld_scale

		z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass

		

		z_q = z_q.reshape((B,N,self.groups, self.embedding_dim//self.groups))
		
		z_q = z_q.reshape((B,N,self.embedding_dim))

		###this part tells whether two node should interact
		
		interaction_indicator=[]
		for i in range(N):
			Interaction_or_not=self.interaction_classidier(z_q[:,i,:])#use the first group/segment to determine existance of itneraction
			Interaction_or_not=torch.mean(Interaction_or_not,dim=0,keepdim=True)
			Interaction_or_not= F.gumbel_softmax(Interaction_or_not, dim=1, hard=True)#straight-through gumbel
			interaction_indicator.append(Interaction_or_not.flatten())

		interaction_indicator=torch.stack(interaction_indicator)
		interaction_indicator=interaction_indicator[:,0].to(torch.float)#use the first value as interaction indivator
		interaction_indicator=interaction_indicator.reshape((1,interaction_indicator.shape[0],1))







		if self.Using_projection_layer:
			###linear projection of quantized embeddings after quantization
			z_q_projected=[]
			for i in range(N):
				z_q_projected.append(self.proj(z_q[:,i,:]))

			z_q=torch.stack(z_q_projected,1)



		#convert all z_q values where coressponding interaction_indcator is zero to zero
		z_q = z_q*interaction_indicator




		ind = ind.reshape(B,N,self.groups).view(N,B*self.groups)
	

		if random.uniform(0,1) < 0.000001:
			print('encoded ind',ind.view(N,B,self.groups)[:,0,:])
			print('before', z[0])
			print('after', z_q[0])
			print('extra loss on layer', diff)

		return z_q, diff, ind.view(N,B,self.groups)
		



	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.weight)







class Quantize_onehot_VQVAE(nn.Module):
	"""
	Neural Discrete Representation Learning, van den Oord et al. 2017
	https://arxiv.org/abs/1711.00937
	Follows the original DeepMind implementation
	https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
	https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
	"""
	def __init__(self, dim, n_embed, groups):
		super().__init__()

		num_hiddens=dim
		embedding_dim=dim
		self.embedding_dim = embedding_dim
		self.n_embed = n_embed
		self.groups = groups

		self.kld_scale = 10.0

		#self.proj = nn.Linear(num_hiddens, embedding_dim)
		self.embed = nn.Embedding(n_embed, embedding_dim//groups)

		self.register_buffer('data_initialized', torch.zeros(1))

		self.temperature=0.5

	def forward(self, z):
		#####input is batch size (B)X Number of units Xembedding size

		B, N,D= z.size()
		#W = 1

		# project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
		#z_e = self.proj(z)
		#z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
		#z_e = z.reshape((B, H, self.groups, self.embedding_dim//self.groups)).reshape((B, H*self.groups, self.embedding_dim//self.groups))
		
		z_e = z.reshape((B, N, self.groups, self.embedding_dim//self.groups)).reshape((B*N*self.groups, self.embedding_dim//self.groups))
		
		flatten = z_e.reshape(-1, self.embedding_dim//self.groups)

		#flatten = flatten.reshape((flatten.shape[0], self.groups, self.embedding_dim//self.groups)).reshape((flatten.shape[0] * self.groups, self.embedding_dim//self.groups))

		# DeepMind def does not do this but I find I have to... ;\
		if self.training and self.data_initialized.item() == 0:
			print('running kmeans!!') # data driven initialization for the embeddings
			rp = torch.randint(0,flatten.size(0),(20000,))###batch size is small in RIM, up sampling here for clustering 
			kd = kmeans2(flatten[rp].data.cpu().numpy(), self.n_embed, minit='points')
			self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
			self.data_initialized.fill_(1)
			# TODO: this won't work in multi-GPU setups

		
		dist = (
			flatten.pow(2).sum(1, keepdim=True)
			- 2 * flatten @ self.embed.weight.t()
			+ self.embed.weight.pow(2).sum(1, keepdim=True).t()
		)


		embed_onehot= F.gumbel_softmax(-dist, tau=self.temperature, dim=1, hard=True)#straight-through gumbel
			
		embed_onehot.type(flatten.dtype)
		
		
		ind  = embed_onehot.argmax(dim=1)
		
		
		#ind = ind.view(B,self.groups)
		
		# vector quantization cost that trains the embedding vectors
		z_q = self.embed_code(ind) # (B, H, W, C)
		commitment_cost = 0.25
		diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
		diff *= self.kld_scale
		
		z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass

		z_q = z_q.reshape((B,N,self.groups, self.embedding_dim//self.groups)).reshape((B,N,self.embedding_dim))

		ind = ind.reshape(B,N,self.groups).view(N,B*self.groups)
		
		embed_onehot=embed_onehot.reshape((B,N,self.groups,self.n_embed)).reshape((B,N,self.groups*self.n_embed))
	

		
		if random.uniform(0,1) < 0.000001:
			print('encoded ind',ind.view(N,B,self.groups)[:,0,:])
			print('before', z[0])
			print('after', z_q[0])
			print('extra loss on layer', diff)

		return embed_onehot, diff, ind


	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.weight)







class Quantize_onehot_gumbel(nn.Module):
	"""
	Neural Discrete Representation Learning, van den Oord et al. 2017
	https://arxiv.org/abs/1711.00937
	Follows the original DeepMind implementation
	https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
	https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
	"""
	def __init__(self, dim, n_embed, groups):
		super().__init__()

		self.dim=dim
		
		self.n_embed = n_embed
		self.groups = groups

		self.temperature = 0.5##for gumbler softmax

		self.proj = nn.Linear(self.dim//self.groups, n_embed)


	def forward(self, z):
		#####input is batch size (B)X Number of units Xembedding size

		B, N,D= z.size()
		#W = 1

		# project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
		#z_e = self.proj(z)
		#z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
		#z_e = z.reshape((B, H, self.groups, self.embedding_dim//self.groups)).reshape((B, H*self.groups, self.embedding_dim//self.groups))
		
		z_e = z.reshape((B, N, self.groups, self.dim//self.groups)).reshape((B*N*self.groups, self.dim//self.groups))
		
	

		z_e=self.proj(z_e)

		embed_onehot= F.gumbel_softmax(z_e, tau=self.temperature, dim=1, hard=True)#straight-through gumbel
			
		ind  = embed_onehot.argmax(dim=1)

		diff=torch.tensor(0.0)#not used , placer holder
		#flatten = flatten.reshape((flatten.shape[0], self.groups, self.embedding_dim//self.groups)).reshape((flatten.shape[0] * self.groups, self.embedding_dim//self.groups))
		embed_onehot=embed_onehot.reshape((B,N,self.groups,self.n_embed)).reshape((B,N,self.groups*self.n_embed))
	

		ind = ind.reshape(B,N,self.groups).view(N,B*self.groups)
		if random.uniform(0,1) < 0.000001:
			print('encoded ind',ind.view(N,B,self.groups)[:,0,:])

		return embed_onehot, diff, ind


	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.weight)





if __name__ == "__main__":


	#######these codes are simply used for debugging and quality control purpose 
	


	# class Encoder(nn.Module):
	# 	def __init__(
	# 		self,
	# 		inputSize=213,
	# 		embed_dim=256,
	# 		n_embed=512,
	# 		decay=0.99,
	# 		n_segments=4
	# 	):
	# 		super().__init__()

	# 		self.enc = nn.Linear(inputSize,embed_dim)
	# 		self.quantize= Quantize(embed_dim,n_embed,groups=n_segments)
			
	# 	def forward(self, input):
	# 		x=self.enc(input)
			
	# 		quant_x,diff,CB_index =self.quantize(x)
		   
	# 		return quant_x, diff,CB_index


	# Model=Encoder()
	

	# for name, param in Model.named_parameters():
	# 	if param.requires_grad:
	# 		print (name, param.data.shape)

	# input_dim=213
	# embed_dim=256
	# n_embed=512
	# N_units=6
	# AllSampleSize=100000
	# batchSize=63

	# Input=torch.randn((AllSampleSize,N_units,input_dim))



	# optimizer = torch.optim.Adam(Model.parameters())
	# Y=torch.randn((AllSampleSize,N_units,embed_dim))
	# #Y=torch.randint(0,n_embed,(AllSampleSize,))
	# #Y=torch.randint(0,2,(1000,n_embed)).float()
	# #Y=torch.randn((AllSampleSize,n_embed))

	# criterion = nn.MSELoss()
	# #criterion = nn.CrossEntropyLoss()
	# ###visualize computation graph

	# #dot=make_dot(Y.mean(), params=dict(Model.named_parameters()),show_saved=True)
	# #dot.render('Computationalgraph.gv') 
	

	# for epoch in range(10):
	# 	rp = torch.randperm(AllSampleSize)[:batchSize]
			
	# 	x_=Input[rp,:]
	# 	#y_=Y[rp,:]
	# 	y_=Y[rp]
	# 	optimizer.zero_grad()
	# 	quantize,latent_loss, embed_ind=Model(x_)
	# 	print("here")
	# 	print(embed_ind.shape)
	# 	recon_loss = criterion(quantize, y_)


	# 	#recon_loss = criterion(one_hot, Y)
		
	# 	latent_loss = latent_loss.mean()
	# 	latent_loss_weight=0.25
	# 	loss = recon_loss + latent_loss_weight * latent_loss
	# 	#print(quantize)
		
	# 	print("recon_loss: "+str(float(recon_loss)))

	# 	print("latent_loss: "+str(float(latent_loss)))


	# 	print("codebook index counts")
	# 	print(torch.unique(embed_ind).shape)
	# 	#print("loss: "+str(float(loss)))
	# 	loss.backward()
	# 	optimizer.step()



	class Encoder(nn.Module):
		def __init__(
			self,
			inputSize=213,
			embed_dim=200,
			n_embed=512,
			decay=0.99,
			n_segments=4
		):
			super().__init__()

			self.enc = nn.Linear(inputSize,embed_dim)
			self.quantize= Quantize_conditional(embed_dim,n_embed,groups=n_segments)
			
		def forward(self, input):
			x=self.enc(input)
			
			quant_x,diff,CB_index =self.quantize(x)
		   
			return quant_x, diff,CB_index


	Model=Encoder()
	

	for name, param in Model.named_parameters():
		if param.requires_grad:
			print (name, param.data.shape)

	input_dim=213
	embed_dim=200
	n_embed=512
	N_units=6
	AllSampleSize=100000
	batchSize=63

	Input=torch.randn((AllSampleSize,N_units,input_dim))



	optimizer = torch.optim.Adam(Model.parameters())
	Y=torch.randn((AllSampleSize,N_units,embed_dim))
	#Y=torch.randint(0,n_embed,(AllSampleSize,))
	#Y=torch.randint(0,2,(1000,n_embed)).float()
	#Y=torch.randn((AllSampleSize,n_embed))

	criterion = nn.MSELoss()
	#criterion = nn.CrossEntropyLoss()
	###visualize computation graph

	#dot=make_dot(Y.mean(), params=dict(Model.named_parameters()),show_saved=True)
	#dot.render('Computationalgraph.gv') 
	

	for epoch in range(10):
		rp = torch.randperm(AllSampleSize)[:batchSize]
			
		x_=Input[rp,:]
		#y_=Y[rp,:]
		y_=Y[rp]
		optimizer.zero_grad()
		quantize,latent_loss, embed_ind=Model(x_)
	
		recon_loss = criterion(quantize, y_)


		#recon_loss = criterion(one_hot, Y)
		
		latent_loss = latent_loss.mean()
		latent_loss_weight=0.25
		loss = recon_loss + latent_loss_weight * latent_loss
		#print(quantize)
		
		print("recon_loss: "+str(float(recon_loss)))

		print("latent_loss: "+str(float(latent_loss)))


		print("codebook index counts")
		print(torch.unique(embed_ind).shape)
		#print("loss: "+str(float(loss)))
		loss.backward()
		optimizer.step()








	#one hot VQVAclear


	# class Encoder(nn.Module):
	# 	def __init__(
	# 		self,
	# 		inputSize=213,
	# 		embed_dim=256,
	# 		n_embed=11,
	# 		decay=0.99,
	
	# 	):
	# 		super().__init__()

	# 		self.enc = nn.Linear(inputSize,embed_dim)
	# 		self.quantize= Quantize_onehot_VQVAE(embed_dim,n_embed,groups=4)
			
	# 	def forward(self, input):
	# 		x=self.enc(input)
			
	# 		quant_x,diff,CB_index =self.quantize(x)
		   
	# 		return quant_x, diff,CB_index


	# Model=Encoder()
	

	# for name, param in Model.named_parameters():
	# 	if param.requires_grad:
	# 		print (name, param.data.shape)

	# input_dim=213
	# embed_dim=256
	# n_embed=11

	# N_units=6
	# groups=4
	# AllSampleSize=100000
	# batchSize=63

	# Input=torch.randn((AllSampleSize,N_units,input_dim))



	# optimizer = torch.optim.Adam(Model.parameters())
	# #Y=torch.randn((AllSampleSize,embed_dim))
	# #Y=torch.randint(0,n_embed,(AllSampleSize,))
	# #Y=torch.randint(0,2,(1000,n_embed)).float()
	# Y=torch.randn((AllSampleSize,N_units,n_embed*groups))

	# criterion = nn.MSELoss()
	# #criterion = nn.CrossEntropyLoss()
	# ###visualize computation graph

	# #dot=make_dot(Y.mean(), params=dict(Model.named_parameters()),show_saved=True)
	# #dot.render('Computationalgraph.gv') 
	

	# for epoch in range(30):
	# 	rp = torch.randperm(AllSampleSize)[:batchSize]
			
	# 	x_=Input[rp,:]
	# 	#y_=Y[rp,:]
	# 	y_=Y[rp]
	# 	optimizer.zero_grad()
	# 	quantize,latent_loss, embed_ind=Model(x_)

	# 	recon_loss = criterion(quantize, y_)


	# 	#recon_loss = criterion(one_hot, Y)
		
	# 	latent_loss = latent_loss.mean()
	# 	latent_loss_weight=0.25
	# 	loss = recon_loss + latent_loss_weight * latent_loss
	# 	#print(quantize)
		
	# 	print("recon_loss: "+str(float(recon_loss)))

	# 	print("latent_loss: "+str(float(latent_loss)))


	# 	print("codebook index counts")
	# 	print(torch.unique(embed_ind).shape)
	# 	#print("loss: "+str(float(loss)))
	# 	loss.backward()
	# 	optimizer.step()



	# class Encoder(nn.Module):
	# 	def __init__(
	# 		self,
	# 		inputSize=213,
	# 		embed_dim=256,
	# 		n_embed=11,
	# 		decay=0.99,
	
	# 	):
	# 		super().__init__()

	# 		self.enc = nn.Linear(inputSize,embed_dim)
	# 		self.quantize= Quantize_onehot_gumbel(embed_dim,n_embed,groups=4)
			
	# 	def forward(self, input):
	# 		x=self.enc(input)
			
	# 		quant_x,diff,CB_index =self.quantize(x)
		   
	# 		return quant_x, diff,CB_index


	# Model=Encoder()
	

	# for name, param in Model.named_parameters():
	# 	if param.requires_grad:
	# 		print (name, param.data.shape)

	# input_dim=213
	# embed_dim=256
	# n_embed=11
	# N_units=6
	# AllSampleSize=100000
	# batchSize=63
	# groups=4
	# Input=torch.randn((AllSampleSize,N_units,input_dim))



	# optimizer = torch.optim.Adam(Model.parameters())
	# #Y=torch.randn((AllSampleSize,embed_dim))
	# #Y=torch.randint(0,n_embed,(AllSampleSize,))
	# #Y=torch.randint(0,2,(1000,n_embed)).float()
	# Y=torch.randn((AllSampleSize,N_units,n_embed*groups))

	# criterion = nn.MSELoss()
	# #criterion = nn.CrossEntropyLoss()
	# ###visualize computation graph

	# #dot=make_dot(Y.mean(), params=dict(Model.named_parameters()),show_saved=True)
	# #dot.render('Computationalgraph.gv') 
	

	# for epoch in range(20):
	# 	rp = torch.randperm(AllSampleSize)[:batchSize]
			
	# 	x_=Input[rp,:]
	# 	#y_=Y[rp,:]
	# 	y_=Y[rp]
	# 	optimizer.zero_grad()
	# 	quantize,latent_loss, embed_ind=Model(x_)
		
	# 	recon_loss = criterion(quantize, y_)


	# 	#recon_loss = criterion(one_hot, Y)
		
	# 	latent_loss = latent_loss.mean()
	# 	latent_loss_weight=0.25
	# 	loss = recon_loss + latent_loss_weight * latent_loss
	# 	#print(quantize)
		
	# 	print("recon_loss: "+str(float(recon_loss)))

	# 	print("latent_loss: "+str(float(latent_loss)))


	# 	print("codebook index counts")
	# 	print(torch.unique(embed_ind).shape)
	# 	#print("loss: "+str(float(loss)))
	# 	loss.backward()
	# 	optimizer.step()