from pathlib import Path
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import argparse
import copy
import math
import random
import numpy as np
import pandas as pd
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn

from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch



class Arguments:
	def __init__(self, dataset='cora', device='cpu', epochs=500, mode='all', model='GCN',
				edge_message_ratio=0.6, layers=2, hidden_dim=64, batch_size=1, data_split=[0.85, 0.05, 0.1],
				verbose=False, adapt=False, try_epochs=3, adapt_epochs=50, criterion='val', random=False):
		self.dataset = dataset
		self.device = device
		self.epochs = epochs
		self.mode = mode
		self.model = model
		self.edge_message_ratio = edge_message_ratio
		self.layers = layers
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.data_split = data_split
		self.verbose = verbose

		self.adapt = adapt
		self.try_epochs = try_epochs
		self.adapt_epochs = adapt_epochs
		self.criterion = criterion

		self.random = random



def arg_parse():
    parser = argparse.ArgumentParser(description='Link prediction arguments.')
    parser.add_argument('--gpu', type=int,
                        help='GPU device.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset.')
    parser.add_argument('--num_layers', type=int,
                        help='Number of layers of GNN.')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')
    parser.add_argument('--verbose', action='store_true',
                        help='Whether output is verbose.')

    parser.set_defaults(
            gpu=0,
            dataset='cora',
            num_layers=2,
            hidden_dim=64,
            verbose=False
    )
    return parser.parse_args()



class Net(torch.nn.Module):
    def __init__(self, input_dim, args):
        super(Net, self).__init__()
        self.model = args.model
        if self.model == 'GCN':
            self.conv_first = pyg_nn.GCNConv(input_dim, args.hidden_dim)
            self.convs = torch.nn.ModuleList([pyg_nn.GCNConv(args.hidden_dim, args.hidden_dim) for i in range(args.layers - 2)])
            self.conv_last = pyg_nn.GCNConv(args.hidden_dim, args.hidden_dim)
        else:
            raise ValueError('unknown conv')
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, graph):
        x = F.dropout(graph.node_feature, p=0.2, training=self.training)
        x = F.relu(self._conv_op(self.conv_first, x, graph))

        for i in range(len(self.convs)):
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.relu(self._conv_op(self.convs[i], x, graph))
                

        x = F.dropout(x, p=0.2, training=self.training)
        x = self._conv_op(self.conv_last, x, graph)

        nodes_first = torch.index_select(x, 0, graph.edge_label_index[0,:].long())
        nodes_second = torch.index_select(x, 0, graph.edge_label_index[1,:].long())
        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        return pred
    
    def _conv_op(self, conv, x, graph):
        if self.model == 'GCN':
            return conv(x, graph.edge_index)
        elif self.model == 'spline':
            return conv(x, graph.edge_index, graph.edge_feature)

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)



def try_edge_message_ratio(edge_message_ratio, model, datasets, dataloaders, optimizer, args, scheduler=None):
	datasets['train'].edge_message_ratio = edge_message_ratio

	val_max = -math.inf
	best_model = model

	mean_accs = {mode: 0 for mode, dataloader in dataloaders.items()}
	for epoch in range(args.try_epochs):
		for iter_i, batch in enumerate(dataloaders['train']):
			batch.to(args.device)
			model.train()
			optimizer.zero_grad()
			pred = model(batch)
			loss = model.loss(pred, batch.edge_label.type(pred.dtype))
			loss.backward()
			optimizer.step()
			if scheduler is not None:
				scheduler.step()

			accs, _ = test(model, dataloaders, args)
			for mode in accs:
				mean_accs[mode] += accs[mode]

			if val_max < accs['val']:
				val_max = accs['val']
				best_model = copy.deepcopy(model)


	for mode in mean_accs:
		mean_accs[mode] /= args.try_epochs

	log = 'Edge message ratio: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
	if(args.verbose):
		print(log.format(edge_message_ratio, mean_accs['train'], mean_accs['val'], mean_accs['test']))
	
	return mean_accs, val_max, best_model	



def train(model, datasets, dataloaders, optimizer, args, scheduler=None):
	# training loop
	val_max = -math.inf
	best_model = model

	input_dim = datasets['train'].num_node_features
	num_classes = datasets['train'].num_edge_labels

	waiting = 0
	edge_message_ratio_changes = []
	
	for epoch in range(0, args.epochs):
		if(waiting > 0):
			waiting -= 1
			continue


		# search to determine the next edge_message_ratio
		if(args.adapt and epoch % args.adapt_epochs == 0):
			best_ratio = 0.0
			try_val_max = None

			if(args.criterion == 'val'):
				try_val_max = -math.inf

			if(args.criterion == 'gap'):
				try_val_max = math.inf


			best_try_model = model
			best_try_optimizer = optimizer
			best_try_scheduler = scheduler

			# best version of the best_try_model during training for try epochs (highest validation accuracy)
			best_val_model = model
			best_val = -math.inf


			edge_message_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
			for edge_message_ratio in edge_message_ratios:
				new_model = Net(input_dim, args).to(args.device)
				new_model.load_state_dict(model.state_dict())

				new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
				new_optimizer.load_state_dict(optimizer.state_dict())

				new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=args.epochs)
				new_scheduler.load_state_dict(scheduler.state_dict())

				try_accs, current_val, current_val_model = try_edge_message_ratio(edge_message_ratio, 
																new_model, datasets, dataloaders, new_optimizer, 
																args, new_scheduler)

				if (args.criterion == 'val' and try_val_max < try_accs['val']):
					try_val_max = try_accs['val']
					best_ratio = edge_message_ratio
					best_try_model = new_model
					best_try_optimizer = new_optimizer
					best_try_scheduler = new_scheduler

					best_val = current_val
					best_val_model = current_val_model


				if (args.criterion == 'gap' and try_val_max > abs(try_accs['train'] - try_accs['val'])):
					try_val_max = abs(try_accs['train'] - try_accs['val'])
					best_ratio = edge_message_ratio
					best_try_model = new_model
					best_try_optimizer = new_optimizer
					best_try_scheduler = new_scheduler

					best_val = current_val
					best_val_model = current_val_model



			model = best_try_model
			optimizer = best_try_optimizer
			scheduler = best_try_scheduler
			datasets['train'].edge_message_ratio = best_ratio 

			edge_message_ratio_changes.append(datasets['train'].edge_message_ratio)

			if val_max < best_val:
				val_max = best_val
				best_model = best_val_model

			waiting = args.try_epochs - 1
			continue


				

		if(args.random):
			datasets['train'].edge_message_ratio = random.uniform(0.1, 0.9)

		for iter_i, batch in enumerate(dataloaders['train']):
			batch.to(args.device)
			model.train()
			optimizer.zero_grad()
			pred = model(batch)
			loss = model.loss(pred, batch.edge_label.type(pred.dtype))
			loss.backward()
			optimizer.step()
			if scheduler is not None:
				scheduler.step()
				

		log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
		accs, _ = test(model, dataloaders, args)

		if(args.verbose):
			print(log.format(epoch, accs['train'], accs['val'], accs['test']))

		if val_max < accs['val']:
			val_max = accs['val']
			best_model = copy.deepcopy(model)


	log = 'Best, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
	accs, _ = test(best_model, dataloaders, args)
	print(log.format(accs['train'], accs['val'], accs['test']))
	return np.array([accs['train'], accs['val'], accs['test']]), edge_message_ratio_changes


def test(model, dataloaders, args, max_train_batches=1):
	model.eval()
	accs = {}
	losses = {}
	for mode, dataloader in dataloaders.items():
		acc = 0
		loss = 0
		num_batches = 0
		for batch in dataloader:
			batch.to(args.device)
			pred = model(batch)
			# only 1 graph in dataset. In general needs aggregation
			loss += model.loss(pred, batch.edge_label.type(pred.dtype)).cpu().data.numpy()
			acc += roc_auc_score(batch.edge_label.flatten().cpu().numpy(), 
								pred.flatten().data.cpu().numpy())
			num_batches += 1
			if mode == 'train' and num_batches >= max_train_batches:
				# do not eval on the entire training set for efficiency
				break

		accs[mode] = acc / num_batches
		losses[mode] = loss / num_batches
		
	return accs, losses



def run(args):
	pyg_dataset = None
	if(args.dataset == 'cora'):
		pyg_dataset = Planetoid('./datasets', 'Cora', transform=T.TargetIndegree())

	if(args.dataset == 'citeseer'):
		pyg_dataset = Planetoid('./datasets', 'CiteSeer', transform=T.TargetIndegree())

	if(args.dataset == 'pubmed'):
		pyg_dataset = Planetoid('./datasets', 'PubMed', transform=T.TargetIndegree())



	# the input that we assume users have
	edge_train_mode = args.mode
	if(args.verbose):
		print('edge train mode: {}'.format(edge_train_mode))

	graphs = GraphDataset.pyg_to_graphs(pyg_dataset)

	dataset = GraphDataset(graphs, 
	   task='link_pred', 
	   edge_message_ratio=args.edge_message_ratio, 
	   edge_train_mode=edge_train_mode,
	   resample_disjoint=True,
       resample_disjoint_period=1)
	
	if(args.verbose):
		print('Initial dataset: {}'.format(dataset))


	# split dataset
	datasets = {}
	datasets['train'], datasets['val'], datasets['test'] = dataset.split(
		transductive=True, split_ratio=args.data_split)

	if(args.verbose):
		print('after split')
		print('Train message-passing graph: {} nodes; {} edges.'.format(
			datasets['train'][0].G.number_of_nodes(),
			datasets['train'][0].G.number_of_edges()))
		print('Val message-passing graph: {} nodes; {} edges.'.format(
			datasets['val'][0].G.number_of_nodes(),
			datasets['val'][0].G.number_of_edges()))
		print('Test message-passing graph: {} nodes; {} edges.'.format(
			datasets['test'][0].G.number_of_nodes(),
			datasets['test'][0].G.number_of_edges()))


	# node feature dimension
	input_dim = datasets['train'].num_node_features
	# link prediction needs 2 classes (0, 1)
	num_classes = datasets['train'].num_edge_labels

	model = Net(input_dim, args).to(args.device)

	optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	follow_batch = []

	dataloaders = {split: DataLoader(
	ds, collate_fn=Batch.collate(follow_batch), 
		batch_size=args.batch_size, shuffle=(split=='train'))
		for split, ds in datasets.items()}
	
	if(args.verbose):
		print('Graphs after split: ')
		for key, dataloader in dataloaders.items():
			for batch in dataloader:
				print(key, ': ', batch)

	return train(model, datasets, dataloaders, optimizer, args, scheduler=scheduler)




def write_file(file_name, total_acc):
    print(total_acc)
    total_acc = total_acc.reshape(1, -1)
    frame = pd.DataFrame(data=total_acc, columns=['Train', 'Validation', 'Test'])
    frame.to_csv(file_name, index=False)
    print(frame)



def experiment(iterations, args, file_name_results=None, file_name_edge_message_ratio_changes=None):
	edge_message_ratio_changes_iterations = []

	total_acc = np.zeros(3)
	for seed in range(1, iterations + 1):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

		acc, edge_message_ratio_changes = run(args)
		total_acc = np.add(total_acc, acc)
		edge_message_ratio_changes_iterations.append(edge_message_ratio_changes)

	total_acc /= iterations
	if(file_name_results != None):
		write_file(file_name_results, total_acc)

	if(file_name_edge_message_ratio_changes != None):
		edge_message_ratio_changes_iterations = np.array(edge_message_ratio_changes_iterations)
		data = {"Iteration {}".format(i) : edge_message_ratio_changes_iterations[i - 1, :] for i in range(1, iterations + 1)}
		frame = pd.DataFrame(data)
		frame.to_csv(file_name_edge_message_ratio_changes, index=False)
		

def main():
	global_args = arg_parse()

	iterations = 3
	device = torch.device('cuda:{}'.format(global_args.gpu) if torch.cuda.is_available() else 'cpu')
	dataset_name = global_args.dataset
	verbose = global_args.verbose

	layer = global_args.num_layers
	hidden_dim = global_args.hidden_dim

	data_splits = [[0.8, 0.1, 0.1], [0.2, 0.4, 0.4], [0.5, 0.25, 0.25]]
	adapt_epochs = [100, 50, 10]
	try_epochs = [1, 5, -1]
	criterions = ['val', 'gap']

	edge_message_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

	
	for data_split in data_splits:
		folder_dataset = "{}-{}-{}".format(dataset_name, layer, hidden_dim)
		folder_split = "/{}-{}-{}".format(int(100 * data_split[0]), int(100 * data_split[1]), int(100 * data_split[2]))
		folder_name = folder_dataset + "/results/" + folder_split

		Path(folder_name).mkdir(parents=True, exist_ok=True)



		# complete search - constant edge message ratio
		for edge_message_ratio in edge_message_ratios:
			args = Arguments(device=device, mode="disjoint", verbose=verbose, dataset=dataset_name, adapt=False, 
				edge_message_ratio=edge_message_ratio, hidden_dim=hidden_dim, layers=layer, data_split=data_split)
			file_name = folder_name + "/normal_{}.csv".format(int(100 * edge_message_ratio))

			experiment(iterations, args, file_name)



		# random search - changes edge message randomly after every epoch
		args = Arguments(device=device, mode="disjoint", verbose=verbose, dataset=dataset_name,
			adapt=False, random=True, hidden_dim=hidden_dim, layers=layer, data_split=data_split)
		file_name = folder_name + "/random.csv"
		
		experiment(iterations, args, file_name)
		


		# AdaGrid
		for criterion in criterions:
			for adapt_epoch in adapt_epochs:
				for try_epoch in try_epochs:
					if(try_epoch == -1):
						try_epoch = adapt_epoch

					args = Arguments(device=device, mode="disjoint", verbose=verbose, dataset=dataset_name, 
						adapt=True, adapt_epochs=adapt_epoch, try_epochs=try_epoch, criterion=criterion,
						hidden_dim=hidden_dim, layers=layer, data_split=data_split)

					file_name_results = folder_name + "/adapt_{}_{}_{}.csv".format(criterion, adapt_epoch, try_epoch)
					file_name_edge_message_ratio = folder_name + "/edge_message_ratio_adapt_{}_{}_{}.csv".format(criterion, adapt_epoch, try_epoch)
					experiment(iterations, args, file_name_results, file_name_edge_message_ratio)	


if __name__ == '__main__':
    main()
