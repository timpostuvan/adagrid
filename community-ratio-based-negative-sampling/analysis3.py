"""
Plots:
- selected edge message ratio vs. epochs during training for adaptive hill climbing approach 
  (on the same figure more subfigures are plotted)
- on each subfigure also the best grid search message ratio is plotted
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (12, 5)



def arg_parse():
	parser = argparse.ArgumentParser(description='Dataset and configuration for analysis.')
	parser.add_argument('--dataset', type=str,
						help='Dataset.')
	parser.add_argument('--num_layers', type=int,
						help='Number of layers of GNN.')
	parser.add_argument('--hidden_dim', type=int,
						help='Hidden dimension of GNN.')

	parser.set_defaults(
			dataset='cora',
			num_layers=2,
			hidden_dim=64
	)
	return parser.parse_args()

    
args = arg_parse()
folder = f"{args.dataset}-{args.num_layers}-{args.hidden_dim}"

iterations = 1
num_epochs = 500
data_splits = [[0.2, 0.4, 0.4], [0.5, 0.25, 0.25], [0.8, 0.1, 0.1]]
adapt_epochs = [100, 50, 10]
try_epochs = [1, 5, -1]
criterions = ['val', 'gap']

grid_search_edge_message_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


for data_split in data_splits:
	folder_split = "{}-{}-{}".format(int(100 * data_split[0]), int(100 * data_split[1]), int(100 * data_split[2]))
	folder_results = folder + "/results/" + folder_split
	analysis_folder = folder + "/analysis/" + folder_split
	
	Path(analysis_folder).mkdir(parents=True, exist_ok=True)


	best_grid_search_val_acc = -np.inf
	best_grid_search_message_ratio = 0.0
	for edge_message_ratio in grid_search_edge_message_ratios:
		file_name = folder_results + "/normal_{}.csv".format(int(100 * edge_message_ratio))
		frame = pd.read_csv(file_name)
		val_acc = frame.to_numpy()[0][-2]

		if(val_acc > best_grid_search_val_acc):
			best_grid_search_val_acc = val_acc
			best_grid_search_message_ratio = edge_message_ratio



	color = ["b-.", "g:", "y--"]
	for adapt_epoch in adapt_epochs:
		fig, axs = plt.subplots(1, 2)

		for criterion_i in range(len(criterions)):
			ax = axs[criterion_i]
			epochs = np.linspace(0, num_epochs, int(num_epochs / adapt_epoch), endpoint=False)
			grid_search_ratio = [best_grid_search_message_ratio for i in range(int(num_epochs / adapt_epoch))]
			ax.plot(epochs, grid_search_ratio, "r", label="Best complete search")
				
			for try_epoch_i in range(len(try_epochs)):
				criterion = criterions[criterion_i]
				try_epoch = try_epochs[try_epoch_i]

				if(try_epoch == -1):
					try_epoch = adapt_epoch

				file_name = folder_results + "/edge_message_ratio_adapt_{}_{}_{}.csv".format(criterion, 
																							adapt_epoch, try_epoch)
				frame = pd.read_csv(file_name)
				for iteration in range(1, iterations + 1):
					edge_message_ratios = frame["Iteration {}".format(iteration)].to_numpy()
					epochs = np.linspace(0, num_epochs, len(edge_message_ratios), endpoint=False)

					ax.plot(epochs, edge_message_ratios, color[try_epoch_i], label="Try epochs = {}".format(try_epoch))


				if(criterion == 'val'):
					criterion = 'Validation'

				if(criterion == 'gap'):
					criterion = 'Gap'

				ax.legend()
				ax.set_title("{} criterion, adapt epochs = {}".format(criterion, adapt_epoch))
				ax.set(xlabel="Epochs", ylabel="Edge message ratio")
		
		
		file_name = analysis_folder + "/edge_message_ratios_{}.png".format(adapt_epoch)
		plt.tight_layout()
		plt.savefig(file_name)
		plt.clf()