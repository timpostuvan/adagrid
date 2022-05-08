"""
Plots:
- grid search AUC vs. edge message ratio graph
- difference between validation criterion and best grid search AUC heatmap
- difference between gap criterion and best grid search AUC heatmap
- difference between validation and gap criterion AUC heatmap
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def plot_grid_search_results(file_name, grid_search_results, edge_message_ratios, random_test_acc):
	random_results = [random_test_acc for x in edge_message_ratios]

	plt.plot(edge_message_ratios, grid_search_results, "b", label="Complete search AUC")
	plt.plot(edge_message_ratios, random_results, "r", label="Random approach AUC")
	plt.legend()
	
	plt.title("AUC vs. edge message ratio")
	plt.xlabel("Edge message ratio")
	plt.ylabel("AUC")	
#	plt.show()

	plt.savefig(file_name)
	plt.clf()



def plot_difference_heatmap(file_name, plot_title, results, baseline_results, adapt_epochs, try_epochs):
	ax = sns.heatmap(100 * results, xticklabels=try_epochs, yticklabels=adapt_epochs, annot=True, fmt=".4g")
	ax.set(title=plot_title, ylabel="Adapt epochs", xlabel="Try epochs")
#	plt.show()

	plt.savefig(file_name)
	plt.clf()



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

data_splits = [[0.2, 0.4, 0.4], [0.5, 0.25, 0.25], [0.8, 0.1, 0.1]]
adapt_epochs = [100, 50, 10]
try_epochs = [1, 5, -1]
criterions = ['val', 'gap']

edge_message_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


for data_split in data_splits:
	folder_split = "{}-{}-{}".format(int(100 * data_split[0]), int(100 * data_split[1]), int(100 * data_split[2]))
	folder_results = folder + "/results/" + folder_split
	analysis_folder = folder + "/analysis/" + folder_split

	Path(analysis_folder).mkdir(parents=True, exist_ok=True)


	best_grid_search_val_acc = -np.inf
	best_grid_search_test_acc = -np.inf

	grid_search_results = []
	for edge_message_ratio in edge_message_ratios:
		file_name = folder_results + "/normal_{}.csv".format(int(100 * edge_message_ratio))
		frame = pd.read_csv(file_name)
		val_acc = frame.to_numpy()[0][-2]
		test_acc = frame.to_numpy()[0][-1]
		grid_search_results.append(test_acc)

		if(val_acc > best_grid_search_val_acc):
			best_grid_search_val_acc = val_acc
			best_grid_search_test_acc = test_acc

	grid_search_results = np.array(grid_search_results)
	best_grid_search_acc = best_grid_search_test_acc


	file_name = folder_results + "/random.csv"
	frame = pd.read_csv(file_name)
	random_test_acc = frame.to_numpy()[0][-1]
	
	print(f"Data split: {folder_split}")
	print(f"Complete search acc: {100 * best_grid_search_acc}")
	print(f"Random search acc: {100 * random_test_acc}")
	print()

	val_results = []
	for adapt_epoch in adapt_epochs:
		current_results = []
		for try_epoch in try_epochs:
			if(try_epoch == -1):
				try_epoch = adapt_epoch

			file_name = folder_results + "/adapt_val_{}_{}.csv".format(adapt_epoch, try_epoch)
			frame = pd.read_csv(file_name)
			test_acc = frame.to_numpy()[0][-1]
			current_results.append(test_acc)

		val_results.append(current_results)

	val_results = np.array(val_results)



	gap_results = []
	for adapt_epoch in adapt_epochs:
		current_results = []
		for try_epoch in try_epochs:
			if(try_epoch == -1):
				try_epoch = adapt_epoch

			file_name = folder_results + "/adapt_gap_{}_{}.csv".format(adapt_epoch, try_epoch)
			frame = pd.read_csv(file_name)
			test_acc = frame.to_numpy()[0][-1]
			current_results.append(test_acc)

		gap_results.append(current_results)

	gap_results = np.array(gap_results)



	file_name = analysis_folder + "/grid_search_results.png"
	plot_grid_search_results(file_name, grid_search_results, edge_message_ratios, random_test_acc)

	file_name = analysis_folder + "/difference_val_grid_search.png"
	plot_difference_heatmap(file_name, "Difference between validation criterion and \n best complete search AUC (val - best complete search)", 
					val_results, best_grid_search_acc, adapt_epochs, try_epochs)

	file_name = analysis_folder + "/difference_gap_grid_search.png"
	plot_difference_heatmap(file_name, "Difference between gap criterion and \n best complete search AUC (gap - best complete search)", 
					gap_results, best_grid_search_acc, adapt_epochs, try_epochs)

	file_name = analysis_folder + "/difference_val_gap.png"
	plot_difference_heatmap(file_name, "Difference between validation and \n gap criterion AUC (val - gap)", 
					val_results, gap_results, adapt_epochs, try_epochs)