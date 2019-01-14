import pandas as pd
import numpy as np
from itertools import permutations
from pgmpy.models import BayesianModel
import matplotlib.pyplot as plt
import networkx as nx
from math import *

def init(data):
	variables = list(data.columns.values)
	state_names = {variable: sorted(list(data.ix[:, variable].dropna().unique())) for variable in variables}
	return variables,state_names

def state_count(data,variable,parents,state_names):
	data = data.dropna()
	if not parents:
		state_count_data = data.ix[:, variable].value_counts()
		state_counts = state_count_data.reindex(state_names[variable]).fillna(0).to_frame()
	else:
		parents_states = [state_names[parent] for parent in parents]
		state_count_data = data.groupby([variable] + parents).size().unstack(parents)
		row_index = state_names[variable]
		column_index = pd.MultiIndex.from_product(parents_states, names=parents)
		state_counts = state_count_data.reindex(index=row_index, columns=column_index).fillna(0)
	return state_counts

def score_BD(data, variable,parents,state_names):
	equivalent_sample_size=700
	var_states = state_names[variable]
	var_cardinality = len(var_states)
	state_counts = state_count(data,variable, parents,state_names)
	num_parents_states = float(len(state_counts.columns))
	score = 0
	for parents_state in state_counts: 
		conditional_sample_size = sum(state_counts[parents_state])
		score += (lgamma(equivalent_sample_size / num_parents_states) -lgamma(conditional_sample_size + equivalent_sample_size / num_parents_states))
		for state in var_states:
			if state_counts[parents_state][state] > 0:
				score += (lgamma(state_counts[parents_state][state] +equivalent_sample_size / (num_parents_states * var_cardinality)) -lgamma(equivalent_sample_size / (num_parents_states * var_cardinality)))
	return score

def operations(data,model,state_names):
	nodes = state_names.keys()
	potential_new_edges = (set(permutations(nodes, 2)) -set(model.edges()) -set([(Y, X) for (X, Y) in model.edges()]))
	best_score=0
	for (X, Y) in potential_new_edges:  # add
		if nx.is_directed_acyclic_graph(nx.DiGraph(model.edges() + [(X, Y)])):
			operation = ('add', (X, Y))

			old_parents = model.get_parents(Y)
			new_parents = old_parents + [X]
			score_delta = score_BD(data,Y, new_parents,state_names) - score_BD(data,Y, old_parents,state_names)
			if score_delta>0:
				yield(operation, score_delta)

	best_score=0
	for (X, Y) in model.edges():  # remove 
		operation = ('remove', (X, Y))
		old_parents = model.get_parents(Y)
		new_parents = old_parents[:]
		new_parents.remove(X)
		score_delta = score_BD(data,Y, new_parents,state_names) - score_BD(data,Y, old_parents,state_names)
		if score_delta>0:
			yield(operation, score_delta)

	for (X, Y) in model.edges():  # reverse
		new_edges = model.edges() + [(Y, X)]
		new_edges.remove((X, Y))
		if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
			operation = ('reverse', (X, Y))
			old_X_parents = model.get_parents(X)
			old_Y_parents = model.get_parents(Y)
			new_X_parents = old_X_parents + [Y]
			new_Y_parents = old_Y_parents[:]
			new_Y_parents.remove(X)
			score_delta = (score_BD(data,X, new_X_parents,state_names) +score_BD(data,Y, new_Y_parents,state_names) -score_BD(data,X, old_X_parents,state_names) -score_BD(data,Y, old_Y_parents,state_names))
			if score_delta>0:
				yield(operation, score_delta)

def Hill_Climb_Search(data,state_names):
	epsilon = 1e-14
	nodes = state_names.keys()
	start = BayesianModel()
	start.add_nodes_from(nodes)
	current_model = start
	while True:
		best_score_delta = 0
		best_operation = None
		for operation, score_delta in operations(data,current_model, state_names):
			if score_delta > best_score_delta:
				best_operation = operation
				best_score_delta = score_delta
		if best_operation is None or best_score_delta < epsilon:
			break
		elif best_operation[0] == 'add':
			current_model.add_edge(*best_operation[1])
		elif best_operation[0] == 'delete':
			current_model.remove_edge(*best_operation[1])
		elif best_operation[0] == 'reverse':
			X, Y = best_operation[1]
			current_model.remove_edge(X, Y)
			current_model.add_edge(Y, X)
		print 'Iteration:'
		print current_model.edges()
		print 'Score: ',best_score_delta
		print 'Best operation: ',best_operation
	print current_model.edges()
	return current_model

data = pd.read_csv("final.csv")
variables, state_names=init(data)
final_model=Hill_Climb_Search(data,state_names)
print len(final_model.edges())
nx.draw(final_model,with_labels=True, node_size=1500)
plt.draw()
plt.show()

'''
PGM Report:
Input: 9 files having 11 variables with their values which are continuous in nature.
Data Preprocessing: 9 files combined to 1. Outliers(having data variance>3) are removed, binned into 3 stages i.e., 0, 1, 2.

Algorithm Implemented:
	Scoring function used: BDeu scoring (write formula), Score decomposibilty used.
	Start point: Empty Bayesian Graph.
	while the operation choosen improves the score of graph:
		Check all possibility of operation at this stage (add, delete, reverse edge).
		Only one operation permitted at a time.
		Choose the operation which maximizes the score of graph.
		Using decomposibilty compute the new score of the graph.

return the graph

'''