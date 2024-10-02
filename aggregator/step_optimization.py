import numpy as np
import os
import time
import pandas as pd
from scipy.optimize import  Bounds, LinearConstraint, minimize

from tcl.tcl import Tcl
from fonction_objectif import *
from curtailement import  *


def update_fully_corrective(params,warm_start,aggregator_profiles,agent_profiles,aggregator_costs,agents_costs,agents_list):

    # trouve le omega qui minimise la fonction objectif, linéarisée au niveau des couts lineaires
    dimension_omega = len(warm_start)
    bound = Bounds(0,1)
    cons = LinearConstraint(A = np.ones((1,dimension_omega)), lb= 1, ub = 1)
    x0 = warm_start

    res = minimize( lambda omega: fully_linearized_cost(omega,params, aggregator_profiles,agent_profiles,
                                                        aggregator_costs,agents_costs,agents_list),
                    x0=x0, bounds = bound, constraints = cons, method = 'trust-constr')

    return  res.x,res.fun


def update_warm_start_step_it(num_it,params,step_it):


    memory_size = min(params.depth_fully_corrective, num_it + 1)

    if num_it == 1:
        warm_start = np.ones(memory_size + 1 )/(memory_size+1)

    elif len(step_it) < params.depth_fully_corrective + 1:
            warm_start = np.concatenate([step_it, np.zeros(1)])
    else:
        residual = step_it[-1]
        warm_start = np.concatenate([step_it[:-1], np.array([residual])])

    return warm_start,memory_size


def preparation_dict_cvx_combination(num_it,memory_size,aggregator_profiles_dict_per_it,profile_aggreg,profiles_dict_per_it,
                                     agent_averaged_profile_dict,aggregator_costs_dict_per_it,cost_aggregator_averaged,
                                     costs_dict_per_it,costs_dict_individual_averaged,agents_list) :


    aggregator_profiles = {k: np.copy(aggregator_profiles_dict_per_it[num_it + 1 - memory_size + k]) for k in
                           range(memory_size)}
    aggregator_profiles[memory_size] = np.copy(profile_aggreg)

    agent_profiles = {
        k: {agent.name: np.copy(profiles_dict_per_it[num_it + 1 - memory_size + k][agent.name]) for agent in
            agents_list} for k in range(memory_size)}
    agent_profiles[memory_size] = {agent.name: np.copy(agent_averaged_profile_dict[agent.name]) for agent in
                                   agents_list}

    aggregator_costs = {k: aggregator_costs_dict_per_it[num_it + 1 - memory_size + k] for k in range(memory_size)}
    aggregator_costs[memory_size] = cost_aggregator_averaged

    agents_costs = {k: costs_dict_per_it[num_it + 1 - memory_size + k] for k in range(memory_size)}
    agents_costs[memory_size] = costs_dict_individual_averaged

    return aggregator_profiles,  agent_profiles, aggregator_costs, agents_costs


def closed_loop_step(num_it):

    return 2/(2+num_it)