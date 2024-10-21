import numpy as np
from curtailement import  fobj, f_0

def fully_linearized_cost(omega,params, aggregator_profiles,agent_profiles,aggregator_costs,agents_costs,agents_list):
    """
    This function evaluates the linearized cost associated with a list of weights (omega)
    """

    linear_comb_agg_profiles = sum([omega[k]*aggregator_profiles[k] for k in aggregator_profiles.keys() ])
    linear_comb_market_value_income = fobj(params,linear_comb_agg_profiles)
    linear_comb_local_cost_linearized = sum([omega[k] * agents_costs[k][agent.name] for agent in agents_list for k in agents_costs.keys() ])

   # linear_comb_agents_profiles = sum([omega[k]*np.array(sum([agent_profiles[k][agent.name] for agent in agents_list]))  for k in agent_profiles.keys()])
    linear_comb_agents_profiles = sum([np.array(sum(np.array([omega[k]*agent_profiles[k][agent.name] for k in agent_profiles.keys()])))  for agent in agents_list])

    penalization_norm_2 = f_0(params,linear_comb_agg_profiles, linear_comb_agents_profiles)
    return linear_comb_market_value_income + penalization_norm_2 + linear_comb_local_cost_linearized



def build_dico_individual_cost(agents_profile_ditct,agents_list):
    """
    Retourne un dictionnaire avec en clef le nom des agents et en valeur
    """
    dict_individual_cost = {}
    for agent_bb in agents_list:
        if agent_bb.__class__.__name__ == 'BlackBoxAgent':
            dict_individual_cost[agent_bb.name] = agent_bb.get_local_cost(np.array(agents_profile_ditct[agent_bb.name]))
        elif agent_bb.__class__.__name__ == 'Tcl':
            dict_individual_cost[agent_bb.name] = agent_bb.individual_cost(agents_profile_ditct[agent_bb.name])
    return dict_individual_cost

def objective_fun(params,profile_aggreg,agents_profile_ditct,agents_list,dict_individual_cost=None):
    p_agents_aggregated  = np.zeros(48)
    cost_agents_sum = 0
    for agent_bb in agents_list:
        p_agents_aggregated = p_agents_aggregated + agents_profile_ditct[agent_bb.name]
        if dict_individual_cost != None:
            cost_agents_sum += dict_individual_cost[agent_bb.name]
        else :
            if agent_bb.__class__.__name__ == 'BlackBoxAgent':
                cost_agents_sum += agent_bb.get_local_cost(np.array(agents_profile_ditct[agent_bb.name]))
            elif agent_bb.__class__.__name__ == 'Tcl':
                cost_agents_sum += agent_bb.individual_cost(agents_profile_ditct[agent_bb.name])
    return fobj(params, profile_aggreg) + f_0(params, profile_aggreg,p_agents_aggregated) + cost_agents_sum
