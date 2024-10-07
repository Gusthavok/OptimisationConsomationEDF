import numpy as np
import os
import time
import pandas as pd
from scipy.optimize import  Bounds, LinearConstraint, minimize

from tcl.tcl import Tcl

from recombi import recombine

import fonction_objectif 
import curtailement 
import step_optimization

os.environ['XPRESS'] = '/opt/shared/agregi/env_python/bin/xpauth.xpr'

from linopy import Model

import multiprocessing as mp

def optimize(params, agents_list, suffix: str) :

    #Frank Wolfe
    # (agent_profiles_dict, agent_costs_dict) = optim_bruteforce_lambdas(params, agents_list, suffix)
    (agent_profiles_dict, agent_costs_dict) = optim_frankwolfe(params, agents_list, suffix)
    #Recombinaison
    # best_iteration_per_agent =None
    best_iteration_per_agent=recombine(params, agents_list, suffix, agent_profiles_dict, agent_costs_dict)
    
    return (agent_profiles_dict, agent_costs_dict, best_iteration_per_agent)



def write_dicts_to_csv(profiles_dict_per_it, costs_dict_per_it, lambdas_per_it, output_dir, date, suffix: str):
    agents_list = profiles_dict_per_it[0].keys() 
    
    costs_df = pd.DataFrame.from_dict(costs_dict_per_it, orient='index')
    costs_df.to_csv(os.path.join(output_dir, "costs_"+str(pd.to_datetime(date).strftime("%Y%m%d")) + suffix + ".csv"), sep=";", float_format='%.3f', index_label='iter')
    num_its = len(lambdas_per_it)
    to_concat = []
    for it in range(0,num_its):
        df = pd.DataFrame(pd.date_range(start=date, periods=48, freq="30min"))
        df["iter"] = it
        df = pd.concat([df, pd.DataFrame(lambdas_per_it[it], columns= [ "price"])], axis = 1)
        to_concat.append(df)

    pd.concat(to_concat).to_csv(os.path.join(output_dir, "lambdas_"+str(pd.to_datetime(date).strftime("%Y%m%d")) + suffix + ".csv"), sep=";", index=False, float_format='%.3f', index_label='datetime')

    to_concat = []
    for it in range(0,num_its):
        df = pd.DataFrame(pd.date_range(start=date, periods=48, freq="30min"))
        df["iter"] = it
        df = pd.concat([df, pd.DataFrame(profiles_dict_per_it[it])], axis = 1)
        to_concat.append(df)

    pd.concat(to_concat).to_csv(os.path.join(output_dir, "loads_"+str(pd.to_datetime(date).strftime("%Y%m%d")) + suffix +".csv"), sep=";", index=False, float_format='%.3f', index_label='datetime')



def agent_tcl_update_load_helper(args):
    agent, lambdas, date = args
    try:
        path = os.path.join(os.getcwd(), 'OptimisationConsomationEDF')
        agent.tcl_update_load(lambdas, path = path)
    except RuntimeError:
        return agent.name, 1
    return agent.name, 0



def optim_frankwolfe(params, agents_list, suffix: str, lambda_start=np.zeros(48)):


    profile_aggreg = np.zeros(48)
    num_it = 0
    step_it = [0]

    profiles_dict_per_it = {} ## keep in memory successive profiles through iteratations
    costs_dict_per_it = {}  ## keep in memory successive agents costs through iteratations
    lambdas_per_it = {}
    aggregator_profiles_dict_per_it = {} ## keep in memory successive profiles through iteratations
    aggregator_costs_dict_per_it= {}  ## keep in memory successive agregator costs through iteratations

    df_conv = pd.DataFrame(data = None, columns=['iteration','cost','upper_bound','rho'])

    agent_averaged_profile_dict = { agent_bb.name: np.zeros(48) for agent_bb in agents_list}
        
    while (num_it <= params.max_iter_FrankWolfe):
        print("\nIteration Frank-Wolfe : ", num_it, "(agents: ", suffix,")")

        # Compute the new signal to send to each agent
        lambdas = curtailement.derivarite_f0(params, profile_aggreg,agent_averaged_profile_dict, agents_list)
        lambdas_per_it[num_it] = lambdas


        profiles_dict_per_it[num_it] = {} ## keep in memory successive profiles through iteratations
        costs_dict_per_it[num_it] = {}  ## keep in memory successive agents costs through iteratations

        parallel_run = True

        if parallel_run:
            N = len(agents_list)
            ProcessPool = mp.Pool(processes=N)
            work = [(agent, lambdas[agent.name], params.date) for agent in agents_list]
            res = ProcessPool.map(agent_tcl_update_load_helper, work)
            results =  dict(res)

        for agent in agents_list:
            try:
                if not parallel_run:
                    path = os.path.join(os.getcwd(),'OptimisationConsomationEDF')
                    agent.tcl_update_load(lambdas[agent.name], path = path)
                else:
                    if results[agent.name] == 1:
                        raise RuntimeError(f"{agent.name} failed to run")

                path = os.path.join(os.getcwd(), 'OptimisationConsomationEDF')
                new_computed_profile,new_computed_cost = agent.read_output( path = path)
                new_computed_profile = np.array(new_computed_profile)
                new_computed_cost = np.array([new_computed_cost])

                profiles_dict_per_it[num_it][agent.name] = new_computed_profile ## keep in memory successive profiles through iteratations

                ##profile cost without lambda term
                costs_dict_per_it[num_it][agent.name]= new_computed_cost[0] - np.dot(lambdas[agent.name],new_computed_profile)


            except RuntimeError:
                profiles_dict_per_it[num_it][agent.name] = (None)

        #market value of iteration
        costs_dict_per_it[num_it]["market"] = curtailement.calcule_vobj_marche(params,agents_list, profiles_dict_per_it[num_it])

        # optimize market subproblem wrt lambda
        new_profile_aggreg = curtailement.optim_aggreg_profile_new(params,lambdas['aggregator'])
        aggregator_profiles_dict_per_it[num_it] = new_profile_aggreg
        aggregator_costs_dict_per_it[num_it] =  curtailement.fobj(params,new_profile_aggreg)

        if num_it>0 and params.fully_corrective:
            
            warm_start_step_it, memory_size = step_optimization.update_warm_start_step_it(num_it,params,step_it)

            aggregator_profiles, agent_profiles, aggregator_costs, agents_costs =step_optimization.preparation_dict_cvx_combination(num_it,memory_size,aggregator_profiles_dict_per_it,profile_aggreg,profiles_dict_per_it,
                                                                                                                    agent_averaged_profile_dict,aggregator_costs_dict_per_it,cost_aggregator_averaged,                                                                                                                    costs_dict_per_it,costs_dict_individual_averaged,agents_list)

            step_it, val = step_optimization.update_fully_corrective(params,warm_start_step_it,aggregator_profiles,agent_profiles,aggregator_costs,agents_costs,agents_list)

            print(f"Pas optimal : {step_it} value {val}")

            # Nouveau profil aggregateur
            profile_aggreg = sum([step_it[k]*aggregator_profiles[k] for k in aggregator_profiles.keys()])
            #nouveau profils agents
            for agent_bb in agents_list:
                agent_averaged_profile_dict[agent_bb.name] = sum(np.array([step_it[k]*agent_profiles[k][agent_bb.name] for k in agent_profiles.keys()]))
        else:
            step_it = step_optimization.closed_loop_step(num_it)

            for agent_bb in agents_list:
                agent_averaged_profile_dict[agent_bb.name] = step_it*profiles_dict_per_it[num_it][agent_bb.name] + (1-step_it)*agent_averaged_profile_dict[agent_bb.name]

            profile_aggreg = step_it*new_profile_aggreg + (1-step_it)*profile_aggreg

        aggreg_averaged_indiv_profiles = sum(agent_averaged_profile_dict[agent_bb.name] for agent_bb in agents_list)
        aggreg_indiv_profiles = sum(profiles_dict_per_it[num_it][agent_bb.name] for agent_bb in agents_list)

        #Computations of the new individual cost averaged cost
        costs_dict_individual_averaged = fonction_objectif.build_dico_individual_cost(agent_averaged_profile_dict,agents_list)
        cost_aggregator_averaged = curtailement.fobj(params, profile_aggreg)

        #Print profiles on reference and shaving periods
        print_profile = True
        if print_profile :
            print("\nAnswers for frank-wolfe iteration ", num_it, "(agents: ", suffix,")")

            print("Lambdas to local agents : \n",lambdas[agents_list[0].name][params.period_init[0]:params.period_finale[1]])
            print(f' Rho : {params.rho}')

            print('new aggreg indiv profiles (somme des pi chapeau):\n', aggreg_indiv_profiles[params.period_init[0]:params.period_finale[1]])

            print('new aggreg profile (p chapeau):\n', new_profile_aggreg[params.period_init[0]:params.period_finale[1]])

            print("Averaged profiles:")

            print('new aggreg averaged indiv profiles (somme des pi):\n', aggreg_averaged_indiv_profiles[params.period_init[0]:params.period_finale[1]])

            print('new averaged aggreg profile (p):\n', profile_aggreg[params.period_init[0]:params.period_finale[1]])

        avg_cost = fonction_objectif.objective_fun(params,profile_aggreg,agent_averaged_profile_dict,agents_list,costs_dict_individual_averaged)
        print(f'Cost of avg strat : {avg_cost}')

        # Calcul d'un upper bound à la distance de la valeur du problème. Upper bound donné https://www.iro.umontreal.ca/~marcotte/ARTIPS/1986_MP.pdf
        compute_upper_bound = True
        if compute_upper_bound:
            upper_bound = np.dot(np.array(lambdas['aggregator']), profile_aggreg - new_profile_aggreg)
            upper_bound = upper_bound + curtailement.fobj(params,profile_aggreg) - curtailement.fobj(params,new_profile_aggreg)
            for agent_bb in agents_list:
                upper_bound = upper_bound + np.dot(np.array(lambdas[agent_bb.name]),agent_averaged_profile_dict[agent_bb.name] - profiles_dict_per_it[num_it][agent_bb.name]  )
                local_cost_averaged_profile  = agent_bb.individual_cost(agent_averaged_profile_dict[agent_bb.name])
                upper_bound = upper_bound + local_cost_averaged_profile - costs_dict_per_it[num_it][agent_bb.name]

            print(f'Distance to the value of the problem: {upper_bound}')
            print(f'rho : {params.rho}')
            if upper_bound < 0:
                print("problem upper bound negatif")

        # Save results of the iteration
        df_it = pd.DataFrame.from_dict({'iteration': [num_it],'cost':[avg_cost],'upper_bound':[upper_bound], 'rho':params.rho})
        df_conv = pd.concat([df_conv if not df_conv.empty else None,df_it],ignore_index=True)
        df_conv.to_csv(os.path.join(params.output_dir,'convergence.csv'), sep = ';')

        num_it += 1

    if params.output_dir: ##export results
        write_dicts_to_csv(profiles_dict_per_it, costs_dict_per_it, lambdas_per_it, params.output_dir, params.date, suffix)
        
    return profiles_dict_per_it, costs_dict_per_it



