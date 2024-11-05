import numpy as np
import os
import time
import pandas as pd
from scipy.optimize import  Bounds, LinearConstraint, minimize
import matplotlib.pyplot as plt
import copy 
from tcl.tcl import Tcl

from .recombi import recombine

from .fonction_objectif import build_dico_individual_cost, objective_fun
from .curtailement import *
from .step_optimization import *

os.environ['XPRESS'] = '/opt/shared/agregi/env_python/bin/xpauth.xpr'

import multiprocessing as mp

affichage_y_min=.1


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
        path = os.getcwd()
        agent.tcl_update_load(lambdas, path = path)
    except RuntimeError:
        return agent.name, 1
    return agent.name, 0



def optim_frankwolfe(params, agents_list, suffix: str, lambda_start=np.zeros(48)):
    ## affichage
    fig, ax = plt.subplots(3,2, figsize= (10,8))
    line, = ax[0,0].plot([], label='cost evolution')
    line2, = ax[0,1].plot([], label='average invoice (if full retribution)')
    line3, = ax[1,0].plot([], label='upper_bound evolution')
    line4, = ax[1,1].plot([], label='time per iteration')
    line5, = ax[2,0].plot([], label='num_try evolution')
    line6, = ax[2,1].plot([], label='step_it evolution')
    
    for a in ax.flat:
        a.set_xlabel('Iteration')
        a.set_xlim(0, 10)  # Ajuster la limite x si nécessaire
        a.set_ylim(affichage_y_min, affichage_y_min+1)   # Ajuster la limite y si nécessaire
        
    ax[0,0].set_ylabel('cout min')
    ax[0,1].set_ylabel('average cost')
    ax[1,0].set_ylabel('upper_bound')
    ax[1,0].set_yscale('log')    
    ax[1,1].set_ylabel('time')
    ax[2,0].set_ylabel('num_try')
    ax[2,1].set_ylabel('step_it')
    
    for a in ax.flat:
        a.legend()

    profile_aggreg = np.zeros(48)
    num_it = 0
    step_it = [0]

    profiles_dict_per_it = {} ## keep in memory successive profiles through iteratations
    costs_dict_per_it = {}  ## keep in memory successive agents costs through iteratations
    lambdas_per_it = {}
    aggregator_profiles_dict_per_it = {} ## keep in memory successive profiles through iteratations
    aggregator_costs_dict_per_it= {}  ## keep in memory successive agregator costs through iteratations
    cout_min_liste=[]
    time_per_iteration = []
    facture_list = []
    upper_bound_list = []
    num_try_list = []
    step_it_list = []

    df_conv = pd.DataFrame(data = None, columns=['iteration','cost','upper_bound','rho'])

    agent_averaged_profile_dict = { agent_bb.name: np.zeros(48) for agent_bb in agents_list}
    
    while (num_it <= params.max_iter_FrankWolfe):
        t_init_iter = time.time()
        print("\nIteration Frank-Wolfe : ", num_it, "(agents: ", suffix,")")

        # Compute the new signal to send to each agent
        lambdas = derivarite_f0(params, profile_aggreg,agent_averaged_profile_dict, agents_list)
        lambdas_per_it[num_it] = lambdas

        profiles_dict_per_it[num_it] = {} ## keep in memory successive profiles through iteratations
        costs_dict_per_it[num_it] = {}
        if num_it>0:
            for key in profiles_dict_per_it[num_it-1]:
                profiles_dict_per_it[num_it][key] = profiles_dict_per_it[num_it-1][key]  ## keep in memory successive agents costs through iteratations

            for key in costs_dict_per_it[num_it-1]:
                costs_dict_per_it[num_it][key] = costs_dict_per_it[num_it-1][key]  ## keep in memory successive agents costs through iteratations

        
        if num_it<=1:
            num_try=1
            num_try_list.append(num_try)
            bernoulli=np.ones(shape=(1, len(agents_list)))
        else:
            num_try = 1 # int(1+np.sqrt(num_it))
            num_try_list.append(num_try)
            bernoulli = np.random.binomial(n=1, p=step_it, size=(num_try, len(agents_list)))
            bernoulli = np.concatenate((np.zeros(shape=(1, len(agents_list))), bernoulli), axis=0)

        flags = np.sum(bernoulli, axis=0)


        parallel_run = False
    
        if parallel_run:
            N = np.sum(flags != 0)
            work = [(agent, lambdas[agent.name], params.date) for k, agent in enumerate(agents_list) if flags[k]]

            with mp.Pool(processes=N) as ProcessPool:
                res = ProcessPool.map(agent_tcl_update_load_helper, work)
            results =  dict(res)
            
        
        for k, agent in enumerate(agents_list):
            try:
                if flags[k]:
                    if not parallel_run:
                        path = os.path.join(os.getcwd(),'')
                        agent.tcl_update_load(lambdas[agent.name], path = path)
                    else:
                        if results[agent.name] == 1:
                            raise RuntimeError(f"{agent.name} failed to run")

                path = os.path.join(os.getcwd(), '')
                new_computed_profile,new_computed_cost, facture = agent.read_output( path = path)
                new_computed_profile = np.array(new_computed_profile)
                new_computed_cost = np.array([new_computed_cost])

                profiles_dict_per_it[num_it][agent.name] = new_computed_profile ## keep in memory successive profiles through iteratations

                ##profile cost without lambda term
                if flags[k]:
                    costs_dict_per_it[num_it][agent.name]= new_computed_cost[0] - np.dot(lambdas[agent.name],new_computed_profile)
                else:
                    costs_dict_per_it[num_it][agent.name]= costs_dict_per_it[num_it-1][agent.name]

            except RuntimeError:
                profiles_dict_per_it[num_it][agent.name] = (None)


        #market value of iteration
        costs_dict_per_it[num_it]["market"] = calcule_vobj_marche(params,agents_list, profiles_dict_per_it[num_it])

        # optimize market subproblem wrt lambda
        new_profile_aggreg = optim_aggreg_profile_new(params,lambdas['aggregator'])
        aggregator_profiles_dict_per_it[num_it] = new_profile_aggreg
        aggregator_costs_dict_per_it[num_it] =  fobj(params,new_profile_aggreg)

        if num_it>0 and params.fully_corrective:
            
            warm_start_step_it, memory_size = update_warm_start_step_it(num_it,params,step_it)

            aggregator_profiles, agent_profiles, aggregator_costs, agents_costs =preparation_dict_cvx_combination(num_it,memory_size,aggregator_profiles_dict_per_it,profile_aggreg,profiles_dict_per_it,
                                                                                                                    agent_averaged_profile_dict,aggregator_costs_dict_per_it,cost_aggregator_averaged, 
                                                                                                                    costs_dict_per_it,costs_dict_individual_averaged,agents_list)

            step_it, val = update_fully_corrective(params,warm_start_step_it,aggregator_profiles,agent_profiles,aggregator_costs,agents_costs,agents_list)
            step_it_list.append(step_it)
            print(f"Pas optimal : {step_it} value {val}")

            cumulative_intervals = np.cumsum(step_it)
            r = np.random.random()
            result = np.searchsorted(cumulative_intervals, r) + 1
            # Nouveau profil aggregateur
            profile_aggreg = aggregator_profiles[result]
            #nouveau profils agents
            for agent_bb in agents_list:
                r = np.random.random()
                result = np.searchsorted(cumulative_intervals, r) + 1
                agent_averaged_profile_dict[agent_bb.name] = agent_profiles[result][agent_bb.name]
        else:
            step_it = closed_loop_step(num_it)
            step_it_list.append(step_it)
            cout_min=None
            agent_averaged_profile_dict_best_test = {}

            profile_aggreg = step_it*new_profile_aggreg + (1-step_it)*profile_aggreg

            for test in range(num_try):
                agent_averaged_profile_dict_test={}
                
                for k, agent_bb in enumerate(agents_list):
                    if bernoulli[test][k]:
                        agent_averaged_profile_dict_test[agent_bb.name] = profiles_dict_per_it[num_it][agent_bb.name] 
                    else:
                        agent_averaged_profile_dict_test[agent_bb.name] = agent_averaged_profile_dict[agent_bb.name] 

                costs_dict_individual_averaged_test= build_dico_individual_cost(agent_averaged_profile_dict_test,agents_list)

                cout_test = objective_fun(params,profile_aggreg,agent_averaged_profile_dict_test,agents_list,costs_dict_individual_averaged_test)
                
                if cout_min is None or cout_min>cout_test:
                    cout_min=cout_test
                    agent_averaged_profile_dict_best_test = copy.deepcopy(agent_averaged_profile_dict_test)
                    
            agent_averaged_profile_dict=agent_averaged_profile_dict_best_test
            cout_min_liste.append(cout_min)

            
        aggreg_averaged_indiv_profiles = sum(agent_averaged_profile_dict[agent_bb.name] for agent_bb in agents_list)
        aggreg_indiv_profiles = sum(profiles_dict_per_it[num_it][agent_bb.name] for agent_bb in agents_list)

        #Computations of the new individual cost averaged cost
        costs_dict_individual_averaged = build_dico_individual_cost(agent_averaged_profile_dict,agents_list)
        cost_aggregator_averaged = fobj(params, profile_aggreg)

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

        avg_cost = objective_fun(params,profile_aggreg,agent_averaged_profile_dict,agents_list,costs_dict_individual_averaged)
        print(f'Cost of avg strat : {avg_cost}')
        
        facture_list.append((sum([costs_dict_per_it[num_it][agent.name] for agent in agents_list])+costs_dict_per_it[num_it]["market"])/len(agents_list))
        print("costs_dict_per_it[num_it]['market']", costs_dict_per_it[num_it]["market"])# négatif
        print("sum([costs_dict_per_it[num_it][agent.name]", costs_dict_per_it[num_it][agents_list[0].name]) #positif
        # Calcul d'un upper bound à la distance de la valeur du problème. Upper bound donné https://www.iro.umontreal.ca/~marcotte/ARTIPS/1986_MP.pdf
        compute_upper_bound = True
        if compute_upper_bound:
            upper_bound = np.dot(np.array(lambdas['aggregator']), profile_aggreg - new_profile_aggreg)
            upper_bound = upper_bound + fobj(params,profile_aggreg) - fobj(params,new_profile_aggreg)
            for agent_bb in agents_list:
                upper_bound = upper_bound + np.dot(np.array(lambdas[agent_bb.name]),agent_averaged_profile_dict[agent_bb.name] - profiles_dict_per_it[num_it][agent_bb.name]  )
                local_cost_averaged_profile  = agent_bb.individual_cost(agent_averaged_profile_dict[agent_bb.name])
                upper_bound = upper_bound + local_cost_averaged_profile - costs_dict_per_it[num_it][agent_bb.name]

            print(f'Distance to the value of the problem: {upper_bound}')
            print(f'rho : {params.rho}')
            print(f'rémunération : {fobj(params,profile_aggreg)}')
            if upper_bound < 0:
                print("problem upper bound negatif")

        # Save results of the iteration
        upper_bound_list.append(max(10, upper_bound))
        df_it = pd.DataFrame.from_dict({'iteration': [num_it],'cost':[avg_cost],'upper_bound':[upper_bound], 'rho':params.rho})
        df_conv = pd.concat([df_conv if not df_conv.empty else None,df_it],ignore_index=True)
        df_conv.to_csv(os.path.join(params.output_dir,'convergence.csv'), sep = ';')
        time_lenght_iter = time.time()-t_init_iter
        time_per_iteration.append(time_lenght_iter)


        ## Sauvegarder les graphes
        line.set_xdata(range(len(cout_min_liste)))
        line2.set_xdata(range(len(facture_list)))
        line3.set_xdata(range(len(upper_bound_list)))
        line4.set_xdata(range(len(time_per_iteration)))
        line5.set_xdata(range(len(num_try_list)))
        line6.set_xdata(range(len(step_it_list)))
        line.set_ydata(cout_min_liste)
        line2.set_ydata(facture_list)
        line3.set_ydata(upper_bound_list)
        line4.set_ydata(time_per_iteration)
        line5.set_ydata(num_try_list)
        line6.set_ydata(step_it_list)
        
        for a in ax.flat:
            a.set_xlim(0, len(cout_min_liste))
            
        
        ax[0,0].set_ylim(-3000, 6000)
        ax[0,1].set_ylim(min(facture_list)-5, max(facture_list)+5)
        ax[1,0].set_ylim(0, max(upper_bound_list))
        ax[1,1].set_ylim(0, max(time_per_iteration))
        ax[2,0].set_ylim(0, max(num_try_list)+1)
        ax[2,1].set_ylim(0, max(step_it_list))
        
        plt.savefig('plot_fully_corrective_sto.png')   
        time.sleep(.1)     
        
        num_it += 1


    if params.output_dir: ##export results
        write_dicts_to_csv(profiles_dict_per_it, costs_dict_per_it, lambdas_per_it, params.output_dir, params.date, suffix)
        
    return profiles_dict_per_it, costs_dict_per_it



