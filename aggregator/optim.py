import numpy as np
import os
import time

from .optim_utils import *
from .step_optimization import *

os.environ['XPRESS'] = '/opt/shared/agregi/env_python/bin/xpauth.xpr'


def optimize(params, agents_list, suffix: str, type_optim, function_numtry_name) :
    if type_optim=="fixed_step":
        print(f"optimisation via pas fixés de {len(agents_list)} tcls, avec un rho égal à {params.rho} et en utilisant la fonction de num_try :{function_numtry_name}")
        optim_frankwolfe_fixed_step(params, agents_list, suffix, index_function_choice=function_numtry_name)
    elif type_optim=="convexe":
        print(f"optimisation via relaxation convexe de {len(agents_list)} tcls, avec un rho égal à {params.rho}")
        optim_frankwolfe_fixed_step_relaxation_convexe(params, agents_list, suffix)
    elif type_optim=="line_search":
        print(f"optimisation via lile search de {len(agents_list)} tcls, avec un rho égal à {params.rho} et en utilisant la fonction de num_try :{function_numtry_name}")
        params.depth_fully_corrective=2
        optim_frankwolfe_line_search_and_fully_corrective(params, agents_list, suffix, index_function_choice=function_numtry_name)
    elif type_optim=="fully_corrective":
        print(f"optimisation via fully corrective de {len(agents_list)} tcls, avec un rho égal à {params.rho} et en utilisant la fonction de num_try :{function_numtry_name}")
        params.depth_fully_corrective=200
        optim_frankwolfe_line_search_and_fully_corrective(params, agents_list, suffix, index_function_choice=function_numtry_name)
    else:
        raise NameError(f"le type d'optim {type_optim} n'est pas défini") 


        


def optim_frankwolfe_fixed_step(params, agents_list, suffix: str, lambda_start=np.zeros(48), index_function_choice="constant_to_1"):

    graphiques = Graphiques()
    list_of_iterations = [Iteration(agents_list)]
    
    function_choice = {"constant_to_1": lambda x: 1, 
                       "constant_to_5": lambda x: 5,
                       "sqrt": lambda x: 1 + int(np.sqrt(x)), 
                       "linear": lambda x: x, 
                       "quadratic": lambda x:x**2}
    file_name = f"./output/optim/fixed_step/ntcl:{len(agents_list)}_function_choice:{index_function_choice}_rho:{str(params.rho)}"

    
    for num_it in range(params.max_iter_FrankWolfe):
        last_iteration:Iteration = list_of_iterations[-1]
        new_iteration = Iteration(agents_list)
        
        t_init_iter = time.time()
        print("\nIteration Frank-Wolfe : ", num_it, "(agents: ", suffix,")")
        
        # Préparation des optimisations des sous problèmes
        step_it = closed_loop_step(num_it) # i.e.  step_it = 2/(2+num_it)
        new_iteration.lambdas = derivarite_f0(params, last_iteration.averaged_load_profile_aggregator, last_iteration.stochastic_load_profiles_tcl, agents_list)

        ##### Optimisation du sous pb de l'aggrégateur #####
        new_iteration.load_profile_aggregator = optim_aggreg_profile_new(params, new_iteration.lambdas['aggregator'])
        new_iteration.cost_aggregator = fobj(params, new_iteration.load_profile_aggregator)
        # Mélange entre la solution de l'optimisation et l'ancienne solution. 
        new_iteration.averaged_load_profile_aggregator = step_it*new_iteration.load_profile_aggregator + (1-step_it)*last_iteration.averaged_load_profile_aggregator
        new_iteration.averaged_cost_aggregator = fobj(params, new_iteration.averaged_load_profile_aggregator)
        
        
        ##### Optimisation des sous problèmes de TCL #####
        # Choix stochastique des tcl qui seront optimisés
        num_try = function_choice[index_function_choice](num_it)
        tab_bernouilli, flags = get_num_tcl_to_optimize(num_it, step_it, num_try, len(agents_list))
        
        # Optimisations du profil des tcls qui ont étés tirés
        new_iteration.load_profiles_tcl,  new_iteration.costs_tcl = get_partial_optimization(flags, agents_list, new_iteration.lambdas)
        
        # Détermination du tirage parmis les différents tirage réalisés (dimension num_try de bernouilli) qui minimise la fonction à optimiser. 
        new_iteration.stochastic_load_profiles_tcl, new_iteration.stochastic_costs_tcl, cout_minimal = get_best_try(tab_bernouilli, [last_iteration, new_iteration], agents_list, params)


        ##### Save results of the iteration ##### 
        # Graphics
        
        somme_factures = sum([new_iteration.stochastic_costs_tcl[agent.name] for agent in agents_list])
        remuneration_nebef = calcule_vobj_marche(params,agents_list, new_iteration.stochastic_load_profiles_tcl)
        remuneration_nebef_uagg = fobj(params, new_iteration.averaged_load_profile_aggregator)
        print("somme_factures", somme_factures)
        print("remuneration_nebef", remuneration_nebef)
        print("remuneration_nebef_uagg", remuneration_nebef_uagg)
        
        graphiques.add_to_graphics(cout_min=cout_minimal, 
                                           time=time.time()-t_init_iter, 
                                           facture=(somme_factures+remuneration_nebef)/len(agents_list), 
                                           upper_bound=0, # compute_upper_bound(new_iteration, params, agents_list) n'a pas de sens ici
                                           num_try=num_try, 
                                           step_it=step_it,
                                           somme_pi=sum([tcl_load for key, tcl_load in new_iteration.stochastic_load_profiles_tcl.items()]), 
                                           p_agg=new_iteration.averaged_load_profile_aggregator )
        
        graphiques.create_graphic(file_name) # add a ".png"
        graphiques.save_to_file(file_name)

        # Profiles
        list_of_iterations.append(new_iteration)    

def optim_frankwolfe_line_search_and_fully_corrective(params, agents_list, suffix: str, lambda_start=np.zeros(48), index_function_choice="sqrt"):

    graphiques = Graphiques()
    list_of_iterations = [Iteration(agents_list)]
    
    function_choice = {"constant_to_1": lambda x: 1, 
                       "constant_to_5": lambda x: 5,
                       "sqrt": lambda x: 1 + int(np.sqrt(x)), 
                       "linear": lambda x: x, 
                       "quadratic": lambda x:x**2}
    memory_size = params.depth_fully_corrective 
    linesearch=False
    if memory_size==2: # linesearch
        file_name = f"./output/optim/line_search/ntcl:{len(agents_list)}_function_choice:{index_function_choice}_rho:{str(params.rho)}"
        linesearch=True
    else:
        file_name = f"./output/optim/fully_corrective/ntcl:{len(agents_list)}_function_choice:{index_function_choice}_rho:{str(params.rho)}"
    
    for num_it in range(params.max_iter_FrankWolfe):
        last_iteration:Iteration = list_of_iterations[-1]
        new_iteration = Iteration(agents_list)
        
        t_init_iter = time.time()
        print("\nIteration Frank-Wolfe : ", num_it, "(agents: ", suffix,")")
        
        # Préparation des optimisations des sous problèmes
        new_iteration.lambdas = derivarite_f0(params, last_iteration.averaged_load_profile_aggregator, last_iteration.stochastic_load_profiles_tcl, agents_list)

        ##### Optimisation des sous pb #####
        # aggrégateur : 
        new_iteration.load_profile_aggregator = optim_aggreg_profile_new(params, new_iteration.lambdas['aggregator'])
        new_iteration.cost_aggregator = fobj(params, new_iteration.load_profile_aggregator)
        # tcls : 
        flags = np.ones(len(agents_list)) # tous les tcls doivent être optimisés pour pouvoir faire du linesearch
        new_iteration.load_profiles_tcl, new_iteration.costs_tcl = get_partial_optimization(flags, agents_list, new_iteration.lambdas)

        ##### Calcul du mélange optimal #####
        if num_it==0:
            step_it = (0, 1) # on prends la première actualisation directement dans [0, 1]
        else: 
            warm_start = update_warm_start_step_it(min(memory_size, len(list_of_iterations)+1), num_it, step_it) # step_it fait ici référence à celui de l'itération de boucle précédente
            if linesearch:
                aggregator_profiles,  agent_profiles, aggregator_costs, agents_costs = preparation_dict_cvx_combination_line_search(last_iteration, new_iteration)
            else:
                aggregator_profiles,  agent_profiles, aggregator_costs, agents_costs = preparation_dict_cvx_combination_fully_corrective(list_of_iterations + [new_iteration], memory_size)
            
            step_it, _ = update_fully_corrective(params, warm_start, aggregator_profiles, agent_profiles, aggregator_costs, agents_costs, agents_list)
            print("step_it = ", step_it)
        ##### Mélange entre la solution de l'optimisation et l'ancienne solution ##### 
        if linesearch:
            # aggrégateur :
            new_iteration.averaged_load_profile_aggregator = step_it[1]*new_iteration.load_profile_aggregator + step_it[0]*last_iteration.averaged_load_profile_aggregator
            new_iteration.averaged_cost_aggregator = fobj(params, new_iteration.averaged_load_profile_aggregator)
            
            # tcls : 
            #-> Choix stochastique des tcl qui seront optimisés
            num_try = function_choice[index_function_choice](num_it)
            tab_bernouilli, _ = get_num_tcl_to_optimize(num_it, step_it[1], num_try, len(agents_list)) # on peut encore faire comme ca, mais pour fully corrective ce sera plus galere
            #-> Détermination du tirage parmis les différents tirage réalisés (dimension num_try de bernouilli) qui minimise la fonction à optimiser. 
            new_iteration.stochastic_load_profiles_tcl, new_iteration.stochastic_costs_tcl, cout_minimal = get_best_try(tab_bernouilli, [last_iteration, new_iteration], agents_list, params)

        else:
            new_iteration.averaged_load_profile_aggregator = sum([step_it[k]*iteration.load_profile_aggregator for k, iteration in enumerate(list_of_iterations + [new_iteration])])
            new_iteration.averaged_cost_aggregator = fobj(params, new_iteration.averaged_load_profile_aggregator)
            
            num_try = function_choice[index_function_choice](num_it)
            tab_bernouilli = np.random.choice(len(step_it), size=(num_try, len(agents_list)), p=step_it)

            new_iteration.stochastic_load_profiles_tcl, new_iteration.stochastic_costs_tcl, cout_minimal = get_best_try(tab_bernouilli, list_of_iterations + [new_iteration], agents_list, params, fully_corrective=True)

        ##### Save results of the iteration ##### 
        # Graphics
        graphiques.add_to_graphics(cout_min=cout_minimal, 
                                           time=time.time()-t_init_iter, 
                                           facture=(sum([new_iteration.stochastic_costs_tcl[agent.name] for agent in agents_list]) + calcule_vobj_marche(params,agents_list, new_iteration.stochastic_load_profiles_tcl))/len(agents_list), 
                                           upper_bound=compute_upper_bound(new_iteration, params, agents_list),
                                           num_try=num_try, 
                                           step_it=step_it[-1], 
                                           somme_pi=sum([tcl_load for key, tcl_load in new_iteration.stochastic_load_profiles_tcl.items()]), 
                                           p_agg=new_iteration.averaged_load_profile_aggregator )
        
        graphiques.create_graphic(file_name) # add a ".png"
        graphiques.save_to_file(file_name)

        # if num_it==0:
        #     list_of_iterations = []
        #     step_it = [1]
        
        if len(list_of_iterations) >= memory_size-1:
            list_of_iterations.pop(0) 
        list_of_iterations.append(new_iteration)

def optim_frankwolfe_fixed_step_relaxation_convexe(params, agents_list, suffix: str, lambda_start=np.zeros(48)):

    graphiques = Graphiques()
    list_of_iterations = [Iteration(agents_list)]
    
    file_name = f"./output/optim/continuous_relaxation/ntcl:{len(agents_list)}_rho:{str(params.rho)}"
    
    for num_it in range(params.max_iter_FrankWolfe):
        last_iteration:Iteration = list_of_iterations[-1]
        new_iteration = Iteration(agents_list)
        
        t_init_iter = time.time()
        print("\nIteration Frank-Wolfe : ", num_it, "(agents: ", suffix,")")
        
        # Préparation des optimisations des sous problèmes
        new_iteration.lambdas = derivarite_f0(params, last_iteration.averaged_load_profile_aggregator, last_iteration.stochastic_load_profiles_tcl, agents_list)

        ##### Optimisation des sous pb #####
        # aggrégateur : 
        new_iteration.load_profile_aggregator = optim_aggreg_profile_new(params, new_iteration.lambdas['aggregator'])
        new_iteration.cost_aggregator = fobj(params, new_iteration.load_profile_aggregator)
        # tcls : 
        flags = np.ones(len(agents_list)) # tous les tcls doivent être optimisés pour pouvoir faire du linesearch
        new_iteration.load_profiles_tcl, new_iteration.costs_tcl = get_partial_optimization(flags, agents_list, new_iteration.lambdas, mode_convexe=True)

        ##### step_it fixé #####
        step_it = closed_loop_step(num_it) # i.e.  step_it = 2/(2+num_it)

        ##### Mélange entre la solution de l'optimisation et l'ancienne solution ##### 
        new_iteration.averaged_load_profile_aggregator = step_it*new_iteration.load_profile_aggregator + (1-step_it)*last_iteration.averaged_load_profile_aggregator
        new_iteration.averaged_cost_aggregator = fobj(params, new_iteration.averaged_load_profile_aggregator)
        
        # tcls : 
        new_iteration.stochastic_load_profiles_tcl = {agent.name: step_it*new_iteration.load_profiles_tcl[agent.name] + (1-step_it)*last_iteration.stochastic_load_profiles_tcl[agent.name] for agent in agents_list} # les noms de variables sont un peu bizarre, plus que les profiles sont moyenés (averaged) et non choisi de manière stochastique
        new_iteration.stochastic_costs_tcl = {agent.name: step_it*new_iteration.costs_tcl[agent.name] + (1-step_it)*last_iteration.stochastic_costs_tcl[agent.name] for agent in agents_list} # same

        individual_costs = build_dico_individual_cost(new_iteration.stochastic_load_profiles_tcl, agents_list)
        cout_minimal = objective_fun(params, new_iteration.averaged_load_profile_aggregator, new_iteration.stochastic_load_profiles_tcl, agents_list, individual_costs)
        

        ##### Save results of the iteration ##### 
        # Graphics
        graphiques.add_to_graphics(cout_min=cout_minimal, 
                                           time=time.time()-t_init_iter, 
                                           facture=(sum([new_iteration.stochastic_costs_tcl[agent.name] for agent in agents_list]) + calcule_vobj_marche(params,agents_list, new_iteration.stochastic_load_profiles_tcl))/len(agents_list), 
                                           upper_bound=compute_upper_bound(new_iteration, params, agents_list),
                                           num_try=0, 
                                           step_it=step_it,
                                           somme_pi=sum([tcl_load for key, tcl_load in new_iteration.stochastic_load_profiles_tcl.items()]), 
                                           p_agg=new_iteration.averaged_load_profile_aggregator )
        
        graphiques.create_graphic(file_name) # add a ".png"
        graphiques.save_to_file(file_name)

        # Profiles
        list_of_iterations.append(new_iteration)    


