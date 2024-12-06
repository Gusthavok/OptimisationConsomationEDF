import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
import copy 

from .recombi import recombine

from .fonction_objectif import build_dico_individual_cost, objective_fun
from .curtailement import *
from .step_optimization import *

os.environ['XPRESS'] = '/opt/shared/agregi/env_python/bin/xpauth.xpr'

affichage_y_min=.1


def optimize(params, agents_list, suffix: str) :
    optim_frankwolfe_fixed_step(params, agents_list, suffix)

class Iteration:
    def __init__(self, agents_list):
        # Optimized points (usefull to store for fully corrective and line search)
        self.load_profiles_tcl = {} 
        self.costs_tcl = {} 
        self.lambdas = {} 
        self.load_profile_aggregator = np.zeros(48)
        self.cost_aggregator=0
        
        # Solution proposed by the algorithm (throught stochastic selection for tcl) : 
        self.stochastic_load_profiles_tcl = { agent_bb.name: np.zeros(48) for agent_bb in agents_list}
        self.stochastic_costs_tcl = {}
        self.averaged_load_profile_aggregator = np.zeros(48)
        self.averaged_cost_aggregator = 0

    # profiles_dict_per_it = {} ## keep in memory successive profiles through iteratations
    # costs_dict_per_it = {}  ## keep in memory successive agents costs through iteratations
    # lambdas_per_it = {}
    # aggregator_profiles_dict_per_it = {} ## keep in memory successive profiles through iteratations
    # aggregator_costs_dict_per_it= {}  ## keep in memory successive agregator costs through iteratations
    

class Graphiques:
    def __init__(self):
        self.cout_min_liste = []
        self.time_per_iteration = []
        self.facture_list = []
        self.upper_bound_list = []
        self.num_try_list = []
        self.step_it_list = []

    def add_to_graphics(self, cout_min, time, facture, upper_bound, num_try, step_it):
        self.cout_min_liste.append(cout_min)
        self.time_per_iteration.append(time)
        self.facture_list.append(facture)
        self.upper_bound_list.append(upper_bound)
        self.num_try_list.append(num_try)
        self.step_it_list.append(step_it)
    
    def save_to_file(self, filename):
        """Enregistre les 6 listes dans un fichier texte."""
        data = {
            'cout_min_liste': self.cout_min_liste,
            'time_per_iteration': self.time_per_iteration,
            'facture_list': self.facture_list,
            'upper_bound_list': self.upper_bound_list,
            'num_try_list': self.num_try_list,
            'step_it_list': self.step_it_list
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    
    def create_graphic(self):
        ## Sauvegarde fichier textuel : 
        self.save_to_file("./lists_for_graphic")
        
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
            

        ## Sauvegarder les graphes
        line.set_xdata(range(len(self.cout_min_liste)))
        line2.set_xdata(range(len(self.facture_list)))
        line3.set_xdata(range(len(self.upper_bound_list)))
        line4.set_xdata(range(len(self.time_per_iteration)))
        line5.set_xdata(range(len(self.num_try_list)))
        line6.set_xdata(range(len(self.step_it_list)))
        line.set_ydata(self.cout_min_liste)
        line2.set_ydata(self.facture_list)
        line3.set_ydata(self.upper_bound_list)
        line4.set_ydata(self.time_per_iteration)
        line5.set_ydata(self.num_try_list)
        line6.set_ydata(self.step_it_list)
        
        for a in ax.flat:
            a.set_xlim(0, len(self.cout_min_liste))
            
        
        ax[0,0].set_ylim(-3000, 6000)
        ax[0,1].set_ylim(min(self.facture_list)-5, max(self.facture_list)+5)
        ax[1,0].set_ylim(0, max(self.upper_bound_list))
        ax[1,1].set_ylim(0, max(self.time_per_iteration))
        ax[2,0].set_ylim(0, max(self.num_try_list)+1)
        ax[2,1].set_ylim(0, max(self.step_it_list))
        
        plt.savefig('plot_100_tcl_fixed_step.png')   
        time.sleep(.05)     

    def reload_from_file(self, filename):
        """Récupère les 6 listes à partir d'un fichier texte."""
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.cout_min_liste = data.get('cout_min_liste', [])
            self.time_per_iteration = data.get('time_per_iteration', [])
            self.facture_list = data.get('facture_list', [])
            self.upper_bound_list = data.get('upper_bound_list', [])
            self.num_try_list = data.get('num_try_list', [])
            self.step_it_list = data.get('step_it_list', [])


def get_num_tcl_to_optimize(num_it, step_it, num_try, number_of_agent):
    if num_it == 0: # A la première iteration, on optimise tous les tcls. 
        tab_bernouilli = np.ones((1, number_of_agent))
    else:
        tab_bernouilli = np.random.binomial(n=1, p=step_it, size=(num_try, number_of_agent))
        tab_bernouilli = np.concatenate((np.zeros(shape=(1, number_of_agent)), tab_bernouilli), axis=0) # On se garde la possibilité de ne pas faire de modification. (Bien sur l'aggregateur bougera quand meme)
    flags = np.sum(tab_bernouilli, axis=0)
    return tab_bernouilli, flags

def get_partial_optimization(flags, agents_list, lambdas):
    profiles_dict = {}
    costs_dict = {}
    for k, agent in enumerate(agents_list):
        if flags[k]:
            path = os.path.join(os.getcwd(),'')
            agent.tcl_update_load(lambdas[agent.name], path = path)

            path = os.path.join(os.getcwd(), '')
            new_computed_profile,new_computed_cost, facture = agent.read_output( path = path)
            new_computed_profile = np.array(new_computed_profile)
            new_computed_cost = np.array([new_computed_cost])

            profiles_dict[agent.name] = new_computed_profile ## keep in memory successive profiles through iteratations

            costs_dict[agent.name]= new_computed_cost[0] - np.dot(lambdas[agent.name],new_computed_profile)
    
    return profiles_dict, costs_dict

def get_best_try(tab_bernouilli: np.ndarray, new_iteration : Iteration, last_iteration: Iteration, agents_list, params):         
    cout_min=None
    best_stochastic_profiles = {}
    best_stochastic_costs = {}

    for test in range(tab_bernouilli.shape[0]):
        # Selection associé au tirage de bernouilli tab_bernouilli[test] : 
        test_stochastic_profiles={}
        test_stochastic_costs={}
        for k, agent_bb in enumerate(agents_list):
            if tab_bernouilli[test][k]:
                test_stochastic_profiles[agent_bb.name] = new_iteration.load_profiles_tcl[agent_bb.name] 
                test_stochastic_costs[agent_bb.name] = new_iteration.costs_tcl[agent_bb.name] 
            else:
                test_stochastic_profiles[agent_bb.name] = last_iteration.stochastic_load_profiles_tcl[agent_bb.name] 
                test_stochastic_costs[agent_bb.name] = last_iteration.stochastic_costs_tcl[agent_bb.name] 

        # Couts de ce test : 
        test_individual_costs = build_dico_individual_cost(test_stochastic_profiles, agents_list)

        cout_test = objective_fun(params, new_iteration.averaged_load_profile_aggregator, test_stochastic_profiles, agents_list, test_individual_costs)
        
        if cout_min is None or cout_min>cout_test:
            cout_min=cout_test
            best_stochastic_profiles = copy.deepcopy(test_stochastic_profiles)
            best_stochastic_costs = copy.deepcopy(test_stochastic_costs)
            
    return best_stochastic_profiles, best_stochastic_costs, cout_min

def compute_upper_bound(iteration: Iteration, params, agents_list):
    # Calcul d'un upper bound à la distance de la valeur du problème. Upper bound donné https://www.iro.umontreal.ca/~marcotte/ARTIPS/1986_MP.pdf
    upper_bound = np.dot(np.array(iteration.lambdas['aggregator']), iteration.averaged_load_profile_aggregator - iteration.load_profile_aggregator)
    upper_bound = upper_bound + fobj(params, iteration.averaged_load_profile_aggregator) - fobj(params, iteration.load_profile_aggregator)
    for agent_bb in agents_list:
        upper_bound = upper_bound + np.dot(np.array(iteration.lambdas[agent_bb.name]), iteration.stochastic_load_profiles_tcl[agent_bb.name] - iteration.load_profiles_tcl[agent_bb.name]  )
        local_cost_averaged_profile  = agent_bb.individual_cost(iteration.stochastic_load_profiles_tcl[agent_bb.name])
        upper_bound = upper_bound + local_cost_averaged_profile - iteration.costs_tcl[agent_bb.name]

    print(f'Distance to the value of the problem: {upper_bound}')
    print(f'rho : {params.rho}')
    print(f'rémunération : {fobj(params, iteration.averaged_load_profile_aggregator)}')
    if upper_bound < 0:
        print("problem upper bound negatif")
    return upper_bound

def optim_frankwolfe_fixed_step(params, agents_list, suffix: str, lambda_start=np.zeros(48)):

    graphiques = Graphiques()
    list_of_iterations = [Iteration(agents_list)]

    
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
        num_try = 1 # int(1+np.sqrt(num_it))
        tab_bernouilli, flags = get_num_tcl_to_optimize(num_it, step_it, num_try, len(agents_list))
        
        # Optimisations du profil des tcls qui ont étés tirés
        new_iteration.load_profiles_tcl,  new_iteration.costs_tcl = get_partial_optimization(flags, agents_list, new_iteration.lambdas)
        
        # Détermination du tirage parmis les différents tirage réalisés (dimension num_try de bernouilli) qui minimise la fonction à optimiser. 
        new_iteration.stochastic_load_profiles_tcl, new_iteration.stochastic_costs_tcl, cout_minimal = get_best_try(tab_bernouilli, new_iteration, last_iteration, agents_list, params)


        ##### Save results of the iteration ##### 
        # Graphics
        graphiques.add_to_graphics(cout_min=cout_minimal, 
                                           time=time.time()-t_init_iter, 
                                           facture=(sum([new_iteration.stochastic_costs_tcl[agent.name] for agent in agents_list]) + calcule_vobj_marche(params,agents_list, new_iteration.stochastic_load_profiles_tcl))/len(agents_list), 
                                           upper_bound=0, # compute_upper_bound(new_iteration, params, agents_list) n'a pas de sens ici
                                           num_try=num_try, 
                                           step_it=step_it)
        graphiques.create_graphic()
        graphiques.save_to_file("./graphics_lists")

        # Profiles
        list_of_iterations.append(new_iteration)    

