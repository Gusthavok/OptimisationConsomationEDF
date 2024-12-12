import pickle
import copy 
import matplotlib.pyplot as plt
import numpy as np


from .fonction_objectif import build_dico_individual_cost, objective_fun
from .curtailement import *

class Iteration:
    def __init__(self, agents_list):
        # Optimized points (usefull to store for fully corrective and line search)
        self.load_profiles_tcl = { agent_bb.name: np.zeros(48) for agent_bb in agents_list}
        self.costs_tcl = {agent_bb.name: 0 for agent_bb in agents_list}
        self.lambdas = {} 
        self.load_profile_aggregator = np.zeros(48)
        self.cost_aggregator=0
        
        # Solution proposed by the algorithm (throught stochastic selection for tcl) : 
        self.stochastic_load_profiles_tcl = { agent_bb.name: np.zeros(48) for agent_bb in agents_list}
        self.stochastic_costs_tcl = {agent_bb.name: 0 for agent_bb in agents_list}
        self.averaged_load_profile_aggregator = np.zeros(48)
        self.averaged_cost_aggregator = 0

class Graphiques:
    def __init__(self):
        self.cout_min_liste = []
        self.time_per_iteration = []
        self.facture_list = []
        self.upper_bound_list = []
        self.num_try_list = []
        self.step_it_list = []
        self.somme_pi_list = []
        self.p_agg_list = []

    def add_to_graphics(self, cout_min, time, facture, upper_bound, num_try, step_it, somme_pi, p_agg):
        self.cout_min_liste.append(cout_min)
        self.time_per_iteration.append(time)
        self.facture_list.append(facture)
        self.upper_bound_list.append(upper_bound)
        self.num_try_list.append(num_try)
        self.step_it_list.append(step_it)
        self.somme_pi_list.append(somme_pi)
        self.p_agg_list.append(p_agg)
    
    def save_to_file(self, filename):
        """Enregistre les 6 listes dans un fichier texte."""
        data = {
            'cout_min_liste': self.cout_min_liste,
            'time_per_iteration': self.time_per_iteration,
            'facture_list': self.facture_list,
            'upper_bound_list': self.upper_bound_list,
            'num_try_list': self.num_try_list,
            'step_it_list': self.step_it_list, 
            'somme_pi_list': self.somme_pi_list, 
            'p_agg_list': self.p_agg_list,
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

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
            self.somme_pi_list = data.get('somme_pi_list', [])
            self.p_agg_list = data.get('p_agg_list', [])
  
    def create_graphic(self, plot_name):        
        ## creation des supports
        fig, ax = plt.subplots(3,2, figsize= (10,8))
        line, = ax[0,0].plot([], label='cost evolution')
        line2, = ax[0,1].plot([], label='average invoice (if full retribution)')
        line3, = ax[1,0].plot([], label='upper_bound evolution')
        line4, = ax[1,1].plot([], label='time per iteration')
        line5, = ax[2,0].plot([], label='num_try evolution')
        line6, = ax[2,1].plot([], label='step_it evolution')
        
        ax[0,0].set_ylabel('cout min')
        ax[0,1].set_ylabel('average cost')
        ax[1,0].set_ylabel('upper_bound')
        ax[1,0].set_yscale('log')    
        ax[1,1].set_ylabel('time')
        ax[2,0].set_ylabel('num_try')
        ax[2,1].set_ylabel('step_it')
        
        for a in ax.flat:
            a.legend()
            

        ## ajouts des données
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
        
        ## Set up des axes
        for a in ax.flat:
            a.set_xlim(0, len(self.cout_min_liste))
            
        ax[0,0].set_ylim(-3000, 6000)
        ax[0,1].set_ylim(min(self.facture_list)-5, max(self.facture_list)+5)
        ax[1,0].set_ylim(0, max(self.upper_bound_list))
        ax[1,1].set_ylim(0, max(self.time_per_iteration))
        ax[2,0].set_ylim(0, max(self.num_try_list)+1)
        ax[2,1].set_ylim(0, max(self.step_it_list))
        
        ## Sauvegarde
        plt.savefig(f'{plot_name}.png')   
        time.sleep(.05)     

# fonction pour récupérer la liste d'instance de graphiques à partir de la lise des noms
def load_graphics_from_files(file_list):
    graphics_instances = []
    for file_name, legend_name in file_list:
        graphic = Graphiques()
        graphic.reload_from_file(file_name)
        graphics_instances.append((graphic, legend_name))
    return graphics_instances

# Fonction pour tracer des courbes superposées.
def create_superposed_graphics(graphics_instances, list_names, filename, x_log=False, y_log=False, y_symlog=False, linthreshy=1, y_min=None, y_max=None):
    """
    Trace et sauvegarde un graphique avec des courbes superposées.

    Args:
        graphics_instances (list): Liste de tuples (instance de Graphiques, nom pour la légende).
        list_names (list): Noms des listes à tracer (par exemple ['cout_min_liste']).
        filename (str): Nom du fichier de sortie pour le graphique.
        x_log (bool, optional): Si True, axe des abscisses en échelle logarithmique.
        y_log (bool, optional): Si True, axe des ordonnées en échelle logarithmique.
        y_symlog (bool, optional): Si True, axe des ordonnées en échelle symétrique logarithmique.
        linthreshy (float, optional): Seuil de transition entre échelle linéaire et logarithmique pour `symlog`.
        y_min (float, optional): Valeur minimale pour l'axe des ordonnées.
        y_max (float, optional): Valeur maximale pour l'axe des ordonnées.
    """
    # Création de la figure
    plt.figure(figsize=(10, 6))

    for list_name in list_names:
        for graphic, legend_name in graphics_instances:
            # Récupération de la liste à tracer
            data = getattr(graphic, list_name, None)
            if data is not None and len(data) > 0:
                # Ajout de la courbe au graphique
                plt.plot(range(len(data)), data, label=f"{legend_name}")

    # Configuration des axes
    if x_log:
        plt.xscale('log')
    if y_symlog:
        plt.yscale('symlog', linthresh=linthreshy)
    elif y_log:
        plt.yscale('log')

    if y_min is not None or y_max is not None:
        plt.ylim(y_min, y_max)

    # Légende et labels
    plt.xlabel('Iterations')
    plt.ylabel('Valeurs')
    plt.title('Superposition des courbes')
    plt.legend()

    # Sauvegarde du graphique
    plt.savefig(filename)
    plt.close()


def get_num_tcl_to_optimize(num_it, step_it, num_try, number_of_agent):
    if num_it == 0: # A la première iteration, on optimise tous les tcls. 
        tab_bernouilli = np.ones((1, number_of_agent))
    else:
        tab_bernouilli = np.random.binomial(n=1, p=step_it, size=(num_try, number_of_agent))
        tab_bernouilli = np.concatenate((np.zeros(shape=(1, number_of_agent)), tab_bernouilli), axis=0) # On se garde la possibilité de ne pas faire de modification. (Bien sur l'aggregateur bougera quand meme)
    flags = np.sum(tab_bernouilli, axis=0)
    return tab_bernouilli, flags

def get_partial_optimization(flags, agents_list, lambdas, mode_convexe=False):
    profiles_dict = {}
    costs_dict = {}
    for k, agent in enumerate(agents_list):
        if flags[k]:
            path = os.path.join(os.getcwd(),'')
            agent.tcl_update_load(lambdas[agent.name], mode_convexe=mode_convexe, path = path)

            path = os.path.join(os.getcwd(), '')
            new_computed_profile,new_computed_cost, facture = agent.read_output( path = path)
            new_computed_profile = np.array(new_computed_profile)
            new_computed_cost = np.array([new_computed_cost])

            profiles_dict[agent.name] = new_computed_profile ## keep in memory successive profiles through iteratations

            costs_dict[agent.name]= new_computed_cost[0] - np.dot(lambdas[agent.name],new_computed_profile)
    
    return profiles_dict, costs_dict

def get_best_try(tab_bernouilli: np.ndarray, liste_iteration : list[Iteration], agents_list, params, fully_corrective=False):         
    cout_min=None
    best_stochastic_profiles = {}
    best_stochastic_costs = {}

    for test in range(tab_bernouilli.shape[0]):
        # Selection associé au tirage de bernouilli tab_bernouilli[test] : 
        test_stochastic_profiles={}
        test_stochastic_costs={}
        for k, agent_bb in enumerate(agents_list):
            iteration_indice = round(tab_bernouilli[test][k])
            print("iteration_indice", iteration_indice)
            if (iteration_indice == len(liste_iteration)) or (fully_corrective): 
                test_stochastic_profiles[agent_bb.name] = liste_iteration[iteration_indice].load_profiles_tcl[agent_bb.name] 
                test_stochastic_costs[agent_bb.name] = liste_iteration[iteration_indice].costs_tcl[agent_bb.name] 
            else: # pour linesearch, on prends le profil moyenné précédent
                test_stochastic_profiles[agent_bb.name] = liste_iteration[iteration_indice].stochastic_load_profiles_tcl[agent_bb.name] 
                test_stochastic_costs[agent_bb.name] = liste_iteration[iteration_indice].stochastic_costs_tcl[agent_bb.name] 

        # Couts de ce test : 
        test_individual_costs = build_dico_individual_cost(test_stochastic_profiles, agents_list)

        cout_test = objective_fun(params, liste_iteration[-1].averaged_load_profile_aggregator, test_stochastic_profiles, agents_list, test_individual_costs)
        
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
