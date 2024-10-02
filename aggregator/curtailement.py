import numpy as np
import os
import time
import pandas as pd

from linopy import Model


def optim_aggreg_profile_new(params,lambdas: np.array):
    #Model linopy
    m = Model()

    ##Profil agrege: p[t]=profil agrege au pas t
    p={}
    expr_obj=0
    for t in range(params.T):
        p[t]= m.add_variables(lower=params.pmin, upper=params.pmax, name="p_{}".format(t))
        m.add_constraints(p[t] >= params.pmin) #/!\ necessaire sinon linopy plante sur les modeles avec lambda=0
        expr_obj+=p[t]*(lambdas[t])
        if t < params.period_init[0] or t>=params.period_finale[1]:
            m.add_constraints(p[t] == 0)

    # market_obj = add_expr_fonction_objectif(params, m, expr_obj, p)
   # expr_obj += market_obj
    expr_obj = add_expr_fonction_objectif(params, m, expr_obj, p)
    m.add_objective(expr_obj)
    m.solve(solution_fn=params.output_dir+"\\_sol_log_xpress.txt", log_fn=params.output_dir+"\\_log_xpress.txt")

    loadAgg = np.array([p[t].solution for t in range(params.T)])
    return loadAgg



#mean of vector p on period = (t1, t2)
def mean_on_period(p, period):
    return sum(p[period[0]:period[1]])/(period[1] - period[0])

#pour une solution agregee donnee, calcule le gain marché associé
def fobj(params,p):
    pref_init=mean_on_period(p, params.period_init)
    pref_final=mean_on_period(p, params.period_finale)
    pref=min(pref_init,pref_final)
    valo=0
    rectangle=0
    for t in range(params.period_eff[0], params.period_eff[1], params.largeur_rectangles):
        pmax=max(p[t:t+params.largeur_rectangles])
        valo+=params.spot_prices[rectangle]*(pref-pmax)
        rectangle+=1
    return -valo


#pour une solution donnée pour un ensemble d'agents, calcule le gain marché associé
def calcule_vobj_marche(params, list_agents, profils) :
    p_agrege=sum([profils[agent.name] for agent in list_agents])
    return fobj(params, p_agrege)



#terme -m(p) dependant de l'agregat p dans la fonction objectif (rémunération marché)
#input:
    #model linopy
    #expression objectif à laquelle ajouter l'expression de -m(p)
    #p_agrege: variable p_agrege

def add_expr_fonction_objectif(params, model, expr_obj, p_agrege) :
    expr_cout=0
    expr_mean_init= mean_plne(params.period_init,p_agrege)
    expr_mean_final= mean_plne(params.period_finale,p_agrege)

    #def de la puissance de ref = min(expr_mean_init, expr_mean_final)
    p_ref = model.add_variables(name="p_ref", lower=params.pmin, upper=params.pmax)
    model.add_constraints(p_ref - expr_mean_init <= 0)
    model.add_constraints(p_ref - expr_mean_final <= 0)

    #ajout valo effacement a la f_obj
    rectangle=0
    for t in range(params.period_eff[0], params.period_eff[1], params.largeur_rectangles):
        #on definit le max sur le rectangle
        p_max = model.add_variables(name="p_max_{}".format(t), lower=params.pmin, upper=params.pmax)
        for k in range(params.largeur_rectangles):
            model.add_constraints(p_max - p_agrege[t+k] >= 0)

        # on est en minimisation, on ajoute l'opposé du gain à la f obj
        # pour chaque t de la period d'effacement
        # gain = prix_spot*(p_ref - p_max)
        expr_obj+= - params.spot_prices[rectangle]*(p_ref - p_max)
        rectangle+=1

    return expr_obj

#expression de la moyenne de la variable p sur la period (t_debut, t_fin)
def mean_plne(period, p) :
    expr_mean = 0
    duration_pdt = period[1] - period[0]
    for t in range(period[0], period[1]):
        expr_mean += p[t]
    expr_mean /= duration_pdt
    return expr_mean

 #Calcul de la fonction de pénalisation f_0 (distance L2 entre le profile de l'agregateur et le0 profile agrégé des agents)
def f_0(params,p_aggregator, p_aggreg):

    idx_concerned = [k for k in range(params.period_init[0], params.period_finale[1])]
    return params.rho*np.linalg.norm(p_aggregator[idx_concerned]-p_aggreg[idx_concerned], ord = 2)**2

#Calcul de la dérivée de la fonction de pénalisation f_0
def derivarite_f0(params, p_aggregator,p_aggreg, agents_list):
    derivative = 2 * params.rho * (
                p_aggregator - sum([p_aggreg[agent_bb.name] for agent_bb in agents_list]))
    idx_concerned = [k for k in range(params.period_init[0], params.period_finale[1])]
    reduced_derivative = []
    for k in range(len(derivative)):
        if k in idx_concerned:
            reduced_derivative.append(derivative[k])
        else:
            reduced_derivative.append(0)
    lambdas = {agent.name: [-1 * lbd for lbd in reduced_derivative] for agent in agents_list}
    lambdas['aggregator'] = reduced_derivative

    return lambdas