import os
import pandas as pd

from curtailement import add_expr_fonction_objectif

os.environ['XPRESS'] = '/opt/shared/agregi/env_python/bin/xpauth.xpr'

from linopy import Model


def recombine(params, agent_list, suffix, agent_profiles_dict, agent_costs_dict):
    if agent_list == []: return
    # Pour chaque agent dans agent_list: best_iteration_per_agent[agent.name]=numero de l'iteration retenue
    best_iteration_per_agent = {}

    K = len(agent_profiles_dict)
    n = len(agent_profiles_dict[0])
    T = len(agent_profiles_dict[0][agent_list[0].name])
    print("Recombine profiles of {} agents on {} time steps from {}  iterations ".format(n, T, K))

    range_its = range(K)
    m = Model()

    # variable associee a chaque couple (iter, agent)
    # et ajout a la fonction objectif
    x = {}
    expr_part_indiv_obj = 0
    for it in range_its:
        x[it] = {}
        for agent in agent_list:
            x[it][agent.name] = m.add_variables(name="x_{}_{}".format(it, agent), binary=True)
            expr_part_indiv_obj += agent_costs_dict[it][agent.name] * x[it][agent.name]

    # Contraintes

    ##Convexité: choix d'exactement 1 itération par agent
    for agent in agent_list:
        m.add_constraints(sum([x[it][agent.name] for it in range_its]) == 1)

    ##Profil agrege: p[t]=profil agrege au pas t
    p = {}
    for t in range(T):
        p[t] = m.add_variables(name="p_{}".format(t))
        m.add_constraints(p[t] == sum(
            [agent_profiles_dict[it][agent.name][t] * x[it][agent.name] for it in range_its for agent in agent_list]))

    # Fonction objectif

    expr_obj = expr_part_indiv_obj

    ## Ajout du terme agrege a la f obj (remuneration marche)
    market_obj = add_expr_fonction_objectif(params, m, expr_obj, p)
    expr_obj += market_obj

    obj = m.add_objective(expr_obj)  # minimisation par défaut?

    m.solve(solution_fn=params.output_dir + "\_sol_log_xpress.txt", log_fn=params.output_dir + "\_log_xpress.txt")

    # get solution
    for agent in agent_list:
        for it in range_its:
            if x[it][agent.name].solution == 1:
                best_iteration_per_agent[agent.name] = it

    write_solution_to_csv(params, agent_list, m.objective_value, best_iteration_per_agent, agent_costs_dict, suffix)
    write_load_csv(params, agent_list, best_iteration_per_agent, agent_costs_dict, agent_profiles_dict, suffix)

    return best_iteration_per_agent


def write_load_csv(params, agent_list, best_iteration_per_agent, costs_dict_per_it, agent_profiles_dict, suffix: str):
    filename = params.output_dir + "\\load_recombin_" + params.date_sans_tirets + suffix + ".csv"
    f = open(filename, "x")
    f.write("time;agent;total_load")

    date_of_time_step = pd.date_range(start=params.date, periods=params.T, freq="30T")

    for t in range(params.T):
        total_load = 0
        f.write("\n")
        f.write(str(date_of_time_step[t]))
        for agent in agent_list:
            it = best_iteration_per_agent[agent.name]
            load = agent_profiles_dict[it][agent.name][t]
            f.write(";")
            f.write(str(round(load, 3)))


def write_solution_to_csv(params, agent_list, obj_value, best_iteration_per_agent, costs_dict_per_it, suffix: str):
    filename = params.output_dir + "\\costs_recombin_" + params.date_sans_tirets + suffix + ".csv"
    f = open(filename, "x")
    f.write("agent;iter;cost")
    sum_cost = 0
    for agent in agent_list:
        it = best_iteration_per_agent[agent.name]
        cost_it = costs_dict_per_it[it][agent.name]
        sum_cost += cost_it
        f.write("\n")
        f.write(agent.name)
        f.write(";")
        f.write(str(it))
        f.write(";")
        f.write(str(cost_it))

    # market
    f.write("\n")
    f.write("market;;")
    f.write(str(obj_value - sum_cost))



