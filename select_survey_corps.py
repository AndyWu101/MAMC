import numpy as np
import torch

from model import Actor


def select_survey_corps(actors: list[Actor], survey_corps_size: int) -> list[Actor]:

    fronts = fast_nondominated_sort(actors, survey_corps_size)

    survey_corps_indices = []
    for front in fronts:
        if (len(survey_corps_indices) + len(front)) > survey_corps_size:

            crowding_distances = calculate_crowding_distance(front, actors)

            # 依照擁擠度由大到小排序
            sorted_front = [front_idx for _ , front_idx in sorted(zip(crowding_distances, front), reverse = True)]
            survey_corps_indices.extend(sorted_front[:survey_corps_size - len(survey_corps_indices)])
            break

        survey_corps_indices.extend(front)

    survey_corps = []
    for index in survey_corps_indices:
        survey_corps.append(actors[index])

    return survey_corps


def dominates(actor1: Actor, actor2: Actor) -> bool:
    # 最大化 skill 與 creativity
    return (actor1.skill >= actor2.skill and actor1.creativity >= actor2.creativity) and (actor1.skill > actor2.skill or actor1.creativity > actor2.creativity)


def fast_nondominated_sort(population: list[Actor], size: int) -> list[list[int]]:

    fronts: list[list[int]] = [[]]
    S = [[] for _ in range(len(population))]
    N = np.zeros(len(population))

    for p in range(len(population)):
        for q in range(len(population)):

            if (dominates(population[p], population[q])):
                S[p].append(q)
                N[q] += 1

    for p in range(len(population)):
        if (N[p] == 0):
            fronts[0].append(p)

    current_size = len(fronts[0])
    while True:

        next_front = []

        for p in fronts[-1]:
            for q in S[p]:

                N[q] -= 1

                if (N[q] == 0):
                    next_front.append(q)

        if len(next_front) != 0:
            fronts.append(next_front)
        else:
            break

        current_size += len(next_front)
        if current_size >= size:
            break

    return fronts


def calculate_crowding_distance(front: list[int], population: list[Actor]) -> np.ndarray:

    with torch.no_grad():

        crowding_distances = np.zeros(len(front), dtype = np.float32)

        ###### 基於skill ######
        sorted_indices = list(range(len(front)))
        sorted_indices.sort(key = lambda i: population[front[i]].skill)
        crowding_distances[sorted_indices[0]] = crowding_distances[sorted_indices[-1]] = np.inf
        f_min = population[front[sorted_indices[0]]].skill
        f_max = population[front[sorted_indices[-1]]].skill

        if ((f_max - f_min) > 1e-18): # 避免除以0的錯誤
            for i in range(1, len(front) - 1):
                crowding_distances[sorted_indices[i]] += (population[front[sorted_indices[i + 1]]].skill - population[front[sorted_indices[i - 1]]].skill) / (f_max - f_min)

        ###### 基於creativity ######
        sorted_indices = list(range(len(front)))
        sorted_indices.sort(key = lambda i: population[front[i]].creativity)
        crowding_distances[sorted_indices[0]] = crowding_distances[sorted_indices[-1]] = np.inf
        f_min = population[front[sorted_indices[0]]].creativity
        f_max = population[front[sorted_indices[-1]]].creativity

        if ((f_max - f_min) > 1e-18): # 避免除以0的錯誤
            for i in range(1, len(front) - 1):
                crowding_distances[sorted_indices[i]] += (population[front[sorted_indices[i + 1]]].creativity - population[front[sorted_indices[i - 1]]].creativity) / (f_max - f_min)

    return crowding_distances


