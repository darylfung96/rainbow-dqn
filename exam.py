

def solution(total_money, total_damage, costs, damages):
    # write your code in Python 3.6

    total_costs = [costs[0]]
    indexes = [0]
    damage_array = [damages[0]]

    while indexes[0] < len(costs):
        damage_array.append(damages[indexes[-1]])
        total_costs.append(costs[indexes[-1]])
        indexes.append(indexes[-1]+1)

        if sum(total_costs) <= total_money:
            if sum(damage_array) >= total_damage:
                return True

        while indexes[-1] + 1 >= len(costs):
            indexes = indexes[:-1]
            if len(indexes) == 0:
                return False
        if len(indexes) > 1:
            indexes[-1] = indexes[-1] + 1
        else:
            indexes[0] += 1
        damage_array = damage_array[:len(indexes)]
        total_costs = total_costs[:len(indexes)]




    return False



# def solution(S, T):
#     if len(S) < len(T):
#         return 0
#     if len(S) is len(T) and S != T:
#         return 0
#
#     result = []
#
#     for i in range(len(S)):
#
#         if S[i] in T:
#             result.append(S[i])
#
#     if T in ''.join(result):
#         return 1
#
#     return 0



print(solution(10, 6, [4,5,2,2,3], [1,2,3,1,5]) )