import itertools

def integer_compositions(n, k):
    """Generates all compositions of n into k positive integers (i >= 1)."""
    if k == 1:
        yield (n,)
        return
    for cuts in itertools.combinations(range(1, n), k - 1):
        yield tuple(b - a for a, b in zip((0,) + cuts, cuts + (n,)))

def get_tuple_combinations(I_total, J_max, min_j=1):
    results = []
    available_j = range(min_j, J_max)  # j < J_max
    
    # Combination length k can range from 1 up to min(I_total, len(available_j))
    for k in range(1, min(I_total, len(available_j)) + 1):
        for j_combo in itertools.combinations(available_j, k):
            for i_comp in integer_compositions(I_total, k):
                results.append(list(zip(i_comp, j_combo)))
                
    return results

# Example Usage
combinations = get_tuple_combinations(I_total=4, J_max=5)
for c in combinations:
    print(c)