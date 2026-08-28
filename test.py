def generate_compositions(target, path=[]):
    if target == 0:
        yield path
        return
    for i in range(1, target + 1):
        yield from generate_compositions(target - i, path + [i])

# Example usage:
N = 5
for composition in generate_compositions(N):
    print(composition)