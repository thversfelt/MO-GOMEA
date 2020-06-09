from individual import Individual

a = Individual([2, 5], [4, 3])
b = Individual([2, 1], [3, 3])
assert(a.dominates(b))

