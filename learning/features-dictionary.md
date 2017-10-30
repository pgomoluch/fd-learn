0: The number of the (multi-valued) variables.
1..3: The quartiles of the variable domain size.
4: The size of the goal.
5: The number of unsatisfied subgoals.
6: The number of applicable operators.
7: The number of applicable operators which do not undo one of the goals.
8: The value of the FF heuristic.
9: The value of the CEA heuristic.
10: The number of operators in FF's relaxed plan.
11: The total number of ignored effects in FF's relaxed plan.
12: The average number of ignored effects in FF's relaxed plan.
13: The number of layers in FF's graph.
14-16: The quartiles of occurrences of action schemas in FF's relaxed plan.
17: The number of non-zero elements in "pairwise actions" matrix.
18: The number of symmetric entries p(i,j) = p(j,i) in "pairwise actions" matrix p.

Domain-dependent base index D = 19.
Number of operator schemas S.

D..D+S-1: Schema occurrences (Garrett's "single action features").
D+S-1..D+S-1+S^2: Garret's "pairwise action features".
