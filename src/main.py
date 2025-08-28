# %%
"""
Symbolic Tennis Match Probability Solver.

This script uses SymPy to symbolically solve for the exact probability
that Player 1 wins a tennis game in terms of the point win probability p.

It constructs the full transition matrix and uses the fundamental
matrix method to derive the closed-form expression.
"""

__author__ = "Alan Akil"
__date__ = "2025-08-27"

import sympy as sp

from tennis_markov.helpers import (
    create_symbolic_game_tennis_matrix,
    create_symbolic_match_tennis_matrix,
    create_symbolic_set_tennis_matrix,
    plot_probabilities,
    plot_probability_match_given_point,
    plot_probability_match_split_panel,
    save_publication_plots,
    simplify_expression,
    solve_win_probability,
    write_latex,
)

# Set up SymPy for better display
sp.init_printing(use_unicode=True)
print("Symbolic tennis game probability computation")
print("=" * 60)
level = "match"
# Create the symbolic transition matrix
P, state_names, p = create_symbolic_game_tennis_matrix()
P_g, state_names, p_g = create_symbolic_set_tennis_matrix()
P_s, state_names, p_s = create_symbolic_match_tennis_matrix(best_of=5)

# Solve the full system
print("=" * 60)
num_absorbing_states = 2
levels = ["game", "set", "match"]
transition_matrices = [P, P_g, P_s]
win_probs = []
for level, transition_matrix in zip(levels, transition_matrices):
    _, _, win_prob = solve_win_probability(
        transition_matrix, num_absorbing_states, level
    )
    simplified_prob = simplify_expression(win_prob, level, max_steps=3)
    win_probs.append(simplified_prob)
    plot_probabilities(simplified_prob, level)
    latex_expr = write_latex(simplified_prob, level)


# %%
# Get a final expression for the probability of winning the match as a function of the probability of winning a point

# First substitute p_g into p_s expression
p_s_in_p = win_probs[1].subs(p_g, win_probs[0])
print(f"p_s in terms of p: {p_s_in_p}")
# Then substitute the p_s expression into final expression
result = win_probs[2].subs(p_s, p_s_in_p)
print(f"Final result: {result}")

plot_probability_match_given_point(win_probs[0], p_s_in_p, result)
plot_probability_match_split_panel(win_probs[0], p_s_in_p, result)
save_publication_plots(win_probs[0], p_s_in_p, result)

# %%
