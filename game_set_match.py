# %%
"""
Symbolic Tennis Game Probability Solver

This script uses SymPy to symbolically solve for the exact probability
that Player 1 wins a tennis game in terms of the point win probability p.

It constructs the full transition matrix and uses the fundamental
matrix method to derive the closed-form expression.
"""

from tennis_markov.helpers import create_symbolic_game_tennis_matrix, create_symbolic_set_tennis_matrix, create_symbolic_match_tennis_matrix, solve_win_probability, simplify_expression, plot_probabilities, write_latex
import sympy as sp

# Set up SymPy for better display
sp.init_printing(use_unicode=True)
print("Symbolic tennis game probability computation")
print("=" * 60)
level = "match"
# Create the symbolic transition matrix
if level == "game":
    P, state_names, p = create_symbolic_game_tennis_matrix()
elif level == "set":
    P, state_names, p_g = create_symbolic_set_tennis_matrix()
elif level == "match":
    P, state_names, p_s = create_symbolic_match_tennis_matrix()

# Solve the full system
print("=" * 60)
num_absorbing_states = 2
N, B, win_prob = solve_win_probability(P, num_absorbing_states, level)

# Simplify the expression
simplified_prob = simplify_expression(win_prob, level, max_steps=3)

# LaTeX output
latex_expr = write_latex(simplified_prob, level)

# Plot solution
plot_probabilities(simplified_prob, level)

print("\nNotes:")
print("=" * 60)
print("The exact symbolic expression shows how p propagates through all game states")
print("The full game probability is a polynomial of degree 7")

# %%
