"""Helpers functions for tennis markov chains."""

from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.patches import Rectangle
from sympy import factor
from sympy import latex
from sympy import Matrix
from sympy import simplify
from sympy import Symbol
from sympy import symbols


def create_symbolic_game_tennis_matrix() -> Tuple[Matrix, List[str], Symbol]:  # noqa: C901
    """
    Create the symbolic 21x21 tennis transition matrix.

    Returns:
        tuple: (P, state_names, p_symbol)
    """
    print("Creating symbolic transition matrix...")
    # Define the symbolic variable
    p = symbols("p", real=True, positive=True)

    # State names in order
    state_names = [
        "0-0",
        "0-1",
        "0-2",
        "0-3",
        "1-0",
        "1-1",
        "1-2",
        "1-3",
        "2-0",
        "2-1",
        "2-2",
        "2-3",
        "3-0",
        "3-1",
        "3-2",
        "Deuce",
        "Adv-P1",
        "Adv-P2",
        "P1-Wins",
        "P2-Wins",
    ]

    # Initialize transition matrix
    P = Matrix.zeros(len(state_names), len(state_names))

    # State index mapping
    state_to_idx = {state: i for i, state in enumerate(state_names)}

    # Fill in the transition probabilities
    for i, state in enumerate(state_names):
        if state in ["P1-Wins", "P2-Wins"]:
            # Absorbing states
            P[i, i] = 1

        elif state == "Deuce":
            # From Deuce
            P[i, state_to_idx["Adv-P1"]] = p
            P[i, state_to_idx["Adv-P2"]] = 1 - p

        elif state == "Adv-P1":
            # From Advantage Player 1
            P[i, state_to_idx["P1-Wins"]] = p
            P[i, state_to_idx["Deuce"]] = 1 - p

        elif state == "Adv-P2":
            # From Advantage Player 2
            P[i, state_to_idx["P2-Wins"]] = p
            P[i, state_to_idx["Deuce"]] = 1 - p

        else:
            # Regular scoring states
            p1_score, p2_score = map(int, state.split("-"))

            # Player 1 wins point
            if p1_score == 3 and p2_score < 3:
                # Player 1 wins game
                P[i, state_to_idx["P1-Wins"]] = p
            elif p1_score < 3:
                new_state = f"{p1_score + 1}-{p2_score}"
                if new_state in state_to_idx:
                    P[i, state_to_idx[new_state]] = p

            # Player 2 wins point
            if p2_score == 3 and p1_score < 3:
                # Player 2 wins game
                P[i, state_to_idx["P2-Wins"]] = 1 - p
            elif p2_score < 3:
                new_state = f"{p1_score}-{p2_score + 1}"
                if new_state in state_to_idx:
                    P[i, state_to_idx[new_state]] = 1 - p

            # Special deuce cases
            if p1_score == 3 and p2_score == 2:
                # From 3-2, if P2 wins, go to Deuce
                P[i, state_to_idx["Deuce"]] = 1 - p
            elif p1_score == 2 and p2_score == 3:
                # From 2-3, if P1 wins, go to Deuce
                P[i, state_to_idx["Deuce"]] = p

    print(f"Point transition matrix:\n{P}")
    print(f"Shape of transition matrix P: {P.shape[0]}x{P.shape[1]}")

    return P, state_names, p


def create_symbolic_set_tennis_matrix() -> Tuple[Matrix, List[str], Symbol]:  # noqa: C901
    """
    Derive the exact symbolic expression for Player 1 winning a tennis set.

    Returns:
        SymPy expression for P(Player 1 wins set) in terms of p_g (game win probability)
    """
    # Define symbolic variable
    p_g = symbols("p_g", real=True, positive=True)  # Game win probability for Player 1

    # Define all valid set states
    # States: (i,j) where i,j are games won, plus Tiebreak and terminal states
    state_names = []
    state_to_idx = {}
    idx = 0

    # Regular set states
    for i in range(7):
        for j in range(7):
            # Skip impossible states (would have ended earlier)
            if (i >= 6 or j >= 6) and abs(i - j) >= 2:
                continue
            if i == 6 and j == 6:
                continue  # Goes to tiebreak
            state_name = f"{i}-{j}"
            state_names.append(state_name)
            state_to_idx[state_name] = idx
            idx += 1

    # Add special states
    state_names.extend(["Tiebreak", "P1-Set", "P2-Set"])
    state_to_idx["Tiebreak"] = idx
    state_to_idx["P1-Set"] = idx + 1
    state_to_idx["P2-Set"] = idx + 2

    n_states = len(state_names)
    print(f"Total states: {n_states}")
    print(f"States: {state_names}")

    # Create transition matrix
    P = Matrix.zeros(n_states, n_states)

    for i, state in enumerate(state_names):
        if state in ["P1-Set", "P2-Set"]:
            # Absorbing states
            P[i, i] = 1

        elif state == "Tiebreak":
            # Simplified tiebreak: use game win probability
            P[i, state_to_idx["P1-Set"]] = p_g
            P[i, state_to_idx["P2-Set"]] = 1 - p_g

        else:
            # Regular set states (i-j format)
            p1_games, p2_games = map(int, state.split("-"))

            # Player 1 wins game
            new_p1_games = p1_games + 1
            if new_p1_games >= 6 and new_p1_games - p2_games >= 2:
                # Player 1 wins set
                P[i, state_to_idx["P1-Set"]] = p_g
            elif new_p1_games == 6 and p2_games == 6:
                # Goes to tiebreak
                P[i, state_to_idx["Tiebreak"]] = p_g
            else:
                # Continue playing
                new_state = f"{new_p1_games}-{p2_games}"
                if new_state in state_to_idx:
                    P[i, state_to_idx[new_state]] = p_g

            # Player 2 wins game
            new_p2_games = p2_games + 1
            if new_p2_games >= 6 and new_p2_games - p1_games >= 2:
                # Player 2 wins set
                P[i, state_to_idx["P2-Set"]] = 1 - p_g
            elif new_p2_games == 6 and p1_games == 6:
                # Goes to tiebreak
                P[i, state_to_idx["Tiebreak"]] = 1 - p_g
            else:
                # Continue playing
                new_state = f"{p1_games}-{new_p2_games}"
                if new_state in state_to_idx:
                    P[i, state_to_idx[new_state]] = 1 - p_g
    return P, state_names, p_g


def create_symbolic_match_tennis_matrix(
    best_of: int = 5,
) -> Tuple[Matrix, List[str], Symbol]:
    """
    Derive the exact symbolic expression for Player 1 winning a tennis match.

    Args:
        best_of (int): Match format - 3 for best-of-3, 5 for best-of-5

    Returns:
        SymPy expression for P(Player 1 wins match) in terms of p_s (set win probability)
    """
    # Define symbolic variable
    p_s = symbols("p_s", real=True, positive=True)  # Set win probability for Player 1
    sets_to_win = (best_of + 1) // 2  # Need to win majority of sets

    # Define match states
    state_names = []
    state_to_idx = {}
    idx = 0

    # Regular match states: (i, j) where i, j are sets won
    for i in range(sets_to_win + 1):
        for j in range(sets_to_win + 1):
            if i < sets_to_win and j < sets_to_win:
                state_name = f"{i}-{j}"
                state_names.append(state_name)
                state_to_idx[state_name] = idx
                idx += 1

    # Add terminal states
    state_names.extend(["P1-Match", "P2-Match"])
    state_to_idx["P1-Match"] = idx
    state_to_idx["P2-Match"] = idx + 1

    n_states = len(state_names)
    print(f"Match format: Best of {best_of} (first to {sets_to_win})")
    print(f"Total states: {n_states}")

    # Create transition matrix
    P = Matrix.zeros(n_states, n_states)

    for i, state in enumerate(state_names):
        if state in ["P1-Match", "P2-Match"]:
            # Absorbing states
            P[i, i] = 1
        else:
            # Regular match states
            p1_sets, p2_sets = map(int, state.split("-"))

            # Player 1 wins set
            new_p1_sets = p1_sets + 1
            if new_p1_sets == sets_to_win:
                # Player 1 wins match
                P[i, state_to_idx["P1-Match"]] = p_s
            else:
                new_state = f"{new_p1_sets}-{p2_sets}"
                if new_state in state_to_idx:
                    P[i, state_to_idx[new_state]] = p_s

            # Player 2 wins set
            new_p2_sets = p2_sets + 1
            if new_p2_sets == sets_to_win:
                # Player 2 wins match
                P[i, state_to_idx["P2-Match"]] = 1 - p_s
            else:
                new_state = f"{p1_sets}-{new_p2_sets}"
                if new_state in state_to_idx:
                    P[i, state_to_idx[new_state]] = 1 - p_s
    return P, state_names, p_s


def solve_win_probability(
    P: Matrix,
    num_absorbing_states: int,
    level: str,
) -> Tuple[Matrix, Matrix, Symbol]:
    """
    Solve for the game win probability using symbolic computation.

    Args:
        Q: Transient-to-transient matrix
        R: Transient-to-absorbing matrix
        level: "game", "set", "match"

    Returns:
        tuple: (fundamental_matrix, absorption_probs, game_win_prob)
    """
    print(f"Computing fundamental matrix N = (I - Q)^(-1)... for the {level}")
    num_nonabsorbing_states = P.shape[0] - num_absorbing_states
    Q = P[:num_nonabsorbing_states, :num_nonabsorbing_states]
    R = P[:num_nonabsorbing_states, num_nonabsorbing_states:]

    print(f"Q matrix transient-to-transient transitions:\n{Q}")
    print(f"Shape of Q: {Q.shape[0]}x{Q.shape[1]}\n")
    print(f"R matrix transient-to-absorbing transitions:\n{R}")
    print(f"Shape of matrix R: {R.shape[0]}x{R.shape[1]}\n")

    # Create identity matrix
    Identity = Matrix.eye(num_nonabsorbing_states)

    # Compute I - Q
    I_minus_Q = Identity - Q

    print("Inverting (I - Q) matrix...")

    # Compute fundamental matrix N = (I - Q)^(-1)
    try:
        N = I_minus_Q.inv()
        print("Successfully computed fundamental matrix!")
    except Exception as e:
        print(f"Error computing inverse: {e}")
        return None, None, None

    print("Computing absorption probabilities B = N * R...")

    # Compute absorption probabilities B = N * R
    B = N * R

    print(f"B matrix:\n{B}\n Has shape:\n{B.shape}")
    print(
        "The rows in B denote the starting state, and the columns are the absorbing states (Player 1 wins or Player 2 wins). So the entry [0,0] is the probability that starting the game at 0-0, player 1 will win the game."
    )
    # Game win probability is B[0, 0] (from state 0-0 to P1-Wins)
    win_prob = B[0, 0]

    print(f"Successfully computed {level} win probability!")

    print(f"\nRaw expression for probability of winning {level}:")
    print("Length of raw expression:", len(str(win_prob)))
    print(win_prob)

    return N, B, win_prob


def simplify_expression(expr: Symbol, level: str, max_steps: int = 5) -> Symbol:
    """
    Apply various simplification techniques to clean up the expression.

    Args:
        expr: SymPy expression
        max_steps: Maximum simplification steps

    Returns:
        Simplified expression
    """
    print("=" * 50)
    print("Simplifying expression...")
    current = expr
    for step in range(max_steps):
        print(f"  Step {step + 1}: Applying simplification...")

        # Try different simplification approaches
        simplified = simplify(current)
        factored = factor(simplified)

        # Use the shorter one
        if len(str(factored)) < len(str(simplified)):
            current = factored
        else:
            current = simplified

        print(f"    Current length: {len(str(current))}")

    print("Simplification complete...")
    print("Simplified expression length:", len(str(current)))

    print("\nFinal symbolic expression:")
    print(f"p_{level[0]} := P(Player 1 wins {level}) = {current}")

    return current


def plot_probabilities(symb_expr: Symbol, level: str):
    if level == "game":
        p = sp.Symbol("p", real=True, positive=True)
        f = sp.lambdify(p, symb_expr, "numpy")
    elif level == "set":
        p_g = sp.Symbol("p_g", real=True, positive=True)
        f = sp.lambdify(p_g, symb_expr, "numpy")
    elif level == "match":
        p_s = sp.Symbol("p_s", real=True, positive=True)
        f = sp.lambdify(p_s, symb_expr, "numpy")

    # Create x values and plot
    x_vals = np.linspace(0, 1.0, 1000)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, "b-", linewidth=2, label=symb_expr)
    plt.plot(x_vals, x_vals, "k--")
    plt.grid(True, alpha=0.3)
    if level == "game":
        plt.xlabel("Probability of winning a point, p")
    elif level == "set":
        plt.xlabel("Probability of winning a game, p_g")
    elif level == "match":
        plt.xlabel("Probability of winning a set, p_s")

    plt.ylabel(f"Probability of winning a {level}, p_{level[0]}")
    plt.legend()
    plt.show()


# def plot_probability_match_given_point(game_expr, set_expr, match_expr):
#     p = sp.Symbol('p', real=True, positive=True)

#     f = sp.lambdify(p, game_expr, 'numpy')
#     f_g = sp.lambdify(p, set_expr, 'numpy')
#     f_s = sp.lambdify(p, match_expr, 'numpy')

#     # Create x values and plot
#     x_vals = np.linspace(0, 1.0, 1000)
#     y_vals_game = f(x_vals)
#     y_vals_set = f_g(x_vals)
#     y_vals_match = f_s(x_vals)

#     plt.figure(figsize=(8, 6))
#     plt.plot(x_vals, x_vals, "k--", alpha=0.2)
#     plt.axvline(x=0.6, "k-", alpha=0.2)
#     plt.plot(x_vals, y_vals_game, 'b-', linewidth=2, alpha=0.4, label="P(Player 1 wins game)")
#     plt.plot(x_vals, y_vals_set, 'r-', linewidth=2, alpha=0.4, label="P(Player 1 wins set)")
#     plt.plot(x_vals, y_vals_match, 'g-', linewidth=2, alpha=0.4, label="P(Player 1 wins match)")

#     plt.grid(True, alpha=0.2)
#     plt.xlabel('Probability of winning a point, p')
#     plt.ylabel('Probability')
#     plt.legend()
#     plt.show()


def write_latex(simplified_prob: Symbol, level: str) -> str:
    print("\nExpression written in LaTeX form:")
    print("=" * 60)
    latex_expr = latex(simplified_prob)
    print(f"p_{level[0]} := P(Player 1 wins {level}) = {latex_expr}")
    return latex_expr


def plot_probability_match_given_point(
    game_expr: Symbol, set_expr: Symbol, match_expr: Symbol
):
    """
    Publication-quality plot of tennis win probabilities with annotations.
    """
    # Set publication-quality style
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 1.2,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.framealpha": 0.95,
            "grid.alpha": 0.3,
            "lines.linewidth": 2.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    p = sp.Symbol("p", real=True, positive=True)

    f = sp.lambdify(p, game_expr, "numpy")
    f_g = sp.lambdify(p, set_expr, "numpy")
    f_s = sp.lambdify(p, match_expr, "numpy")

    # Create high-resolution x values
    x_vals = np.linspace(0, 1.0, 2000)
    y_vals_game = f(x_vals)
    y_vals_set = f_g(x_vals)
    y_vals_match = f_s(x_vals)

    # Create figure with golden ratio dimensions
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.18))

    # Define professional color palette
    colors = {
        "identity": "#2C3E50",
        "game": "#3498DB",  # Professional blue
        "set": "#E74C3C",  # Professional red
        "match": "#27AE60",  # Professional green
        "reference": "#95A5A6",  # Gray for reference lines
        "annotation": "#34495E",  # Dark gray for annotations
    }

    # Plot identity line (y=x) with subtle styling
    ax.plot(
        x_vals,
        x_vals,
        color=colors["identity"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.4,
        label="$P_{point} = P_{outcome}$",
    )

    # Plot main curves with professional styling
    ax.plot(
        x_vals,
        y_vals_game,
        color=colors["game"],
        linewidth=3,
        label="Game probability",
        zorder=3,
    )
    ax.plot(
        x_vals,
        y_vals_set,
        color=colors["set"],
        linewidth=3,
        label="Set probability",
        zorder=3,
    )
    ax.plot(
        x_vals,
        y_vals_match,
        color=colors["match"],
        linewidth=3,
        label="Match probability",
        zorder=3,
    )

    # Annotation points with enhanced styling
    annotation_points = [0.5, 0.55, 0.6]
    functions = [f, f_g, f_s]
    curve_colors = [colors["game"], colors["set"], colors["match"]]

    # Add vertical reference lines
    for p_val in annotation_points:
        ax.axvline(
            x=p_val,
            color=colors["reference"],
            linestyle=":",
            linewidth=1,
            alpha=0.6,
            zorder=1,
        )

    # Create annotation table as inset
    table_data = []
    for p_val in annotation_points:
        row_data = [p_val]
        for func in functions:
            row_data.append(func(p_val))
        table_data.append(row_data)

    # Position table in upper left corner
    table_x, table_y = 0.05, 0.95
    table_width, table_height = 0.35, 0.25

    # Create table background
    table_bg = Rectangle(
        (table_x, table_y - table_height),
        table_width,
        table_height,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=colors["annotation"],
        linewidth=1.5,
        alpha=0.95,
    )
    ax.add_patch(table_bg)

    # Add table headers
    headers = ["$p$", "Game", "Set", "Match"]
    header_colors = [colors["annotation"]] + curve_colors

    for i, (header, color) in enumerate(zip(headers, header_colors)):
        ax.text(
            table_x + 0.02 + i * 0.075,
            table_y - 0.03,
            header,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=color,
            ha="center",
        )

    # Add horizontal line under headers
    ax.plot(
        [table_x + 0.01, table_x + table_width - 0.01],
        [table_y - 0.05, table_y - 0.05],
        transform=ax.transAxes,
        color=colors["annotation"],
        linewidth=1,
    )

    # Add table data
    for row_idx, row_data in enumerate(table_data):
        y_pos = table_y - 0.08 - row_idx * 0.05

        # p-value
        ax.text(
            table_x + 0.02,
            y_pos,
            f"{row_data[0]:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            color=colors["annotation"],
        )

        # Probability values with color coding
        for col_idx, (val, color) in enumerate(zip(row_data[1:], curve_colors)):
            ax.text(
                table_x + 0.095 + col_idx * 0.075,
                y_pos,
                f"{val:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                ha="center",
                color=color,
                fontweight="bold",
            )

    # Add marker points on curves
    for p_val in annotation_points:
        for func, color in zip(functions, curve_colors):
            y_val = func(p_val)
            ax.plot(
                p_val,
                y_val,
                "o",
                color=color,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=2,
                zorder=4,
            )

    # Professional grid styling
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Enhanced axis styling
    ax.set_xlabel("Point win probability, $p$", fontsize=14, fontweight="bold")
    ax.set_ylabel("Outcome probability", fontsize=14, fontweight="bold")
    ax.set_title(
        "Tennis Win Probabilities: From Points to Matches",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set professional axis limits and ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Format tick labels
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.2, length=6)
    ax.tick_params(axis="both", which="minor", width=1, length=3)

    # Enhanced legend with custom positioning
    legend = ax.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.3),
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1,
        columnspacing=1,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor(colors["annotation"])

    # Add subtle background color
    ax.set_facecolor("#FAFAFA")

    # Tighten layout
    plt.tight_layout()

    return fig, ax


def plot_probability_match_split_panel(
    game_expr: Symbol, set_expr: Symbol, match_expr: Symbol
):
    """
    Alternative publication-quality plot with separate value panel.
    """
    # Set publication style
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "figure.dpi": 300,
        }
    )

    p = sp.Symbol("p", real=True, positive=True)
    f = sp.lambdify(p, game_expr, "numpy")
    f_g = sp.lambdify(p, set_expr, "numpy")
    f_s = sp.lambdify(p, match_expr, "numpy")

    # Create subplot layout
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[3, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.3
    )

    # Main plot
    ax_main = fig.add_subplot(gs[0, :])

    # High-resolution data
    x_vals = np.linspace(0, 1.0, 2000)
    y_vals_game = f(x_vals)
    y_vals_set = f_g(x_vals)
    y_vals_match = f_s(x_vals)

    # Professional colors
    colors = {"game": "#1f77b4", "set": "#d62728", "match": "#2ca02c"}

    # Plot curves
    ax_main.plot(x_vals, x_vals, "k--", alpha=0.3, linewidth=1.5, label="$y = x$")
    ax_main.plot(x_vals, y_vals_game, color=colors["game"], linewidth=3, label="Game")
    ax_main.plot(x_vals, y_vals_set, color=colors["set"], linewidth=3, label="Set")
    ax_main.plot(
        x_vals, y_vals_match, color=colors["match"], linewidth=3, label="Match"
    )

    # Annotation points
    annotation_points = [0.5, 0.55, 0.6]
    for p_val in annotation_points:
        ax_main.axvline(x=p_val, color="gray", linestyle=":", alpha=0.6)
        for func, color in zip([f, f_g, f_s], colors.values()):
            y_val = func(p_val)
            ax_main.plot(
                p_val,
                y_val,
                "o",
                color=color,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=2,
            )

    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlabel("Point win probability, $p$", fontsize=13, fontweight="bold")
    ax_main.set_ylabel("Outcome probability", fontsize=13, fontweight="bold")
    ax_main.set_title("Tennis Win Probabilities", fontsize=15, fontweight="bold")
    ax_main.legend(frameon=True, fancybox=True)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)

    # Value table
    ax_table = fig.add_subplot(gs[1, :])

    # Prepare data
    table_data = []
    for p_val in annotation_points:
        table_data.append(
            [
                f"{p_val:.2f}",
                f"{f(p_val):.3f}",
                f"{f_g(p_val):.3f}",
                f"{f_s(p_val):.3f}",
            ]
        )

    # Create table
    table = ax_table.table(
        cellText=table_data,
        colLabels=["$p$", "Game", "Set", "Match"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Color headers
    for i, color in enumerate(["black"] + list(colors.values())):
        table[(0, i)].set_facecolor("#E8E8E8")
        table[(0, i)].set_text_props(weight="bold", color=color)

    # Color data cells
    for i in range(len(table_data)):
        for j, color in enumerate([colors["game"], colors["set"], colors["match"]]):
            table[(i + 1, j + 1)].set_text_props(color=color, weight="bold")

    ax_table.axis("off")

    plt.suptitle("Tennis Probability Analysis", fontsize=16, fontweight="bold", y=0.95)

    return fig


# Example usage function
def save_publication_plots(
    game_expr: Symbol,
    set_expr: Symbol,
    match_expr: Symbol,
    filename_base: str = "../data/tennis_probabilities",
):
    """
    Generate and save publication-quality plots in multiple formats.
    """
    # Generate main plot
    fig1, ax1 = plot_probability_match_given_point(game_expr, set_expr, match_expr)

    # Save in multiple formats for publication
    formats = ["png", "pdf"]
    for fmt in formats:
        fig1.savefig(
            f"{filename_base}_main.{fmt}",
            format=fmt,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

    plt.show()

    # Generate split panel version
    fig2 = plot_probability_match_split_panel(game_expr, set_expr, match_expr)

    for fmt in formats:
        fig2.savefig(
            f"{filename_base}_split.{fmt}",
            format=fmt,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

    plt.show()

    print(f"Plots saved in formats: {', '.join(formats)}")
    print(f"Files: {filename_base}_main.* and {filename_base}_split.*")
