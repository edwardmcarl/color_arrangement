import cProfile

from plot_colors import run as runColor

if __name__ == "__main__":
    cProfile.run("runColor()")
