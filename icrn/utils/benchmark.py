def benchmark_well_mixed(problem: WellMixedProblem, specs):
    pass


def benchmark_reaction_diffusion(problem: ReactionDiffusionProblem, specs):
    return benchmark_reaction_diffusion(problem)


def benchmark(problem: AbstractProblem, *args):
    if isinstance(problem, WellMixedProblem):
        return benchmark_well_mixed(problem, *args)
    elif isinstance(problem, ReactionDiffusionProblem):
        return benchmark_reaction_diffusion(problem, *args)
    else:
        raise ValueError(f"Invalid problem type: {type(problem)}")
