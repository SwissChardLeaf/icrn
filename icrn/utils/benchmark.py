def benchmark_well_mixed(problem, specs):
    pass


def benchmark_reaction_diffusion(problem, specs):
    pass


def benchmark(problem, *args):
    raise NotImplementedError(
        f"benchmark is not implemented for {type(problem)!r}"
    )
