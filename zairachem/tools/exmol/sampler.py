from exmol import run_stoned


class StonedSampler(object):
    def __init__(self, max_mutations=2, min_mutations=1):
        self.max_mutations = max_mutations
        self.min_mutations = min_mutations

    def sample(self, smiles, n):
        return run_stoned(
            smiles,
            num_samples=n,
            max_mutations=self.max_mutations,
            min_mutations=self.min_mutations,
        )
