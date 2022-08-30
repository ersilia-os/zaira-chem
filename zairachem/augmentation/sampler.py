import random
import networkx as nx

from ..tools.fpsim2.searcher import SimilaritySearcher, RandomSearcher
from ..tools.exmol.sampler import StonedSampler

TANIMOTO_CUTOFF = 0.6


class Sampler(object):
    def __init__(self, min_similarity=TANIMOTO_CUTOFF):
        self.stoned_sampler = StonedSampler()
        self.similarity_searcher = SimilaritySearcher()
        self.random_searcher = RandomSearcher()
        self.min_similarity = min_similarity

    def has_available_nodes(self, G, query_smiles):
        for n in G.nodes():
            if n not in query_smiles:
                return True
        return False

    def _sample(self, smiles_list, cutoff, max_n):
        query_smiles = set(smiles_list)
        G = nx.Graph()
        for smi in smiles_list:
            results = self.similarity_searcher.search(smi, cutoff=cutoff)
            for r in results:
                G.add_edge(smi, r[1], weight=r[2])
            results = self.stoned_sampler.sample(
                smi, n=int(max_n / len(smiles_list) * 2)
            )
            for i in range(len(results[0])):
                G.add_edge(smi, results[0][i], weight=results[1][i])
        node_clashes = set()
        for e in G.edges(data=True):
            if e[2]["weight"] == 1:
                node_clashes.update([e[1]])
        for n in list(node_clashes):
            if n in query_smiles:
                continue
            else:
                G.remove_node(n)
        sampled = set()
        while self.has_available_nodes(G, query_smiles):
            for smi in smiles_list:
                edges = G.edges(smi, data=True)
                edges = dict(
                    (e[1], e[2]["weight"]) for e in edges if e[1] not in query_smiles
                )
                if not edges:
                    continue
                edge = sorted(edges.items(), key=lambda x: -x[1])[0]
                G.remove_node(edge[0])
                sampled.update([edge[0]])
                if len(sampled) >= max_n:
                    return sampled
            random.shuffle(smiles_list)
        return sampled

    def sample(self, smiles_list, n):
        ref = self._sample(smiles_list, self.min_similarity, max_n=n)
        rnd = self.random_searcher.search(n - len(ref))
        sampled = list(ref.union(rnd))
        random.shuffle(sampled)
        return sampled
