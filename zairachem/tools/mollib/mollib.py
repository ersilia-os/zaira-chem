import os
import shutil
import configparser

from ..molmap.utils.conda import SimpleConda

root = os.path.dirname(os.path.abspath(__file__))

MOLLIB_CONDA_ENVIRONMENT = "mollib"
DATA_TAG = "my_molecules"


class MollibSampler:
    def __init__(self):
        self.cwd = os.getcwd()
        self.exec_folder = os.path.join(root, "virtual_libraries", "experiments")
        self.data_tag = DATA_TAG
        self.data_file_relative = os.path.join("..", "data", self.data_tag + ".txt")
        self.data_file = os.path.join(self.exec_folder, self.data_file_relative)

    def _heuristic_parameters(self, seed_smiles, n_molecules):
        n_seed = len(seed_smiles)
        epochs = 10
        period = 2
        parameters = configparser.ConfigParser()
        parameters.read(os.path.join(self.exec_folder, "parameters_original.ini"))
        parameters["AUGMENTATION"]["fine_tuning"] = str(min(10, int(10000 / n_seed)))
        parameters["MODEL"]["epochs"] = str(epochs)
        parameters["MODEL"]["period"] = str(period)
        parameters["EXPERIMENTS"]["n_sample"] = str(int(n_molecules / period))
        with open(os.path.join(self.exec_folder, "parameters.ini"), "w") as f:
            parameters.write(f)

    def _sample(self, smiles_list):
        with open(self.data_file, "w") as f:
            for s in smiles_list:
                f.write(s + os.linesep)
        cmd = "cd {0}; bash run_morty.sh {1}; cd {2}".format(
            self.exec_folder, self.data_file_relative, self.cwd
        )
        SimpleConda().run_commandlines(MOLLIB_CONDA_ENVIRONMENT, cmd)

    def _read_molecules(self):
        output_folder = os.path.join(
            self.exec_folder, "results", self.data_tag, "novo_molecules"
        )
        smiles = set()
        for fn in os.listdir(output_folder):
            if "molecules_" in fn and ".txt" in fn:
                with open(os.path.join(output_folder, fn), "r") as f:
                    for l in f:
                        smiles.update([l.rstrip(os.linesep)])
        return list(smiles)

    def _clean(self):
        shutil.rmtree(os.path.join(self.exec_folder, "results", self.data_tag))
        shutil.rmtree(os.path.join(self.exec_folder, "results", "data", self.data_tag))
        os.remove(self.data_file)

    def sample(self, seed_smiles, n_molecules):
        self._heuristic_parameters(seed_smiles, n_molecules)
        self._sample(seed_smiles)
        molecules = self._read_molecules()
        self._clean()
        return molecules
