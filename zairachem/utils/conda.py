import os
import tempfile
import subprocess

BASE = "base"


def run_command(cmd):
    if type(cmd) == str:
        subprocess.Popen(cmd, shell=True, env=os.environ).wait()
    else:
        subprocess.check_call(cmd, env=os.environ)


class BaseConda(object):
    def __init__(self):
        pass

    @staticmethod
    def default_env():
        return os.environ["CONDA_DEFAULT_ENV"]

    def is_base(self):
        default_env = self.default_env()
        if default_env == BASE:
            return True
        else:
            return False

    @staticmethod
    def conda_prefix(is_base):
        if is_base:
            return "CONDA_PREFIX"
        else:
            return "CONDA_PREFIX_1"


class CondaUtils(BaseConda):
    def __init__(self):
        BaseConda.__init__(self)

    def activate_base(self):
        if self.is_base():
            return ""
        snippet = """
        source ${0}/etc/profile.d/conda.sh
        conda activate {1}
        """.format(
            self.conda_prefix(False), BASE
        )
        return snippet


class SimpleConda(CondaUtils):
    def __init__(self):
        CondaUtils.__init__(self)

    def _env_list(self):
        tmp_folder = tempfile.mkdtemp(prefix="ersilia-")
        tmp_file = os.path.join(tmp_folder, "env_list.tsv")
        tmp_script = os.path.join(tmp_folder, "script.sh")
        bash_script = """
        source ${0}/etc/profile.d/conda.sh
        conda env list > {1}
        """.format(
            self.conda_prefix(self.is_base()), tmp_file
        )
        with open(tmp_script, "w") as f:
            f.write(bash_script)
        run_command("bash {0}".format(tmp_script))
        with open(tmp_file, "r") as f:
            envs = []
            for l in f:
                envs += [l.rstrip()]
        return envs

    def active_env(self):
        envs = self._env_list()
        for l in envs:
            if "*" in l:
                return l.split()[0]
        return None

    def exists(self, environment):
        envs = self._env_list()
        n = len(environment)
        for l in envs:
            if l[:n] == environment:
                return True
        return False

    def run_commandlines(self, environment, commandlines):
        if not self.exists(environment):
            raise Exception("{0} environment does not exist".format(environment))
        tmp_folder = tempfile.mkdtemp(prefix="ersilia-")
        tmp_script = os.path.join(tmp_folder, "script.sh")
        bash_script = self.activate_base()
        bash_script += """
        source ${0}/etc/profile.d/conda.sh
        conda activate {1}
        conda env list
        {2}
        """.format(
            self.conda_prefix(True), environment, commandlines
        )
        with open(tmp_script, "w") as f:
            f.write(bash_script)
        cmd = "bash {0}".format(tmp_script)
        run_command(cmd)
