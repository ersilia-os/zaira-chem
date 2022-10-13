"""
This script takes as input a folder containing ZairaChem results and deletes all the model checkpoints from it.
"""

import sys
import os
import shutil


def remove(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def ad_hoc_standard_flusher(dir_path):

    if not os.path.exists(dir_path):
        return

    base_folder = os.path.basename(dir_path)
    if not base_folder.startswith("fold"):
        return

    model_path = os.path.join(dir_path, "model")
    if not os.path.exists(model_path):
        return

    print(dir_path)
    test_path = os.path.join(dir_path, "test")
    if not os.path.exists(test_path):
        return

    if not os.path.exists(os.path.join(model_path, "output.csv")):
        return

    if not os.path.exists(os.path.join(test_path, "output.csv")):
        return

    remove(os.path.join(model_path, "descriptors"))
    remove(os.path.join(model_path, "estimators"))
    remove(os.path.join(model_path, "pool"))
    remove(os.path.join(model_path, "lite"))
    remove(os.path.join(test_path, "descriptors"))
    remove(os.path.join(test_path, "estimators"))
    remove(os.path.join(test_path, "pool"))
    remove(os.path.join(test_path, "lite"))


if __name__ == "__main__":
    root_path = os.path.abspath(sys.argv[1])
    print(root_path)
    for currentpath, folders, files in os.walk(root_path):
        ad_hoc_standard_flusher(currentpath)
