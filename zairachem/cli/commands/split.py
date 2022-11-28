import click
import pandas as pd
import os
import tempfile

from . import zairachem_cli
from ..echo import echo

from ...setup.training import TrainSetup
from ... import logger


MIN_PCT_FOLD = 5  # Each fold to be used for testing should have at least this percent of cases
MIN_CASES_FOLD = 3  # Each fold to be used for testing should have at least this number of cases
MIN_POSITIVE_CASES_FOLD = 1  # Each fold to be used for testing should have at least this number of positives


def check_dataset_minimum_size(df, fold_id):
    """Check if a fold of a dataset:
        - Has too few cases (less than MIN_CASES_FOLD)
        - Has too few percent of total cases (less than MIN_PCT_FOLD)
        - Has too few positive cases (less than MIN_POSITIVE_CASES_FOLD)
    If any of this happens, issue a warning.
    """
    
    # Warning if the number of cases in given fold is too small
    fold_size = len(df[df.fold == fold_id])
    if fold_size < MIN_CASES_FOLD:
        logger.warning(f'The resulting number of test cases is too small:'
                f' {fold_size} cases. The minimum is {MIN_CASES_FOLD}.')
    elif (fold_size / len(df))*100 < MIN_PCT_FOLD:
        logger.warning(f'The resulting percent of test cases is too small:'
                f' {fold_size} cases out of {len(df)}. The minimum is {MIN_PCT_FOLD}%.')
        
    # Warning if too few positives
    fold_num_positives = sum(df[df.fold == fold_id].activity)
    if fold_num_positives < MIN_POSITIVE_CASES_FOLD:
        logger.warning(f'The resulting number of test cases with activity=1 (positive)'
                f'is too small: {fold_num_positives} cases. The minimum is {MIN_POSITIVE_CASES_FOLD}.')


def create_dataset_csv(df_subset, csv_dir, csv_name):
    if len(df_subset) > 0:
        df_subset[['smiles', 'activity']].to_csv(
                    os.path.join(csv_dir, csv_name),
                    index=False)


def create_datasets_for_fold(df, test_fold_id, all_fold_ids, output_dir):
    """Given a dataset and a fold id, create three csv files with:
    - test.csv: For test, contains fold indicated by test_fold_id
    - train.csv: For training, contains all folds except test_fold_id
    - input_discarded.csv: Contains all cases that do not have an assigned fold
    
    df must contain variables 'fold', 'smiles' and 'activity'.
    """
    
    # Check if test dataset is too small
    check_dataset_minimum_size(df, test_fold_id)

    train_fold_id_set = all_fold_ids - {test_fold_id}  # The folds not in test go in train set

    # Create train dataset
    create_dataset_csv(df[df.fold.isin(train_fold_id_set)], output_dir, 'train.csv')
    # Create test dataset
    create_dataset_csv(df[df.fold==test_fold_id], output_dir, 'test.csv')
    # Create input_discarded dataset for the cases with no fold
    create_dataset_csv(df[df.fold.isnull()], output_dir, 'input_discarded.csv')


def split_cmd():
    @zairachem_cli.command(help="Split input data set for cross-validation")
    @click.option("--input_file", "-i", type=click.STRING, required=True)
    @click.option("--folds_dir", "-f", default=None, type=click.STRING,
            help="Directorate where the train and test data sets will be created."
                 " By default, same directory as input file.")
    @click.option("--split_criterion", "-s", default='default',
            type=click.Choice(['default', 'random', 'scaffold', 'similarity']))
    @click.option("--kfold", "-k", is_flag=True, default=False,
            help="Create five train-test splits for k-fold cross-validation,"
                 " in subdirectories fold_0, fold_1, etc.")
    def split(input_file, folds_dir, split_criterion, kfold):

        # folds_dir: if not specified, use the same dir as input_file
        if folds_dir is None:
            folds_dir = os.path.dirname(input_file)
        logger.debug(f"Split datasets will be stored at {folds_dir}")

        # Depending on split_criterion, select the variable to use for fold assignments
        if split_criterion == 'random':
            fold_var = 'fld_rnd'
        elif split_criterion == 'scaffold':
            fold_var = 'fld_scf'
        elif split_criterion == 'similarity':
            fold_var = 'fld_lsh'
        elif split_criterion == 'default':
            fold_var = 'fld_aux'
        else:
            logger.error(f'Invalid value "{split_criterion}" for parameter --split_criterion.')
            sys.exit(1)
            
        # Create directory to store the output of TrainSetup
        tmp_output_dir = tempfile.mkdtemp()

        # The TrainSetup method will generate the fold assignments
        s = TrainSetup(
            input_file=input_file,
            reference_file=input_file,
            output_dir=tmp_output_dir,
            parameters=None,
            time_budget=60,
            task=None,
            direction=None,
            threshold=None,
            is_lazy=False,
            augment=False,
        )
        s.setup()
        # FOR TESTING USE /home/marcos/split_dev/model instead of calling s.setup() (which takes long)
        # tmp_output_dir = '/home/marcos/split_dev/model'

        # Read files with input data and fold assignments
        df_input = pd.read_csv(input_file)
        df_folds = pd.read_csv(os.path.join(tmp_output_dir, 'data', 'data.csv'),
                            usecols=['fld_rnd', 'fld_scf', 'fld_lsh', 'fld_aux'])
        df_mapping = pd.read_csv(os.path.join(tmp_output_dir, 'data', 'mapping.csv'),
                                usecols=['uniq_idx'])

        # Merge the input file with the 4 columns containing the fold assignments
        df = df_input.merge(
                df_mapping,
                how='left',
                left_index=True,
                right_index=True)
        df = df.merge(
                df_folds,
                how='left',
                left_on='uniq_idx',
                right_index=True)

        # Create variable "fold" according to parameter --split_criterion
        df['fold'] = df[fold_var]

        fold_counts = df.fold.value_counts(sort=True)  # Number of cases in each fold
        num_folds = len(fold_counts)  # Number of folds
        all_fold_ids = set(fold_counts.index.astype(int))  # Set of all fold ids

        if not kfold:
            # Default: simple split into train and test datasets
            # The second largest fold, as defined by the "fold" variable, will define the test dataset.
            # The other folds will be in the train split.
            test_fold_id = fold_counts.index[1]  # Second largest fold
            create_datasets_for_fold(df, test_fold_id, all_fold_ids, folds_dir)
        else:
            # Option --kfold, for k-fold cross-validation
            # Create a directory for each fold, each with a train and test dataset
            for current_fold_id in all_fold_ids:
                # Create a directory for the fold, if it does not exist
                current_fold_dir = os.path.join(folds_dir, f'fold_{current_fold_id}')
                if not os.path.isdir(current_fold_dir):
                    os.mkdir(current_fold_dir)

                create_datasets_for_fold(df, current_fold_id, all_fold_ids, current_fold_dir)

        echo("Done", fg="green")
