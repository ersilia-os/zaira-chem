from .schema import InputSchema


class SingleFile(InputSchema):
    def __init__(self, input_file):
        InputSchema.__init__(self, input_file)

    def process(self):
        identifier = self.find_identifier_column()
        smiles_column = self.find_smiles_column()
        qualifier_column = self.find_qualifier_column()
        values_column = self.find_values_column()
        date_column = self.find_date_column()
        with open(self.input_file, "r") as f:
            reader = csv.reader(f)
            next(reader)


class CompoundsFile(InputSchema):
    def __init__(self, input_file):
        InputSchema.__init__(self)

    def process(self):
        pass


class AssaysFile(InputSchema):
    def __init__(self):
        InputSchema.__init__(self)

    def process(self):
        pass


class ValuesFile(InputSchema):
    def __init__(self):
        InputSchema.__init__(self)

    def process(self):
        pass
