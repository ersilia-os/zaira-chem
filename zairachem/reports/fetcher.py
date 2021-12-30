from .. import ZairaBase


class Fetcher(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)

    def get_actives_inactives(self):
        pass