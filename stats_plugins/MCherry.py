import StatPlugin

class MCherry(StatPlugin.StatPlugin):
    ENABLED = True

    def required_data(self):
        return list()

    def return_stats(self, data):
        return dict()