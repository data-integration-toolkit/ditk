# Representa los subobjetivos de un CQ
class Predicado:
    nombre = ""
    def __init__(self, nom):
        self.nombre = nom

    def __str__(self):
        return self.nombre
    