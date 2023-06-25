import numpy


class Atom:
    def __init__(self):
        self.name = ""
        self.resn = ""
        self.resi = ""
        self.element = ""
        self.coordinate = numpy.zeros(3)
        self.isSpinCenter = False
        self.indices = {}
