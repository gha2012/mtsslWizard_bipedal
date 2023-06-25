import numpy
import scipy.spatial.distance


class DistanceDistribution:
    def calculateDistanceDistribution(self, spinCenter1Coordinates, spinCenter2Coordinates):
        # Setup parallel computation of distances
        dist = self.quickDistMulti(spinCenter1Coordinates, spinCenter2Coordinates)
        return dist

    def quickDistMulti(self, atoms1, atoms2):
        # if there is only one atom it has to be duplicated for quick_dist to work
        duplicated = False
        atoms1 = numpy.array(atoms1)
        atoms2 = numpy.array(atoms2)

        if len(numpy.shape(atoms1)) == 1:
            duplicated = True
            atoms1 = numpy.tile(atoms1, (2, 1))
        if len(numpy.shape(atoms2)) == 1:
            duplicated = True
            atoms2 = numpy.tile(atoms2, (2, 1))
        dist = scipy.spatial.distance.cdist(atoms1, atoms2)

        # remove the duplication depending on which selection contained the single atom
        if duplicated and dist.shape[0] == 2 and not dist.shape[1] == 2:
            dist = numpy.reshape(dist[0, :], (-1, 1))

        elif duplicated and dist.shape[1] == 2 and not dist.shape[0] == 2:
            dist = numpy.reshape(dist[:, 0], (-1, 1))

        elif duplicated and dist.shape[0] == 2 and dist.shape[1] == 2:
            dist = numpy.reshape(dist[:1, 0], (-1, 1))
        else:
            dist = numpy.reshape(dist, (-1, 1))

        return dist