from mWclasses.distanceDistribution import DistanceDistribution as MWdistanceDistribution
from mWclasses.bipedalEnsemble import BipedalEnsemble
from mWclasses.monopedalEnsemble import MonopedalEnsemble

from Bio.PDB import *
import numpy
import itertools



# get information about labelling sites and labels
pdbfile = "4wh4_monomer_NoSolvent.pdb"
labelPositions = []

#first label
firstLabel = {}
firstLabel["bipedal"] = True
firstLabel["anchor1resi"] = 6
firstLabel["anchor1chain"] = "A"
firstLabel["anchor2resi"] = 8
firstLabel["anchor2chain"] = "A"
firstLabel["label"] = "nta"
firstLabel["vdWcutoff"] = 2.0
firstLabel["clashes"] = 15
labelPositions.append(firstLabel)

#second label
secondLabel = {}
secondLabel["bipedal"] = False
secondLabel["anchor1resi"] = 28
secondLabel["anchor1chain"] = "A"
secondLabel["anchor2resi"] = 17
secondLabel["anchor2chain"] = "A"
secondLabel["label"] = "r1"
secondLabel["vdWcutoff"] = 2.5
secondLabel["clashes"] = 5
labelPositions.append(secondLabel)

spinPositionsForDistance = []
distributionFilename = ""
for labelPosition in labelPositions:
    thisLabel = None
    if labelPosition["bipedal"]:
        thisLabel = BipedalEnsemble(labelPosition["anchor1resi"], labelPosition["anchor1chain"],
                                    labelPosition["anchor2resi"], labelPosition["anchor2chain"],
                                    labelPosition["label"], pdbfile, labelPosition["vdWcutoff"],
                                    labelPosition["clashes"])
        thisLabel.createRotamers()
        distributionFilename += thisLabel.filename
    else:
        thisLabel = MonopedalEnsemble(labelPosition["anchor1resi"], labelPosition["anchor1chain"],
                                    labelPosition["label"], pdbfile, labelPosition["vdWcutoff"],
                                    labelPosition["clashes"])
        thisLabel.createRotamers()
        distributionFilename += thisLabel.filename
    spinPositionsForDistance.append(thisLabel.spinPositions)
    distributionFilename += "_"

dist = []
for pair in itertools.combinations(spinPositionsForDistance, 2):
    # print pair
    mwDistanceDistribution = MWdistanceDistribution()
    pair_distances = mwDistanceDistribution.calculateDistanceDistribution(pair[0], pair[1])
    # print pair_distances
    dist.extend(pair_distances)
histogram = numpy.histogram(dist, numpy.arange(100))
envelopePlot = numpy.zeros((100, 2))
envelopePlot[0:99] = numpy.column_stack((histogram[1][0:len(histogram[1]) - 1], histogram[0]))

# put point in mid of bin
envelopePlot[:, 0] += 0.5
normEnvelopePlot = numpy.copy(envelopePlot)
normEnvelopePlot[:, 1] = normEnvelopePlot[:, 1] / numpy.amax(histogram[0])

# combine dist and histogram to single array before output
output = numpy.column_stack((envelopePlot, normEnvelopePlot[:, 1]))
averageDistance = numpy.average(dist)
# print averageDistance
distributionString = "["
for row in output:
    # print row
    x = row[0]
    y = row[2]
    # print x, y
    newPoint = "{x:%1.2f, y:%1.2f}," % (x, y)
    distributionString += newPoint
distributionString += "]"
csvString = "%s\n" % averageDistance
for row in output:
    # print row
    x = row[0]
    y = row[2]
    # print x, y
    newPoint = "%1.2f\t%1.2f\n" % (x, y)
    csvString += newPoint
distributionFilename = distributionFilename[:-1] # remove trailing _
distributionFilename += ".txt"
distributionFilename = distributionFilename.replace(pdbfile, "") #remove repeated pdbfilename
distributionFilename = "distr_" + pdbfile + "_" + distributionFilename #add pdbfilename once
with open(distributionFilename,"w") as text_file:
    text_file.write(csvString)