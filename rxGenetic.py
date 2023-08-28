from mWclasses.distanceDistribution import DistanceDistribution as MWdistanceDistribution
from mWclasses.bipedalEnsemble import BipedalEnsemble
from mWclasses.monopedalEnsemble import MonopedalEnsemble

from Bio.PDB import *
import numpy
import itertools



# get information about labelling sites and labels
pdbfile = "2zd7_noSolvent.pdb" #Vsp75
#pdbfile = "2lzm_noSolvent.pdb" #2lzm
labelPositions = []

#first label
firstLabel = {}
firstLabel["bipedal"] = False
firstLabel["anchor1resi"] = 87
firstLabel["anchor1chain"] = "A"
#the second anchor point is ignored for non-bipedal labels
firstLabel["anchor2resi"] = 104
firstLabel["anchor2chain"] = "A"
firstLabel["label"] = "rx"
#vdWcutoff/clashes e.g: 2.0/25, 2.5/15, 2.5/5, 3.4/0
#for rx: 2.0/25 works good
firstLabel["vdWcutoff"] = 2.0
firstLabel["clashes"] = 25
#for rx, choose 100 runs to get good coverage of accesible volume, for the Cu labels, 10 often work fine.
firstLabel["numberOfRuns"] = 100
firstLabel["maxTries"] = firstLabel["numberOfRuns"]
#for rx, if both anchors are at consecutive AAs, this should be set to False
#The cu labels never use the internal clash - not needed
firstLabel["internalClashCheck"] = True
labelPositions.append(firstLabel)

#second label
secondLabel = {}
secondLabel["bipedal"] = True
secondLabel["anchor1resi"] = 87
secondLabel["anchor1chain"] = "B"
#the second anchor point is ignored for non-bipedal labels
secondLabel["anchor2resi"] = 104
secondLabel["anchor2chain"] = "B"
secondLabel["label"] = "rx"
#vdWcutoff/clashes e.g: 2.0/25, 2.5/15, 2.5/5, 3.4/0
secondLabel["vdWcutoff"] = 2.0
secondLabel["clashes"] = 25
#for rx, choose 100 runs to get good coverage of accesible volume, for the Cu labels, 10 often work fine.
secondLabel["numberOfRuns"] = 100
secondLabel["maxTries"] = secondLabel["numberOfRuns"]
#for rx, if both anchors are at consecutive AAs, this should be set to False
#The cu labels never use the internal clash - not needed
secondLabel["internalClashCheck"] = True
labelPositions.append(secondLabel)

#calculate ensembles
spinPositionsForDistance = []
distributionFilename = ""
for labelPosition in labelPositions:
    thisLabel = None
    if labelPosition["bipedal"]:
        thisLabel = BipedalEnsemble(labelPosition["anchor1resi"], labelPosition["anchor1chain"],
                                    labelPosition["anchor2resi"], labelPosition["anchor2chain"],
                                    labelPosition["label"], pdbfile, labelPosition["vdWcutoff"],
                                    labelPosition["clashes"], labelPosition["numberOfRuns"], labelPosition["maxTries"], labelPosition["internalClashCheck"])
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

#calculate distance distributions
dist = []
for pair in itertools.combinations(spinPositionsForDistance, 2):
    mwDistanceDistribution = MWdistanceDistribution()
    pair_distances = mwDistanceDistribution.calculateDistanceDistribution(pair[0], pair[1])
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

distributionString = "["
for row in output:
    x = row[0]
    y = row[2]
    newPoint = "{x:%1.2f, y:%1.2f}," % (x, y)
    distributionString += newPoint
distributionString += "]"
csvString = "%s\n" % averageDistance

for row in output:
    x = row[0]
    y = row[2]
    newPoint = "%1.2f\t%1.2f\n" % (x, y)
    csvString += newPoint
distributionFilename = distributionFilename[:-1] # remove trailing _
distributionFilename += ".txt"
distributionFilename = distributionFilename.replace(pdbfile, "") #remove repeated pdbfilename
distributionFilename = "distr_" + pdbfile + "_" + distributionFilename #add pdbfilename once
with open(distributionFilename,"w") as text_file:
    text_file.write(csvString)