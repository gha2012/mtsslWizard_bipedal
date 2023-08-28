from mWclasses.label import Label as MWlabel
from mWclasses.rotamer import Rotamer as MWRotamer
from mWclasses.ensemble import Ensemble as MWensemble
from mWclasses.distanceDistribution import DistanceDistribution as MWdistanceDistribution
from Bio.PDB import *
import numpy
import os
import random
import itertools

def generateRandomChiAngle():
    chi = random.random() * 360.0
    return chi

def rmsd(P, Q):
    diff = P - Q
    rmsd = numpy.sqrt((diff * diff).sum() / P.shape[0])
    return rmsd

class BipedalEnsemble:
    def __init__(self, anchor1resi, anchor1chain, anchor2resi, anchor2chain, label, pdbfile, vdWcutoff = 2.5, clashes = 5, numberOfRuns = 10, maxTries = 20, internalClashCheck = True):
        self.anchor1resi = anchor1resi
        self.anchor1chain = anchor1chain
        self.anchor2resi = anchor2resi
        self.anchor2chain = anchor2chain
        self.label = label
        self.pdbfile = pdbfile
        self.ensemble = None
        self.spinPositions = None
        self.vdWcutoff = vdWcutoff
        self.clashes = clashes
        self.fileName = ""
        self.numberOfRuns = numberOfRuns
        self.maxTries = maxTries
        self.internalClashCheck = internalClashCheck

    def createRotamers(self):
        # get the atoms of the protein, except at the label positions. This is needed for the clash detection
        parser = PDBParser()
        protein = parser.get_structure("protein", self.pdbfile)
        proteincoords = []
        for model in protein:
            for chain in model:
                for residue in chain:
                    if not residue.id == 'W' or not 'H_' in residue.id:
                        if not (residue.id[1] == self.anchor1resi and chain.id == self.anchor1chain):
                            if not (residue.id[1] == self.anchor2resi and chain.id == self.anchor2chain):
                                for atom in residue:
                                    proteincoords.append(atom.get_coord())
        proteincoords = numpy.asarray(proteincoords)

        # get the model of the label and superimpose it onto the protein
        mWlabel = MWlabel.fromfile("labels/%s.txt" % self.label)
        labelStructure = parser.get_structure("label", os.path.dirname(
            os.path.abspath(__file__)) + "/labels/%s.pdb" % self.label)



        labelAtoms = labelStructure.get_atoms()
        labelCoords = []
        labelSuper = []
        labelSuper2 = []
        labelTarget = []

        #this is to get the atoms for the superposition for anchor site 1 and 2 (the target site)
        for atomName in mWlabel.atomsForSuperposition:
            labelAtoms = labelStructure.get_atoms()
            for atom in labelAtoms:
                if atom.get_name() == atomName:
                    labelSuper.append(atom)
        for atomName in mWlabel.atomsForSuperposition2:
            labelAtoms = labelStructure.get_atoms()
            for atom in labelAtoms:
                if atom.get_name() == atomName:
                    labelSuper2.append(atom)

        #now get the atoms of the labelling site of the protein for the superposition
        proteinSuper = []
        proteinSuper2 = []
        chain1 = protein[0][self.anchor1chain]
        residue = chain1[int(self.anchor1resi)]
        chain2 = protein[0][self.anchor2chain]
        residueAa2 = chain2[int(self.anchor2resi)]
        mainchainO_coords1 = None
        mainchainO_coords2 = None

        #get atoms of first anchor site.
        for labelAtom in labelSuper:
            for residueAtom in residue:
                #get coordinate of main chain O atom to adjust this atom of the label accordingly
                if residueAtom.get_name() == "O":
                    mainchainO_coords1 = residueAtom.get_coord()
                if residueAtom.get_name() == labelAtom.get_name():
                    proteinSuper.append(residueAtom)
            for residueAtom in residueAa2:
                if residueAtom.get_name() == labelAtom.get_name():
                    labelTarget.append(residueAtom)

        #get atoms of anchor site 2 (the target site)
        for labelAtom in labelSuper: #need to use the names of the first anchor side, since CA2, N2, C2, ... are not present in amino acids
            for residueAtom in residueAa2:
                #get coordinate of main chain O atom to adjust this atom of the label accordingly
                if residueAtom.get_name() == "O":
                    mainchainO_coords2 = residueAtom.get_coord()
                if residueAtom.get_name() == labelAtom.get_name():
                    proteinSuper2.append(residueAtom)

        #In the following, the label is first put onto the target site (anchor 2) and its mainchain O atom (O2) is adjusted to match that of the target site.
        #This atom is not used for the rmsd calculations but should match to the corresponding atom of the protein structure.
        #Then, the label is put onto the anchor 1 site, that mainchain O atom is adjusted and the label is ready for the following calculation of the ensemble structures

        # put the second half of the label onto the target site and adjust the mainchain O atom.
        super_imposer = Superimposer()
        super_imposer.set_atoms(proteinSuper2, labelSuper2)
        super_imposer.apply(labelStructure.get_atoms())
        print("r.m.s. of label superposition on target site: %1.3f Ang." % super_imposer.rms)
        labelAtoms = labelStructure.get_atoms()
        for atom in labelAtoms:
            if atom.get_name() == "O2":
                # tweak mainchain O atom position
                atom.set_coord(mainchainO_coords2)

        #Now superimpose the label onto the first anchor site. The second mainchain O has already been corrected now.
        super_imposer = Superimposer()
        super_imposer.set_atoms(proteinSuper, labelSuper)
        super_imposer.apply(labelStructure.get_atoms())
        print("r.m.s. of label superposition: %1.3f Ang." % super_imposer.rms)

        # get new coords of the label after superposition
        labelAtoms = labelStructure.get_atoms()
        labelCoords = []

        for atomName in mWlabel.atomNames:
            labelAtoms = labelStructure.get_atoms()
            for atom in labelAtoms:
                if atom.get_name() == atomName:
                    #correct mainchain O atom position
                    if atomName == "O":
                        labelCoords.append(mainchainO_coords1)
                    else:
                        labelCoords.append(atom.get_coord())
        #The label is now sitting on the first anchor site and both mainchain O atoms are corrected to match
        #their target sites. -> update the atoms of the label object now.
        mWlabel.movingAtoms = labelCoords



        #get precalculated chi angles for the bipedal labels. This is only to ensure starting values that correspond to
        #physically reasonable structures of the labels
        if mWlabel.name == "rx":
            allChiAngles = numpy.loadtxt("rxRotamersSmaller.txt")
        elif mWlabel.name == "ida" or mWlabel.name == "nta":
            allChiAngles = numpy.loadtxt("idaRotamers.txt")

        #try to find conformations of the label that start at anchor1 and end at anchor2 (the target site)
        run = 0
        totalTries = 0
        chiAngles = []
        failed = False
        finalSelection = []
        numberOfGenerations = 100
        while run < self.numberOfRuns:
            totalTries += 1
            if totalTries > self.maxTries and len(finalSelection) == 0:
                failed = True
                break
            mWensemble = MWensemble()
            mWensemble.name = "mW"
            # generate new chi angles in every even round of the loop
            if (run % 2) == 0:
                idx = numpy.random.randint(100000, size=8000)
                if mWlabel.name == "ida" or mWlabel.name == "nta":
                    chiAngles = allChiAngles[idx, :]
                elif mWlabel.name == "rx":
                    chiAngles = allChiAngles[idx, :]

            #generate starting structures
            for i in range(len(chiAngles)):
                newRotamer = MWRotamer()
                newRotamer.id = i
                newRotamer.numberOfRotatingBonds = len(mWlabel.rotationInfo)
                newRotamer.chiAngles = chiAngles[i]
                # print(timeit.timeit(some_function, number=1000))
                #score the new rotamer
                newRotamer.score(aa1=proteinSuper, aa2=labelTarget, originalLabel=mWlabel,
                                 environmentAtomCoordinates=proteincoords, vdWcutOff=self.vdWcutoff, maxClash=self.clashes, internalClashCheck= self.internalClashCheck)
                mWensemble.rotamers.append(newRotamer)
            chiAngles = []
            print("run %i of %i (total: %i)..." % (run + 1, self.numberOfRuns, totalTries))

            #start optimization
            for generation in range(numberOfGenerations):
                mWensemble.rotamers = mWensemble.sortRotamers("rmsd")
                numberOfRotamers = len(mWensemble.rotamers)
                mWensemble.killRotamers(0.05) #0.05 works good for rx
                rotamersKilled = numberOfRotamers - len(mWensemble.rotamers)
                mWensemble.produceOffspring(rotamersKilled)
                for rotamer in mWensemble.rotamers:
                    if rotamer.rmsd == -1:
                        rotamer.score(aa1=proteinSuper, aa2=labelTarget, originalLabel=mWlabel,
                                      environmentAtomCoordinates=proteincoords, vdWcutOff=self.vdWcutoff, maxClash=self.clashes, internalClashCheck= self.internalClashCheck)
                try:
                    if mWensemble.rotamers[0].rmsd <= 0.105: # cannot get better than the rmsd of the initial superposition, which is normally 0.1.
                        # good enough!
                        break
                except:
                    print("rmsd check not working...")
                try:
                    if mWensemble.rotamers[0].rmsd > 0.4 and generation > numberOfGenerations/2:
                        #unlikely to quickly find a good solution -> give up and try again
                        break
                except:
                        pass
            if len(mWensemble.rotamers) > 0:
                #check if the quality of the found rotamers is sufficient
                if mWensemble.rotamers[0].rmsd > 0.15 and mWlabel.name == "rx":
                    # Did not find a good solution - repeat this run
                    print("Nothing good... resetting...")
                elif mWensemble.rotamers[0].rmsd > 0.2 and mWlabel.name == "ida":
                    # Did not find a good solution - repeat this run
                    print("Nothing good... resetting...")
                elif mWensemble.rotamers[0].rmsd > 0.2 and mWlabel.name == "nta":
                    # Did not find a good solution - repeat this run
                    print("Nothing good... resetting...")
                else:
                    # found something good get hold of the good chi angles and reuse them (with some noise added) to seed another run
                    # In every second run the recycled chi angles will be reset later
                    for rotamer in mWensemble.rotamers:
                        thisRotamerChiAngles = rotamer.chiAngles
                        chiAngles.append(thisRotamerChiAngles)
                    chiAngles = numpy.array(chiAngles)
                    # add noise on top of the chi angles of this run and re-use them in the next round if run is odd
                    random_values = numpy.random.uniform(-60, 60, chiAngles.shape)
                    chiAngles = chiAngles + random_values
                    finalSelection += mWensemble.rotamers[0:100]
            run += 1
        if failed == False:
            numberOfRotamers = len(finalSelection)
            pairwiseRmsds = numpy.zeros((numberOfRotamers, numberOfRotamers))
            print("removing duplicates")
            chiAngles = []
            for rotamer in finalSelection:
                #final rotamers selected, update all atoms now. The mainchain atoms are already at the final position.
                rotamer.updateAtoms()
                chiAngles.append(rotamer.chiAngles)
                if mWlabel.name == "rx":
                    atomNames = ['N', 'O', 'C', 'CA', 'CB', 'SG1', 'SD1', 'CE1', 'C14', 'C04', 'CE2', 'SD2', 'SG2', 'CB2', 'CA2', 'N2',
                             'C2', 'O2', 'C01', 'C02', 'C03', 'C24', 'C25', 'C26', 'N27', 'O28']
                elif mWlabel.name == "ida" or mWlabel.name == "nta":
                    atomNames = ['N', 'O', 'C', 'CA', 'CB', 'CG',  'CD2', 'NE2', 'Cu1', 'NE22','CD22','CG2', 'CB2', 'CA2',
                                            'O2',  'C2', 'N2', 'O3', 'O4',  'N5',  'O5',  'O6',  'O7',  'C10', 'C11', 'C12', 'C13',
                                            'C14', 'C15', 'O8', 'ND12', 'CE12', 'ND1', 'CE1']
                for atomName in atomNames:
                    rotamer.atomCoordinates.append(rotamer.atoms[atomName].coordinate)

            reducedFinalSelection = []
            reducedFinalSelection.append(finalSelection[0])
            # threshold for detection of identical rotamers
            rmsdThreshold = 0.2
            tooSimilar = False
            for i, rotameri in enumerate(finalSelection):
                for rotamerj in reducedFinalSelection:
                    if rmsd(numpy.asarray(rotameri.atomCoordinates), numpy.asarray(rotamerj.atomCoordinates)) < rmsdThreshold:
                        #if label == "rx": # the ida and nta labels are always very close to each other - dont delete duplicates
                        tooSimilar = True
                if not tooSimilar:
                    reducedFinalSelection.append(rotameri)
                tooSimilar = False

            #save spinpositions
            thisEnsembleSpinPositions = []
            print(mWlabel.spinLocation)
            for rotamer in reducedFinalSelection:
                # print rotamer.atomNames
                for idx, atomName in enumerate(mWlabel.atomNames):
                    #print(atomName)
                    if atomName == mWlabel.spinLocation:
                        coordinate = rotamer.atoms[atomName].coordinate
                        thisEnsembleSpinPositions.append(numpy.asarray(coordinate))
            self.spinPositions=thisEnsembleSpinPositions
            #spinPositionsForDistance.append(thisEnsembleSpinPositions)
            #spinPositionsForDistance.append(spinPositions)
            #write pdb file
            self.ensemble = MWensemble()
            self.ensemble.name = "mW"
            self.ensemble.rotamers = reducedFinalSelection
            self.filename = "ensemble_%s_%s_%s%i-%s%i" %(self.pdbfile, self.label, self.anchor1chain, int(self.anchor1resi), self.anchor2chain, int(self.anchor2resi))
            self.ensemble.writePDB(self.filename + ".pdb")
        else:
            print("Did not find any suitable rotamers.")