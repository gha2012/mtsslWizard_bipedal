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

class MonopedalEnsemble:
    def __init__(self, anchor1resi, anchor1chain, label, pdbfile, vdWcutoff = 2.5, clashes = 5):
        self.anchor1resi = anchor1resi
        self.anchor1chain = anchor1chain
        self.label = label
        self.pdbfile = pdbfile
        self.ensemble = None
        self.spinPositions = None
        self.vdWcutoff = vdWcutoff
        self.clashes = clashes
        self.filename = ""

    def createRotamers(self):
        parser = PDBParser()
        print(self.pdbfile)
        protein = parser.get_structure("protein", self.pdbfile)
        proteincoords = []
        for model in protein:
            for chain in model:
                for residue in chain:
                    if not residue.id == 'W' or not 'H_' in residue.id:
                        if not (residue.id[1] == self.anchor1resi and chain.id == self.anchor1chain):
                            for atom in residue:
                                proteincoords.append(atom.get_coord())
        proteincoords = numpy.asarray(proteincoords)

        #get the model of the label and superimpose it onto the protein
        mWlabel = MWlabel.fromfile("labels/%s.txt" % self.label)
        labelStructure = parser.get_structure("label", os.path.dirname(os.path.abspath(__file__)) + "/labels/%s.pdb" % self.label)

        labelAtoms = labelStructure.get_atoms()
        labelCoords = []
        labelSuper = []
        labelTarget = []

        #this is to get all atoms of the label
        for atomName in mWlabel.atomNames:
            labelAtoms = labelStructure.get_atoms()
            for atom in labelAtoms:
                if atom.get_name() == atomName:
                    labelCoords.append(atom.get_coord())
        #this is to get the atoms for the superposition
        for atomName in mWlabel.atomsForSuperposition:
            labelAtoms = labelStructure.get_atoms()
            for atom in labelAtoms:
                if atom.get_name() == atomName:
                    labelSuper.append(atom)

        #now get the atoms of the labelling site of the protein for the superposition
        proteinSuper = []
        chain1 = protein[0][self.anchor1chain]
        residue = chain1[int(self.anchor1resi)]
        for labelAtom in labelSuper:
            for residueAtom in residue:
                if residueAtom.get_name() == labelAtom.get_name():
                    proteinSuper.append(residueAtom)

        super_imposer = Superimposer()
        super_imposer.set_atoms(proteinSuper, labelSuper)
        super_imposer.apply(labelStructure.get_atoms())
        print("r.m.s. of label superposition: %1.3f Ang." % super_imposer.rms)

        # get new coords after superposition, put in correct order
        labelAtoms = labelStructure.get_atoms()
        labelCoords = []

        for atomName in mWlabel.atomNames:
            labelAtoms = labelStructure.get_atoms()
            for atom in labelAtoms:
                if atom.get_name() == atomName:
                    labelCoords.append(atom.get_coord())
        mWlabel.movingAtoms = labelCoords
        #generateEnsembleMulti(self, movingAtoms, environmentAtomCoordinates, numberToFind, vdWcutOff, maxClash):
        rotamers = mWlabel.generateEnsembleMulti(mWlabel.movingAtoms, proteincoords, mWlabel.numberToFind['thorough'],
                                                 self.vdWcutoff, self.clashes)

        #save spinpositions
        thisEnsembleSpinPositions = []
        print(mWlabel.spinLocation)
        for rotamer in rotamers:
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
        self.ensemble.rotamers = rotamers
        self.filename = "ensemble_%s_%s_%s%i" %(self.pdbfile, self.label, self.anchor1chain, self.anchor1resi)
        self.ensemble.writePDB(self.filename + ".pdb")

