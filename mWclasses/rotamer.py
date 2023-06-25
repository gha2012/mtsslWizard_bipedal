import numpy
import scipy.spatial.distance
from mWclasses.atom import Atom as MWAtom
import copy

class Rotamer:	
	def __init__(self):
		self.id = 0
		self.chi2 = 0
		self.scaledChi2 = 0
		self.atoms = {}
		self.atomNames = []
		self.weights =   []
		self.totalContacts = 0
		self.atomCoordinates = []
		self.initialAtomCoordinates = []
		self.chiAngles = []
		self.rmsd = -1
		for atomName in self.atomNames:
			newAtom = MWAtom.Atom()
			self.atoms[atomName] = newAtom
		self.numberOfRotatingBonds = -1
		self.environmentAtomCoordinates = -1
		self.clashAtoms = []
		self.movingAtoms = []
		self.movingAtomNames = []
		#for ida and nta label, allow tuning of n-cu bondlength
		self.bondlength = numpy.random.uniform(0.5, 1.5)
		
		
	def calculateRmsd(self, P, Q):
		# diff = P - Q
		squared_distance = numpy.sum((P - Q) ** 2, axis=1)
		weights = numpy.ones(P.shape[0])
		weights[3] = 1 #3 works for rx
		weights[2] = 1 #3 works for rx
		squared_distance = squared_distance * weights
		rmsd = numpy.sqrt(numpy.mean(squared_distance))
		#rmsd = numpy.sqrt((diff * diff).sum() / P.shape[0])
		return rmsd
	
	def updateAtoms(self):
		for idx, atomName in enumerate(self.movingAtomNames):
			if idx >= 0:
				thisAtom = MWAtom()
				thisAtom.coordinate = self.movingAtoms[idx]
				thisAtom.name = atomName
				thisAtom.element = atomName[0]		
				self.atoms[atomName] = thisAtom
		return
		
	
	def score(self, aa1, aa2, originalLabel, environmentAtomCoordinates, vdWcutOff, maxClash):
		aa2Array = numpy.array([aa2[0].get_coord(), aa2[1].get_coord(), aa2[2].get_coord(), aa2[3].get_coord()])
		mWlabel = copy.copy(originalLabel)
		#print(mWlabel.rotationInfo)
		referenceAtoms = numpy.copy(mWlabel.movingAtoms)
		refDist = scipy.spatial.distance.cdist(referenceAtoms, referenceAtoms)
		axis = numpy.zeros(shape = (2, 3)) 
		movingAtoms = numpy.copy(referenceAtoms)
		for chi in range (1, self.numberOfRotatingBonds + 1):
			# print(chi)
			translationVector = numpy.array(movingAtoms[mWlabel.rotationInfo[str(chi)][0]])
			movingAtoms -= translationVector
			axis[0] = movingAtoms[mWlabel.rotationInfo[str(chi)][0]]
			axis[1] = movingAtoms[mWlabel.rotationInfo[str(chi)][0] + 1]
			angle = self.chiAngles[chi - 1]
			rotationMatrix = mWlabel.rotation_matrix(axis[1], angle)
			movingAtoms[mWlabel.rotationInfo[str(chi)][1]] = mWlabel.rotatePoints2(movingAtoms[mWlabel.rotationInfo[str(chi)][1]], rotationMatrix)
			# movingAtoms[mWlabel.rotationInfo[str(chi)][1]] = mWlabel.rotatePoints_quaternions(movingAtoms[mWlabel.rotationInfo[str(chi)][1]], axis[1], angle)      #rotatePoints_quaternions(self, points, axis, angle):
			movingAtoms += translationVector
			#print("no")
			if len(mWlabel.rotationInfo[str(chi)]) == 4: # this should only run when there is a "d" in the rotation string, i.e. when the bond length is supposed to be adjusted.
				#print("bondlength")
				distance = numpy.sqrt(numpy.sum((axis[0] - axis[1]) ** 2))
				newDistance = distance * self.bondlength

				if newDistance < 1.9 or newDistance > 2.2: #restrict to sensible bondlengths
					#print("Too long or too short:", newDistance, distance)
					newDistance = distance
				#print(newDistance)
				direction = (axis[1] - axis[0]) / distance
				newBondVector = direction * newDistance
				oldBondVector = axis[1] - axis[0]
				translationVector = newBondVector - oldBondVector
				#print(direction)
				movingAtoms[mWlabel.rotationInfo[str(chi)][0] + 1] += translationVector
				movingAtoms[mWlabel.rotationInfo[str(chi)][1]] += translationVector
				#for thisAtom in movingAtoms[mWlabel.rotationInfo[str(chi)][1]]:
				#	#print(mWlabel.rotationInfo[str(chi)][1])
					#print("next atom:")
					#print(atom)

					#print(translationVector)
				#	print(thisAtom)
				#	thisAtom += translationVector
					#print(atom)

		#   0    1    2    3     4     5      6      7	    8      9      10     11     12     13     14     15    16    17    18     19     20     21     22     23     24     25     26     27     28     29    30      31      32     33
		# ['N', 'O', 'C', 'CA', 'CB', 'SG1', 'SD1', 'CE1', 'C14', 'C04', 'CE2', 'SD2', 'SG2', 'CB2', 'CA2', 'N2', 'C2', 'O2', 'C01', 'C02', 'C03', 'C24', 'C25', 'C26', 'N27', 'O28']
		# ['N', 'O', 'C', 'CA', 'CB', 'CG',  'CD2', 'NE2', 'Cu1', 'NE22','CD22','CG2', 'CB2', 'CA2', 'O2',  'C2', 'N2', 'O3', 'O4',  'N5',  'O5',  'O6',  'O7',  'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'O8', 'ND12', 'CE12', 'ND1', 'CE1']
		# ['N', 'O', 'C', 'CA', 'CB', 'CG',  'CD2', 'NE2', 'Cu1', 'NE22','CD22','CG2', 'CB2', 'CA2', 'O2',  'C2', 'N2', 'ND12','CE12','O3', 'O4',  'N5',  'O5',  'O6',  'O7',  'C10', 'C11', 'C12', 'C13', 'C14','C15',  'O8',   'ND1', 'CE1']

		if originalLabel.name == "rx":
			clashAtomIndices = [5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22, 23, 24, 25]
		elif originalLabel.name == "ida":
			clashAtomIndices = [5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
		elif originalLabel.name == "nta":
			clashAtomIndices = [5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
		clashAtoms = []
		#print(mWlabel.atomNames)
		for index in clashAtomIndices:
			clashAtoms.append(movingAtoms[index])
		self.clashAtoms = clashAtoms
		self.movingAtoms = movingAtoms
		self.movingAtomNames = mWlabel.atomNames
		debug = False
		if debug == True:
			if originalLabel.name == "rx":
				trialAa2Array = numpy.array([movingAtoms[14], movingAtoms[15], movingAtoms[16], movingAtoms[13]])
			elif originalLabel.name == "ida":
				trialAa2Array = numpy.array([movingAtoms[13], movingAtoms[16], movingAtoms[15], movingAtoms[12]])
			self.rmsd = self.calculateRmsd(aa2Array, trialAa2Array)
		else:
			if not mWlabel.quickClash(clashAtoms, environmentAtomCoordinates, vdWcutOff, maxClash):
				if not mWlabel.internalClash2(movingAtoms, refDist) or originalLabel.name == "nta" or originalLabel.name == "ida": #ignore internal clashes for nta or ida, adjusting the N-Cu bonds will screw the check up.
					# only update the main chain atoms now for the rmsd calculation. The remaining atoms are updated at the
					# very end.
					if originalLabel.name == "rx":
						trialAa2Array = numpy.array([movingAtoms[14], movingAtoms[15], movingAtoms[16], movingAtoms[13]])
					elif originalLabel.name == "ida":
						trialAa2Array = numpy.array([movingAtoms[13], movingAtoms[16], movingAtoms[15], movingAtoms[12]])
					elif originalLabel.name == "nta":
						trialAa2Array = numpy.array([movingAtoms[13], movingAtoms[16], movingAtoms[15], movingAtoms[12]])
					self.rmsd = self.calculateRmsd(aa2Array, trialAa2Array)
				else:
					#print("internal clash!")
					self.rmsd = 9999
			else:
				#print("clash!")
				self.rmsd = 9999
		
		

		
	