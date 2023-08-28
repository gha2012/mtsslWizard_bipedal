import math
import os
import random
from mWclasses import rotamer
from mWclasses import atom
import numpy
import scipy.spatial.distance
import re


class Label:
	def __init__(self, data):
		self.uid = data["uid"]
		self.name = data["name"]
		self.identifier = data["identifier"]
		self.modifiedAA = data["modifiedAA"]
		#self.numberOfRotatingBonds = int(data["numberOfRotatingBonds"])
		self.numberOfAtoms = int(data["numberOfAtoms"])
		self.rotate = data["rotate"]
		#self.rotationInfo = data["rotationInfo"]
		self.radius = float(data["radius"])
		self.atomNames, lastRigidAtom = self.extractAtomNamesFromInputFile(data["rotationString"])
		self.rotationInfo = self.extractRotationInfoFromInputString(data["rotationString"])
		self.spinLocation = data["spinLocation"]
		self.highlight = data["highlight"]
		self.atomsForSuperposition = data["atomsForSuperposition"].split(",")
		try:
			#This is for the anchor2 site of bipedal labels.
			self.atomsForSuperposition2 = data["atomsForSuperposition2"].split(",")
		except:
			print("No atomsForSuperposition2 found in label description. Monopedal label?")
		self.defaultVdw = data["defaultVdw"]
		self.numberToFind = self.extractThoroughness(data["numberToFind"])
		self.numberOfTries = self.extractThoroughness(data["numberOfTries"])
		self.info = data["info"]
		self.errorMessage = data["errorMessage"]
		self.trialAtomSphereRadius = float(data["trialAtomSphereRadius"])
		self.exclusionSphereRadius = float(data["exclusionSphereRadius"])
		self.pdbFile = data["pdbFile"]
		self.ensembles = {}
		self.position = 0
		self.spinPosition = ""
		self.clashAtoms = slice(lastRigidAtom, self.numberOfAtoms)
		self.movingAtoms = []

	@classmethod
	def fromfile(cls, filename):
		"Initialize label from file"
		path = os.path.dirname(__file__)
		fileContent = open("%s/%s" %(path, filename)).readlines()
		data = {}
		for string in fileContent:
			#ignore comments
			if string[0] != "#":
				string = string.rstrip()
				key = string.split("=")[0]
				value = string.split("=", 1)[1]
				data[key] = value
		return cls(data)

	def quick_map(self, atoms1, atoms2):
		# if there is only one atom it has to be duplicated for quick_dist2 to work
		duplicated = False
		if len(numpy.shape(atoms1)) == 1:
			duplicated = True
			atoms1 = numpy.tile(atoms1, (2,1))
		if len(numpy.shape(atoms2)) == 1:
			duplicated = True
			atoms2 = numpy.tile(atoms2, (2,1))
		dist = scipy.spatial.distance.cdist(atoms1, atoms2)
		#remove the duplication depending on which selection contained the single atom
		if duplicated and dist.shape[0] == 2 and not dist.shape[1] == 2:
			dist=numpy.reshape(dist[0,:], (-1, 1))

		elif duplicated and dist.shape[1] == 2 and not dist.shape[0] == 2:
			dist=numpy.reshape(dist[:,0], (-1, 1))

		elif duplicated and dist.shape[0] == 2 and dist.shape[1] == 2:
			dist=numpy.reshape(dist[:1,0], (-1, 1))
		return dist
	
	def calculateCone(self, ca, cb, environmentAtoms, numberOfAtoms = 1000):
		# generate umbrella of trial atoms
		p = 2*self.trialAtomSphereRadius * numpy.random.rand(3, numberOfAtoms)- self.trialAtomSphereRadius
		p = p[:, sum(p* p, 0) ** .5 <= self.trialAtomSphereRadius]
		p = p.transpose()
		p = p + cb
		distances = self.quick_map(ca, p)
		indices = numpy.where(numpy.any(distances < self.exclusionSphereRadius, axis=1))
		p = numpy.delete(p, indices, 0)

		#compute distances between trial sphere and environment
		distances = self.quick_map(p, environmentAtoms)
		#check for clashes and discard clashing atoms from sphere
		indices = numpy.where(numpy.any(distances < 3.5, axis=1))
		solutions = numpy.delete(p, indices,0)
		#print solutions
		numberOfConeAtoms = numpy.shape(solutions)[0]
		newRotamers = []
		id = 1
		for solution in solutions:
			newRotamer = rotamer.Rotamer()
			thisAtom = atom.Atom()
			thisAtom.coordinate = solution
			thisAtom.name = "N1"
			thisAtom.element = "N"
			newRotamer.id = id
			newRotamer.atoms["N1"] = thisAtom
			id += 1
			newRotamers.append(newRotamer)
		return newRotamers
	
	def extractThoroughness(self, string):
		dict = {}
		list = string.split(",")
		dict["painstaking"] = int(list[0])
		dict["thorough"] = int(list[1])
		dict["quick"] = int(list[2])
		return dict
	
	def extractAtomNamesFromInputFile(self, string):
		#find number of last atom that is not moved
		counter = 0
		for character in string:
			if character == "-" or character == "=" or character == "+":
				break
			if character == "x":
				counter += 1
		#add 2, because the first rotation will not displace the immediately following atom
		#e.g. for R1: NxOxCxCA-CB-SG-SD-CE-C3xO1xC2xN1xC4xC5xC6xC7xC8xC9
		#counter will count 3*x -> need to add 2
		counter += 2
		atomNames = re.split('x|-|=|\\+', string.replace("!", "")) #plus sign needs to be escaped
		string.replace("!", "")
		return atomNames, counter
	
	def extractRotationInfoFromInputString(self, string):
		#print string
		chiCounter = 1
		atomCounter = 0
		rotationInfo = {}
		correct = 0
		for idx, character in enumerate(string):
			if character == "-" or character == "=" or character == "x" or character == "+":
				atomCounter += 1
			#print atomCounter
			if character == "-" or character == "=" or character == "+":
				i = 1
				while string[idx-i] == "!":
					correct += 1
					i += 1
				#if string[idx-1] == "!":
				#	correct += 1
				rotationInfo[str(chiCounter)] = []
				rotationInfo[str(chiCounter)].append(atomCounter - 1)
				rotationInfo[str(chiCounter)].append(slice(atomCounter - 1 + 2, self.numberOfAtoms - correct))
				if character == "=":
					rotationInfo[str(chiCounter)].append(True)
				else:
					rotationInfo[str(chiCounter)].append(False)
				if character == "+":
					rotationInfo[str(chiCounter)].append("bondlength")
				chiCounter += 1
		#print rotationInfo
		return rotationInfo
					
			
	
	def getAtomIndicesInShell(self, atoms1, atoms2, lowerLimit = 2.2, upperLimit = 5.0):
		# if there is only one atom it has to be duplicated for quick_dist to work
		duplicated = False
		if len(numpy.shape(atoms1)) == 1:
			duplicated = True
			atoms1=numpy.tile(atoms1, (2,1))
		if len(numpy.shape(atoms2)) == 1:
			duplicated = True
			atoms2=numpy.tile(atoms2, (2,1))
		#calculate distances and discard second row
		dist=scipy.spatial.distance.cdist(atoms1, atoms2)[0,:]
		indices = numpy.where((dist > lowerLimit) & (dist <= upperLimit))
		return indices
	
	def internalClash2(self, atoms, refDist):
		#distances in new rotamer
		dist=scipy.spatial.distance.cdist(atoms, atoms)
		#dist = numpy.linalg.norm(atoms[:, None] - atoms, axis=-1)
		#create Boolean array with elements that describe if a distance changes or not
		changingDistances = numpy.absolute(numpy.round(numpy.subtract(dist,refDist),2)) > 0
		#multiply by Boolean array to make all constant distances zero
		dist=changingDistances*dist
		#check for internal clashes
		internalClashes=dist[numpy.nonzero((dist < 2.5) & (dist > 0))]
		#print internalClashes
		if len(internalClashes) > 0:
			return True
		else:
			return False

	def quickClash(self, rotatedAtoms, environmentAtoms, cutoff, maxClash):
		#print "rotatedAtoms: ", rotatedAtoms
		#print "environmentAtoms: ", environmentAtoms
		dist = scipy.spatial.distance.cdist(environmentAtoms, rotatedAtoms)
		clashes = len(numpy.nonzero(dist < cutoff)[0])
		if clashes > maxClash:
			return True
		else:
			return False
		
	def rotation_matrix(self, axis, theta):
		"""
		Return the rotation matrix associated with counterclockwise rotation about
		the given axis by theta radians.
		http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
		"""
		theta = math.radians(theta)
		axis = axis/math.sqrt(numpy.dot(axis, axis))
		a = math.cos(theta/2.0)
		b, c, d = -axis * math.sin(theta/2.0)
		aa, bb, cc, dd = a*a, b*b, c*c, d*d
		bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
		return numpy.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
						 [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
						 [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

	def rotatePoints2(self, points, rotationMatrix):
		#points = numpy.array(points)
		rotatedPoints = numpy.dot(points, rotationMatrix.T)
		return numpy.array(rotatedPoints)

	def generateRandomChiAngle(self):
		chi = random.random() * 360.0
		return chi

	def generatePeptideChiAngle(self):
		deltaChi = numpy.random.randint(-10, 10)
		if random.choice([True, False]):
			chi = 180 + deltaChi
		else:
			chi = 0 + deltaChi
		return chi

	def get_dihedral(self, atom1, atom2, atom3, atom4):
		"""khouli formula
		1 sqrt, 1 cross product
		http://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from
		-four-points-in-cartesian-coordinates-in-python"""
		p0 = atom1
		p1 = atom2
		p2 = atom3
		p3 = atom4

		b0 = -1.0*(p1 - p0)
		b1 = p2 - p1
		b2 = p3 - p2

		# normalize b1 so that it does not influence magnitude of vector
		# rejections that come next
		b1 /= numpy.linalg.norm(b1)

		# vector rejections
		# v = projection of b0 onto plane perpendicular to b1
		#   = b0 minus component that aligns with b1
		# w = projection of b2 onto plane perpendicular to b1
		#   = b2 minus component that aligns with b1
		v = b0 - numpy.dot(b0, b1)*b1
		w = b2 - numpy.dot(b2, b1)*b1

		# angle between v and w in a plane is the torsion angle
		# v and w may not be normalized but that's fine since tan is y/x
		x = numpy.dot(v, w)
		y = numpy.dot(numpy.cross(b1, v), w)
		return numpy.degrees(numpy.arctan2(y, x))

	def generateEnsembleMulti(self, movingAtoms, environmentAtomCoordinates, numberToFind, vdWcutOff, maxClash):
		#print("creating list with possible chi angles")
		#chiAngles=numpy.empty((0,len(self.rotationInfo)), float)
		#print(chiAngles)
		newRotamers = []
		resultsDictionary = {}
		
		numberOfRotatingBonds = len(self.rotationInfo)
		
		#reference atoms are to detect internal clashes
		referenceAtoms = numpy.copy(movingAtoms)
		refDist = scipy.spatial.distance.cdist(referenceAtoms, referenceAtoms)
		
		found = 0
		ntries = 0
		axis = numpy.zeros(shape = (2, 3)) 
		while found < numberToFind: #and ntries < maxNtries:
			#print "nTries:",ntries
			#current_task.update_state(state='PROGRESS', meta={'process_percent': int(100.0/totalNumberOfTries * (ntries + triesSoFar))})
			#reset the moving atoms to reference atoms before creating a new rotamer
			movingAtoms = numpy.copy(referenceAtoms)
			#chiAnglesForThisRotamer=[]
			for chi in range (1, numberOfRotatingBonds + 1):
				#print chi
				translationVector = numpy.array(movingAtoms[self.rotationInfo[str(chi)][0]])
				movingAtoms -= translationVector
				axis[0] = movingAtoms[self.rotationInfo[str(chi)][0]]
				axis[1] = movingAtoms[self.rotationInfo[str(chi)][0] + 1]
				
				#rotate moving atoms around axis
				if not self.rotationInfo[str(chi)][2]:
					angle = self.generateRandomChiAngle()
				else:
					#need to calculate the dihedral of the input label to calculate the necessary adjustment.
					targetDihedral = self.generatePeptideChiAngle()
					currentDihedral = self.get_dihedral(axis[0], axis[1], movingAtoms[self.rotationInfo[str(chi)][1]][0], movingAtoms[self.rotationInfo[str(chi)][1]][1])
					angle = targetDihedral - currentDihedral
				#chiAnglesForThisRotamer.append(angle)
				rotationMatrix = self.rotation_matrix(axis[1], angle)
				movingAtoms[self.rotationInfo[str(chi)][1]] = self.rotatePoints2(movingAtoms[self.rotationInfo[str(chi)][1]], rotationMatrix)
				movingAtoms += translationVector
			
			if not self.quickClash(movingAtoms[self.clashAtoms], environmentAtomCoordinates, vdWcutOff, maxClash):
				if not self.internalClash2(movingAtoms, refDist):
					newRotamer = rotamer.Rotamer()
					newRotamer.id = found
					movingAtomNames = self.atomNames
					for idx, atomName in enumerate(movingAtomNames):
						if idx >= 0:
							thisAtom = atom.Atom()
							thisAtom.coordinate = movingAtoms[idx]
							thisAtom.name = atomName
							thisAtom.element = atomName[0]
							
							#add atom to rotamer
							newRotamer.atoms[atomName] = thisAtom
					#newrow = numpy.asarray(chiAnglesForThisRotamer)
					#chiAngles = numpy.vstack([chiAngles,newrow])

					#chis = ",".join(["%1.2f" %i for i in chiAnglesForThisRotamer])
					#rotamerString = "%1.2f,%1.2f,%1.2f,%1.2f,%s" %(caDistance,cbDistance,nDistance,cDistance,chis)
					#print(rotamerString)
					newRotamers.append(newRotamer)
					found += 1
					#print(found),
				else:
					pass
					#print("i"),
			else:
				pass
				#print("."),
			ntries += 1
		#print(chiAngles)
		#numpy.savetxt("idaRotamers.txt", chiAngles[chiAngles[:, 0].argsort()])
		return newRotamers

