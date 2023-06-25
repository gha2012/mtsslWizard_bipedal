from operator import attrgetter
from Bio.PDB.Chain import Chain as BPChain
from Bio.PDB.Atom import Atom as BPAtom
from Bio.PDB.Residue import Residue as BPResidue
from Bio.PDB.Model import Model as BPModel
from Bio.PDB.Structure import Structure as BPStructure
from Bio.PDB import PDBIO
import random
import pickle
import numpy
import os

class Ensemble:
	def __init__(self):
		self.name = ""
		self.rotamers = []
	
	def rigid_transform_3D(self, A, B, scale):
		assert len(A) == len(B)

		N = A.shape[0]  # total points

		centroid_A = numpy.mean(A, axis=0)
		centroid_B = numpy.mean(B, axis=0)

		# center the points
		AA = A - numpy.tile(centroid_A, (N, 1))
		BB = B - numpy.tile(centroid_B, (N, 1))

		# dot is matrix multiplication for array
		if scale:
			H = numpy.transpose(BB) * AA / N
		else:
			H = numpy.transpose(BB) * AA

		U, S, Vt = numpy.linalg.svd(H)

		R = Vt.T * U.T

		# special reflection case
		if numpy.linalg.det(R) < 0:
			print("Reflection detected")
			Vt[2, :] *= -1
			R = Vt.T * U.T

		if scale:
			varA = numpy.var(A, axis=0).sum()
			c = 1 / (1 / varA * numpy.sum(S))  # scale factor
			t = -R * (centroid_B.T * c) + centroid_A.T
		else:
			c = 1
			t = -R * centroid_B.T + centroid_A.T

		return c, R, t
	
	def moveEnsemble(self, targetAtoms):
		#get current location
		N_atom = self.rotamers[0].atoms["N"].coordinate
		C_atom = self.rotamers[0].atoms["C"].coordinate
		CA_atom = self.rotamers[0].atoms["CA"].coordinate
		CB_atom = self.rotamers[0].atoms["CB"].coordinate
		currentAtoms = numpy.matrix([CA_atom, N_atom, C_atom, CB_atom])
		targetAtoms = numpy.matrix([targetAtoms[0].get_coord(), targetAtoms[1].get_coord(), targetAtoms[2].get_coord(), targetAtoms[3].get_coord()])
		print(currentAtoms)
		print(targetAtoms)
		c,R,T = self.rigid_transform_3D(numpy.asarray(targetAtoms), currentAtoms, False)
		for rotamer in self.rotamers:
			for atom in rotamer.atomNames:
				rotamer.atomNames[atom].transform(R, T)

	def tournament(self, polish = False):
		if polish:
			return self.rotamers[0]
		tournamentSize = int(len(self.rotamers)*0.005)
		# print("tournamentSize: %i" %tournamentSize)
		if tournamentSize < 1:
			tournamentSize = 1
		participants = []
		# select participants of this tournament
		# participants = [rotamer for rotamer in self.rotamers if rotamer.rmsd > 0]
		# avoid unscored rotamers to pollute tournament (rmsd=-1)
		participants = sorted(random.sample(self.rotamers, tournamentSize), key=attrgetter('rmsd'))
		participants = [rotamer for rotamer in participants if rotamer.rmsd > 0]
		return participants[0]
	
	def getGeneticOperator(self):
		operators = ["smallCreepMutation", "randomMutation", "singlePointCrossover"]
		return random.sample(operators, 1)[0]

	#-------------------------------------------------------------------------------------

	def smallCreepMutation(self, parent, spread = 15):
		children = []
		child = pickle.loads(pickle.dumps(parent, -1))
		indices = random.sample(range(len(child.chiAngles)), 1)
		for index in indices:
			oldAngle = child.chiAngles[index]
			newAngle = oldAngle + random.randrange(-spread, spread)
			child.chiAngles[index]=newAngle
			if newAngle > 360:
				newAngle -= 360
			elif newAngle < 0:
				newAngle += 360
		child.bondlength += numpy.random.uniform(-0.1, 0.1)
		child.rmsd = -1
		children.append(child)
		return children

	#-------------------------------------------------------------------------------------

	def randomMutation(self, parent):
		children = []
		# child = copy.deepcopy(parent)
		child = pickle.loads(pickle.dumps(parent, -1))
		# child = ujson.loads(ujson.dumps(parent))
		index = random.randrange(len(child.chiAngles))
		newAngle = random.randrange(360)
		child.chiAngles[index] = newAngle
		child.bondlength = numpy.random.uniform(0.9, 1.5)
		child.rmsd = -1
		children.append(child)
		return children

	#-------------------------------------------------------------------------------------

	def singlePointCrossover(self, parent1, parent2):
		childs = []
		#child1 = copy.deepcopy(parent1)
		child1 = pickle.loads(pickle.dumps(parent1, -1))
		# child1 = ujson.loads(ujson.dumps(parent1))
		# child2 = copy.deepcopy(parent2)
		child2 = pickle.loads(pickle.dumps(parent2, -1))
		# child2 = ujson.loads(ujson.dumps(parent2))
		index = random.randrange(len(child1.chiAngles))
		child1.chiAngles[0:index]=parent1.chiAngles[0:index]
		child1.chiAngles[index:len(child1.chiAngles)]=parent2.chiAngles[index:len(child1.chiAngles)]
		child2.chiAngles[0:index]=parent2.chiAngles[0:index]
		child2.chiAngles[index:len(child2.chiAngles)]=parent1.chiAngles[index:len(child2.chiAngles)]
		child1.rmsd = -1
		child2.rmsd = -1
		childs.append(child1)
		childs.append(child2)
		return childs

	
	#-------------------------------------------------------------------------------------
	
	def killRotamers(self, percentage):
		index = len(self.rotamers)
		#print(index)
		index *= 1-percentage
		#print(index)
		self.rotamers = self.rotamers[0:int(index)]
		#print(len(self.rotamers))
	
	
	def sortRotamers(self, key, reverse = False):
		#sort the rotamers according to key
		sortedRotamers = []
		if not reverse:
			sortedRotamers = sorted(self.rotamers, key=attrgetter(key))
		else:
			sortedRotamers = sorted(self.rotamers, key=attrgetter(key), reverse = True)
		#self.rotamers = sortedRotamers
		try:
			first = getattr(sortedRotamers[0], key)
			print("Best %s: %1.2f" % (key, first))
			#print(sortedRotamers[0].bondlength)
		except:
			pass
		#last = getattr(sortedRotamers[len(sortedRotamers)-1], key)
		#avg = 0
		#for rotamer in sortedRotamers:
		#	avg += getattr(rotamer, key)
		#avg/=len(sortedRotamers)

		#print "Last  %s: %1.2f" %(key, last)
		#print "Avg.  %s: %1.2f" %(key, avg)
		return sortedRotamers

	def produceOffspring(self, numberOfRotamersToProduce, polish=False):
		j = 0
		while j < numberOfRotamersToProduce:
			geneticOperator = ""
			if not polish:
				geneticOperator = self.getGeneticOperator()
			else:
				geneticOperator = "tinyCreepMutation"
			#print(geneticOperator)
			childs = []
			if geneticOperator == "smallCreepMutation":
				parent = self.tournament()
				childs = self.smallCreepMutation(parent)
				j += 1

			elif geneticOperator == "tinyCreepMutation":
				parent = self.tournament(polish = True)
				childs = self.smallCreepMutation(parent, spread = 3)
				j += 1
			elif geneticOperator == "randomMutation":
				parent = self.tournament()
				childs = self.randomMutation(parent)
				j += 1

			elif geneticOperator == "singlePointCrossover" and (numberOfRotamersToProduce - j <= 2):
				parent1 = self.tournament()
				parent2 = self.tournament()
				childs = self.singlePointCrossover(parent1, parent2)
				j += 2
			
			self.rotamers += childs

	def writePDB(self, filename = "output.pdb"):
		structure = BPStructure(1)
		for idx, rotamer in enumerate(self.rotamers):
			model = BPModel(idx + 1)
			chain = BPChain("A")
			residue = BPResidue((" ", 1, " "), "R1A", "1")
			for key in rotamer.atoms:
				atomName = key
				#print key
				element = rotamer.atoms[key].element
				x = float(rotamer.atoms[key].coordinate[0])
				y = float(rotamer.atoms[key].coordinate[1])
				z = float(rotamer.atoms[key].coordinate[2])
				atom = BPAtom(atomName, (x,y,z), 10.0, 1.0, " ", " %s "%atomName, 1, "%s"%element)
				residue.add(atom)
			chain.add(residue)
			model.add(chain)
			structure.add(model)
		io = PDBIO()
		io.set_structure(structure)
		io.save("output.tmp")
		
		#tweak output file so that it can be read as multiple states by Pymol
		bad_words = ["END","TER"]
		good_words = ["ENDMDL"]
		with open("output.tmp") as oldfile, open(filename, 'w') as newfile:
			for line in oldfile:
				if not any(bad_word in line for bad_word in bad_words) or any (good_word in line for good_word in good_words):
					newfile.write(line)
		if len(self.rotamers) > 1:
			with open(filename, 'r+') as f:
				content = f.read()
				f.seek(0, 0)
				f.write("NUMMDL    %i\n" %len(self.rotamers) + content)
		os.remove("output.tmp")
		#print "Written to file: %s/%s" %(os.getcwd(), filename)