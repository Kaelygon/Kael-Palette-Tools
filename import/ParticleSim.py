import math
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from numpy.typing import NDArray

from OkTools import *
from PointList import *

ParticleType = [
	("pos",float, 3),	#position
	("fixed",bool), 	#is immovable?
	("v",float, 3),	#velocity
	("A",float), 		#Drag reference area
	("Cd",float), 		#Coefficient of drag
	("rho",float), 	#Drag (rho) fluid density
	("m",float), 		#Mass
	("k",float), 		#sprint constant
	("COR",float), 	#Coefficient of restitution
	("mu",float), 		#coefficient of friction
	("radius",float), #point radius
	("clipped",bool), #was near gamut boundary
	("stress",float),	#spring crushing froce
]

class ParticleSim:
	"""Physics based spring/collision simulator to distribute points."""

	#Too far: attract, Too close: repel # f=-kx (single particle)
	def calcSpringForce(self, particle: ParticleType, neighbors_dist_sq: [float]*3, neighbor_norm: [float]*3, spring_constant: float = 1.0):
		delta_x = np.sqrt(neighbors_dist_sq) - particle["radius"]
		spring_mag = -1.0 * spring_constant * delta_x #f=-kx
		spring_force = neighbor_norm*spring_mag[:,None]
		spring_force = np.sum(spring_force,axis=0)
		return spring_force

	def moveToVelocity(self, move_delta: [float]*3, time_delta: float):
		return move_delta / time_delta #v = dx/dt
	
	def velocityToForce(self, particles: ParticleType, velocity: [float]*3, time_delta: float):
		return velocity * particles["m"] / time_delta #f = dv*m/dt
	
	def forceToVelocity(self, particles: ParticleType, force: [float]*3, time_delta: float):
		velocity = force * time_delta / particles["m"][:,None] #dv = f*dt/m
		return velocity

	#force from distance moved in time step. f=m*((dx/dt)/dt)
	def moveToForce(self, particles: ParticleType, move_delta: [float]*3, time_delta: float):
		velocity = self.moveToVelocity(move_delta, time_delta) #v = dx/dt
		force = self.velocityToForce(particles, velocity, time_delta)
		return force

	#returns the force needed to reflect p0 velocity  # f = - 2 * (vec . n) * n
	def calcReflectVelocity(self, particles: NDArray[ParticleType], surface_norm: [float]*3, time_delta: float):
		velocity = particles["v"]
		dot_v = np.sum(velocity*surface_norm, axis=1) #(vec . n)
		reflected_v = -1.0 * (particles["COR"]+1.0) * dot_v # - 2 * (vec . n)
		reflected_v = reflected_v[:,None] * surface_norm # - 2 * (vec . n) * n
		velocity = velocity + reflected_v
		return velocity

	def calcTotalEnergy(self, particles: NDArray[ParticleType]):
		kE = 0.5 * particles["m"] * np.linalg.norm(particles["v"],axis=1)**2
		total_energy = np.sum(kE,axis=0)
		return total_energy

	#higher smoothing = bias toward prev_dt if dt increases
	def calcTimeStep(self, prev_dt: float, particles: NDArray[ParticleType], unit_size: float, smoothing: float = 0.5, max_dt: float = 0.5):
		v_sq = np.sum(particles["v"]**2,axis=1)
		max_v_sq = np.max(v_sq)

		new_dt=max_dt
		if max_v_sq != 0.0:
			new_dt = unit_size**2 / max_v_sq #t=d/v #limit movement distance per tick

		if new_dt>prev_dt:
			new_dt = new_dt*(1.0-smoothing) + prev_dt*smoothing
		new_dt = max(min(new_dt,max_dt),1e-4)

		return new_dt

	#update particle.radius adaptively
	def calcParticleRadius(self, particles: NDArray[ParticleType], neighbor_idxs: list[int], max_r_change: float=0.2, time_delta: float=1.0):
		output_radii = np.zeros_like(particles["radius"]) - 1
		particle_count = len(particles)

		radii = particles["radius"]
		scale_limit = radii*max_r_change*time_delta #max radius change in time step

		approx_speed = np.sum(particles["v"],axis=1)
		almost_stopped = approx_speed < (0.1*radii/time_delta)

		neighbor_count = np.zeros(particle_count)
		for i,neighbors in enumerate(neighbor_idxs):
			neighbor_count[i] = len(neighbors)

		#no neighbors cases 
		no_neighbors = neighbor_count==0
		moving_and_alone = no_neighbors & ~almost_stopped
		stopped_and_alone = no_neighbors & almost_stopped

		output_radii[moving_and_alone] = radii[moving_and_alone] #no change
		output_radii[stopped_and_alone] = radii[stopped_and_alone] + scale_limit[stopped_and_alone] #max change
		
		#Scale radius kissing spheres 
		#fewer kissing spheres results in radius increase
		#
		#["clipped"] means the point is hugging a wall, which results kissing_number being lower. So we subtract 6 from kissing target
		#Corners are excpetion where kissing number would be even lower, 
		#but we ignore this because point relation to gamut croners is hard to predict. This may bias corners being sparser.
		kissing_target = 11 - particles["clipped"]*6 
		kissing_number = neighbor_count
		kissing_scale = (kissing_target/(kissing_number+1))**(1/3) 

		#Stress and median radius
		avg_stress = np.zeros(particle_count)
		median_radius = np.zeros(particle_count)
		for i, neighbors in enumerate(neighbor_idxs):
			if len(neighbors) == 0:
				avg_stress[i] = 0.0
				median_radius[i] = particles["radius"][i]
				continue

			avg_stress[i] = np.median(particles["stress"][neighbors])
			median_radius[i] = np.median(particles["radius"][neighbors])

		#If point is being squeezed very hard, the local region is considered unstable, so we limit radius growth
		is_unstable = (particles["stress"] > 2.0*avg_stress) & almost_stopped

		#We also limit clipping points radius growth
		unstable_or_clipping = is_unstable | particles["clipped"]
		if np.any(unstable_or_clipping):
			kissing_scale[unstable_or_clipping] = np.clip(kissing_scale[unstable_or_clipping], 0.0, 1.0)

		new_radius = median_radius*kissing_scale
		new_radius = np.clip(new_radius, radii*0.5, radii+scale_limit)

		has_neighbors = ~no_neighbors
		output_radii[has_neighbors] = new_radius[has_neighbors]

		invalid_output = output_radii==-1
		if np.any(invalid_output):
			print("Missing outputs")
			exit(-1)

		output_radii = np.clip(output_radii,1e-20,2.0)
		return output_radii


	def createParticles(self, point_list: PointList, start_radius: float):

		base_point = np.zeros((1),dtype=ParticleType)

		base_point["pos"]=[-1,-1,-1]
		base_point["fixed"]=-1

		base_point["v"]=[0.0, 0.0, 0.0]
		base_point["A"]=math.pi * (0.1)**2 #m^2
		base_point["Cd"]=0.47 #sphere
		base_point["rho"]=3.0 #kg/l
		base_point["m"]=0.4 #kilograms
		base_point["k"]=0.4
		base_point["COR"]=0.6
		base_point["mu"]=0.2
		base_point["radius"]=start_radius
		base_point["clipped"]=False
		base_point["stress"]=0.0

		particle_list = np.full( (len(point_list.points)), base_point )

		particle_list["pos"] = point_list.points["color"]
		particle_list["fixed"] = point_list.points["fixed"]

		return particle_list

	"""
	Relations
	Fd=0.5*p*v^2*A*Cd : force drag
	Fs=-kx : sprint force
	F=ma : force
	a=dv/dt : acceleration
	w = v - 2 * (v . n) * n : Reflect
	"""
	#Iterative force simulation that pushes points 1 radius apart from each center
	def relaxCloud(self,
		point_list,
		approx_radius,
		iterations = 64,
		min_energy = 0.0,
		record_frames = None,
		log_frequency = 32,
		max_r_change = 0.1, #max change in radius per 1 sim_dt
		anneal_steps = None
	):
		if iterations == 0 or len(point_list.points) == 0:
			return

		self.sim_dt = 0.1

		if anneal_steps == None:
			anneal_steps = min(iterations//8, 64)

		particles = self.createParticles(point_list, approx_radius)

		if record_frames:
			particle_frames=[]
			particle_frames.append(particles["pos"].tolist())

		#min,median,max radius of particles["radius"]
		min_r,med_r,max_r = [approx_radius]*3

		for tick in range(iterations):
			#np.random.shuffle(particles)

			if tick>0:
				unit_size = min(0.25,min_r) * 0.5 #gamut dims is ~1x1x1
				self.sim_dt = self.calcTimeStep(self.sim_dt, particles, unit_size=unit_size, smoothing=0.5, max_dt = 0.5)
		
			#Apply movement from velocity
			movable_idxs = ~particles["fixed"]
			delta_dist = particles["v"] * self.sim_dt #dx = v * dt

			new_pos = particles["pos"] + delta_dist
			invalid_pos = np.any(np.isnan(new_pos), axis=1)

			particles["pos"][movable_idxs] = new_pos[movable_idxs]
			particles["pos"][invalid_pos] = [0.0]*3

			if record_frames:
				frame = particles["pos"].tolist()
				particle_frames.append(frame)


			#Calc and sum forces
			#get neighbors of each point within its radius
			point_tree = cKDTree(particles["pos"])
			neighbors_list = point_tree.query_ball_point(particles["pos"], r=particles["radius"]*1.2)
			for i, neighbors in enumerate(neighbors_list):
				if i in neighbors:
					neighbors.remove(i) #remove self

			#Adaptive radius scale
			if(
				iterations-tick >= anneal_steps and 
				tick >= anneal_steps and 
				tick%anneal_steps==0
			):
				particles["radius"] = self.calcParticleRadius(particles, neighbors_list, max_r_change, self.sim_dt)

			#gamut reflect
			particles["pos"], clip_move = OkTools.clipToOklabGamut(particles["pos"]) # dx
	
			non_zero_move = np.any(clip_move != [0.0]*3, axis=1)
			particles["clipped"] = non_zero_move

			if np.any(non_zero_move):
				surface_norm = OkTools.vec3_array_norm(clip_move[non_zero_move])
				particles["v"][non_zero_move] = self.calcReflectVelocity(particles[non_zero_move], surface_norm, self.sim_dt)


			all_forces=[]
			#neighbor forces
			#every point has different number of neighbors so this can't be fully vectorized
			neighbor_force = np.zeros_like(particles["v"])
			fixed_bounce_norm = np.zeros_like(particles["v"])
			fixed_bounce_idxs = []
			for i, neighbor_idxs in enumerate(neighbors_list):
				if particles["fixed"][i]:
					continue

				neighbor_count = len(neighbor_idxs)
				if neighbor_count == 0:
					continue

				neighbors_dist_vec = particles["pos"][i] - particles["pos"][neighbor_idxs]
				neighbors_dist_sq = np.sum(neighbors_dist_vec**2,axis=1)

				#push away from first overlapping neighbor if any
				inside_neighbor = neighbors_dist_sq == 0.0
				if np.any( neighbors_dist_sq==0.0 ):
					rand_norm = np.random.rand(3)
					push_move = particles["radius"][i] * self.sim_dt * rand_norm
					push_force = self.moveToForce(particles[i], push_move, self.sim_dt)
					neighbor_force[i]+= push_force
					continue

				#spring influence
				spring_mul = np.zeros((neighbor_count)) + 1.0
				outside_radius = neighbors_dist_sq > particles["radius"][i]**2
				spring_mul[outside_radius] = 0.0 #attraction force

				neighbor_norm = OkTools.vec3_array_norm(neighbors_dist_vec)
				spring_force = self.calcSpringForce(particles[i], neighbors_dist_sq, neighbor_norm, spring_mul*particles[i]["k"])
				neighbor_force[i]+= spring_force

				#backlog bounces if inside fixed
				fixed_neighbors = particles[neighbor_idxs]["fixed"]
				if np.any(fixed_neighbors): 
					colliding_fixed_dist_vec = neighbors_dist_vec[fixed_neighbors][0]
					fixed_bounce_norm[i] = OkTools.vec3_norm(colliding_fixed_dist_vec)
					fixed_bounce_idxs.append(i)

			#apply fixed bounces
			if len(fixed_bounce_idxs):
				fixed_bounce_norm = fixed_bounce_norm[fixed_bounce_idxs]
				particles["v"][fixed_bounce_idxs] = self.calcReflectVelocity(particles[fixed_bounce_idxs], fixed_bounce_norm, self.sim_dt)

			particles["stress"] = abs(np.sum(neighbor_force,axis=1))
			all_forces.append(neighbor_force)


			#drag, opposing #Fd=0.5*p*A*Cd*v^2
			drag_factors = -1.0 * 0.5 * particles["rho"] * particles["A"] * particles["Cd"]
			drag_force = drag_factors[:,None] * particles["v"]**2
			all_forces.append(drag_force)


			#internal friction, opposing
			friction_force = particles["v"] * (-1.0*particles["mu"])[:,None] #f=cf
			all_forces.append(friction_force)


			#sum Fdt
			force_delta=np.sum(all_forces,axis=0)
			delta_velocity = self.forceToVelocity(particles, force_delta, self.sim_dt)
			particles["v"] = particles["v"] + delta_velocity
		
			#logging
			if tick%log_frequency==0 or tick==iterations-1:
				total_energy=self.calcTotalEnergy(particles)
				prev_energy=total_energy
				out_str = "DT: " + str(round(self.sim_dt,4)) + " Total energy["+str(tick)+"]:"
				out_str+= " "+str(total_energy)
				print(out_str)
			
				min_r = round(np.min(particles["radius"]),4)
				med_r = round(np.median(particles["radius"]),4)
				max_r = round(np.max(particles["radius"]),4)
				print("P: minr " + str(min_r) + " medr" + str(med_r) + " maxr" + str(max_r))
				if abs(total_energy) < min_energy:
					break
		
		if record_frames:
			with open(record_frames, "w") as f:
				f.write("oklab_frame_list = ")
				f.write(str(particle_frames))
	

		print("relaxCloud Done\n")
		point_list.points["color"] = particles["pos"]
		return point_list