#ParticleSim.py
from dataclasses import dataclass, fields
import numpy as np
from scipy.spatial import cKDTree

from OkTools import *
from PointList import *


class ParticleType:
	"""ParticleType ParticleType(PointList point_list, float start_radius)"""
	@dataclass(frozen=True)
	class DEF: #default values
		v			= [0.0,0,0]
		A			= np.pi * (0.1)**2
		Cd			= 0.47
		rho		= 3.0
		m			= 0.4
		k			= 0.4 
		COR		= 0.6 
		mu			= 0.1
		clipped	= False 
		stress	= 0.0 

	pid:		"int pid"				#identifier
	pos:		"float[][3]	pos"		#position
	fixed:	"bool[]		fixed"	#is immovable?
	v:			"float[][3]	v"			#velocity
	v_mag:	"float[] 	v_mag"	#velocity magnitude
	A:			"float[]		A"			#area
	Cd:		"float[]		Cd"		#coefficiency of drag
	rho:		"float[]		rho"		#fluid density
	m:			"float[]		m"			#mass
	k:			"float[]		k"			#spring constant
	COR:		"float[]		COR"		#coefficiency of restitution
	mu:		"float[]		mu"		#internal friction
	radius:	"float[]		radius"	#radius (OKLab units)
	clipped:	"bool[]		clipped"	#previous tick, was outside gamut boundary
	stress:	"float[]		stress"	#previous tick, neighbors spring force sum

	def __init__(self, point_list, start_radius):
		D = self.DEF
		p_count = len(point_list)
		self.pid 		= np.arange(p_count)
		self.pos			= point_list.points["color"].copy()
		self.fixed		= point_list.points["fixed"].copy()
		self.v			= np.zeros((p_count, 3)) + D.v
		self.v_mag		= np.full(p_count , OkTools.vec3Length(D.v) )
		self.A			= np.full(p_count , D.A 			)
		self.Cd			= np.full(p_count , D.Cd			)
		self.rho			= np.full(p_count , D.rho			)
		self.m			= np.full(p_count , D.m				)
		self.k			= np.full(p_count , D.k				)
		self.COR			= np.full(p_count , D.COR			)
		self.mu			= np.full(p_count , D.mu			)
		self.radius		= np.full(p_count , start_radius	)
		self.clipped	= np.full(p_count , D.clipped		)
		self.stress		= np.full(p_count , D.stress 		)

	def __len__(self):
		return len(self.pos)

	def shuffle(self, order):
		"""void forceToAcceleration(int[] order)"""
		for name, val in vars(self).items():
			val[:] = val[order]

	def createVariance(self, rand, scale):
		"""void forceToAcceleration(np.random.Generator rand, float scale)"""
		def scale_noise(shape, scale, dims=1):
			return 1.0 - (rand.random(shape) - 0.5) * scale

		count = len(self)
		noise_v = (rand.random((count,3)) - 0.5) * scale
		noise_m = scale_noise(count, scale)
		noise_k = scale_noise(count, scale)
		noise_radius = scale_noise(count, scale)
		noise_mu = scale_noise(count, scale)

		self.v = noise_v * self.radius[:,None]
		self.m*=noise_m
		self.k*=noise_k
		self.radius*=noise_radius
		self.mu*=noise_mu
		return

class ParticleSim:
	"""Physics based spring/collision simulator to distribute points."""

	#Calculate spring force from list of neighbor dist and norm 
	def calcSpringForce(self, radius, spring_constant, n_valid_idxs, neighbors_dist, neighbors_norm, attraction_force = 0.0 ):
		"""
		(spring_force float[][3], spring_mag float[]) calcSpringForce(self, 
			float[]	 		radius,
			float[]	 		spring_constant,
			bool[][] 		n_valid_idxs,
			float[][] 		neighbors_dist,
			float[][][3] 	neighbors_norm,
			float				attraction_force,
		)
		"""

		beyond_radius = neighbors_dist > radius[:,None]
		spring_k_array = np.where(beyond_radius, attraction_force, spring_constant[:, None])

		delta_x = np.zeros_like(neighbors_dist)
		delta_x[n_valid_idxs] = neighbors_dist[n_valid_idxs]
		delta_x = delta_x - radius[:,None] * n_valid_idxs
		spring_mag = -1.0 * spring_k_array * delta_x #f=-kx
		spring_force = neighbors_norm*spring_mag[:,:,None]
		spring_force = np.sum(spring_force,axis=1)
		spring_mag = np.sum(spring_mag,axis=1)
		return spring_force, spring_mag

	def forceToAcceleration(self, mass, force):
		"""float[][3] forceToAcceleration(float[] mass, float[][3] force)"""
		acceleration = force / mass[:,None] #a=f/m
		return acceleration

	#returns reflected velocity  # f = - 2 * (vec . n) * n
	def calcReflectVelocity(self, velocity, COR, surface_norm):
		"""float[][3] calcReflectVelocity(float[][3] velocity, float[] COR, float[][3] surface_norm)"""
		dot_v = np.sum(velocity*surface_norm, axis=1) #(vec . n)
		reflected_v = -1.0 * (COR+1.0) * dot_v # - 2 * (vec . n)
		reflected_v = reflected_v[:,None] * surface_norm # - 2 * (vec . n) * n
		velocity = velocity + reflected_v
		return velocity

	#Scale radius by comparing neighbor states
	def calcParticleRadius(self, particles, n_pad_idxs, n_valid_idxs, neighbor_count, max_r_change, time_delta):
		"""
		float[] calcParticleRadius(
			ParticleType particles, 
			int[][] n_pad_idxs, 
			bool[][] n_valid_idxs, 
			int[] neighbor_count
			float max_r_change, 
			float time_delta
		)
		"""
		output_radii = np.zeros_like(particles.radius)
		no_neighbors = neighbor_count==0
		scale_limit = particles.radius*max_r_change*time_delta #max radius change in time step
		output_radii[no_neighbors] = particles.radius[no_neighbors] + scale_limit[no_neighbors] #max change
		
		#Scale radius kissing spheres 
		#fewer kissing spheres results in radius increase
		#
		#.clipped means the point is hugging a wall, which results kissing_number being lower. So we subtract 6 from kissing target
		#Corners are excpetion where kissing number would be even lower, 
		#but we ignore this because point relation to gamut croners is hard to predict. This may bias corners being sparser.
		kissing_target = 11 - particles.clipped*6 
		kissing_number = neighbor_count
		kissing_scale = (kissing_target/(kissing_number+1))**(1/3) 

		n_stress = np.ma.masked_array(particles.stress[:, None] * n_valid_idxs, mask=~n_valid_idxs)
		n_radius = np.ma.masked_array(particles.radius[:, None] * n_valid_idxs, mask=~n_valid_idxs)
		median_stress	= np.ma.median(n_stress, axis=1).filled(0.0)
		median_radius 	= np.ma.median(n_radius, axis=1).filled(particles.radius)

		#Limit unstable and clipped particles radius growth
		is_unstable = particles.stress > 2.0*median_stress
		unstable_or_clipping = is_unstable | particles.clipped
		if np.any(unstable_or_clipping):
			kissing_scale[unstable_or_clipping] = np.clip(kissing_scale[unstable_or_clipping], 0.0, 1.0)

		new_radius = median_radius*kissing_scale
		new_radius = np.clip(new_radius, particles.radius*0.5, particles.radius+scale_limit)

		has_neighbors = ~no_neighbors
		output_radii[has_neighbors] = new_radius[has_neighbors]
		output_radii = np.clip(output_radii,1e-20,2.0)
		return output_radii

	def updateParticleNeighbors(self, particles, search_radius):
		"""int[][] bool[][] int[] updateParticleNeighbors(ParticleType particles, float search_radius)"""
		## Get neighbors of each point within its radius
		point_tree = cKDTree(particles.pos)
		neighbors_list = point_tree.query_ball_point(particles.pos, r=particles.radius * search_radius)
		for i, neighbor_idxs in enumerate(neighbors_list):
			if i in neighbor_idxs:
				neighbor_idxs.remove(i) #remove self

		#every point has different number of neighbors, vectorize with padding
		neighbors_count = np.fromiter((len(row) for row in neighbors_list), dtype=np.int64)
		n_pad_width = np.max(neighbors_count)
		n_pad_idxs = np.full((len(neighbors_list), n_pad_width), -1, dtype=int) #(p_count,n_count) int
		for i, row in enumerate(neighbors_list):
			n_pad_idxs[i, :neighbors_count[i]] = row

		#invalid neighbors contain garbage, so make sure to clear invalid idxs
		n_valid_idxs = n_pad_idxs!=-1 #(p_count,n_count) bool

		return n_pad_idxs, n_valid_idxs, neighbors_count

	def updateParticlePos(self, particles, time_delta):
		"""float[][3] updateParticlePos(ParticleType particles, float time_delta)"""
		movable_idxs = ~particles.fixed
		delta_dist = particles.v * time_delta #dx = v * dt
		new_pos = particles.pos + delta_dist * movable_idxs[:,None]
		return new_pos

	#higher smoothing = bias toward prev_dt if dt increases
	def calcTimeStep(self, velocity_mag, prev_dt, max_move, smoothing, max_dt, min_dt):
		"""
		float calcTimeStep(
			float[] velocity_mag,
			float prev_dt,
			float max_move,
			float smoothing,
			float max_dt,
			float min_dt
		)
		"""
		max_v = np.max(velocity_mag)

		new_dt=max_dt
		if max_v != 0.0:
			new_dt = max_move / max_v #t=d/v #limit movement distance per 1.0 sim_dt

		if new_dt>prev_dt:
			new_dt = new_dt*(1.0-smoothing) + prev_dt*smoothing
		new_dt = max(min(new_dt,max_dt), min_dt)

		return new_dt

	def calcTotalEnergy(self, mass, velocity_mag):
		"""float calcTotalEnergy(float[] mass, float[] velocity_mag)"""
		kE = 0.5 * mass * velocity_mag
		total_energy = np.sum(kE,axis=0)
		return total_energy

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
		iterations = 256,
		record_frame_path = None,
		log_frequency = 64,
		max_r_change = 0.1, #max change in radius per 1 sim_dt
		anneal_steps = None,
		rand = None
	):
		"""
		PointList relaxCloud(
			PointList point_list,
			float approx_radius,
			int iterations,
			string record_frame_path,
			int log_frequency,
			float max_r_change,
			int anneal_steps,
			np.random.Generator rand
		)
		"""
		if len(point_list.points) < 2:
			return point_list #Can't do anything with 0 or 1 points

		if anneal_steps == None:
			anneal_steps = max(1, min(iterations//16, 16))
		relax_end_steps = 4*anneal_steps #Allow start and end relax longer 

		particles = ParticleType(point_list, approx_radius)
		if rand != None:
			#Add run to run variation
			particles.createVariance(rand, 0.01)

		if record_frame_path:
			record_frequency_dt = 1.0 #save frame every N sim_dt

			particle_frames = []
			frame = particles.pos.astype(np.float32)
			particle_frames.append(frame)
			last_record_time = 0.0

		#min,median,max radius of particles.radius
		#med,max are only for debug logging
		print_precision = 4
		min_r,med_r,max_r = [approx_radius]*3

		#time delta vars
		sim_smoothing = 0.5
		sim_max_dt = 0.5
		sim_min_dt = 0.01
		max_radius_move = 1.0 #how many min_r any particle can move in 1.0 sim_dt
		max_global_move = 0.25 #maximum units that particle can move, gamut is ~1x1x1 cube
		sim_dt = 0.1 #simulation time step
		sim_time_elapsed = 0.0

		#point_tree cache
		search_radius = 1.2
		last_tree_pos = [1e10]*3 #force first update
		tree_update_threshold = 0.5 #e.g. 0.5 means when any particle moves 0.5x of its radius, point_tree is updated  
		neighbors_count = None
		n_pad_idxs = None
		n_valid_idxs = None

		### Physics loop ###
		for tick in range(iterations):
			### Calc and sum forces ###
			all_forces=[]

			#Only update particle neighbors when any particle has moved <tree_update_threshold> of its radius
			move_in_radius = OkTools.vec3Length(particles.pos - last_tree_pos, axis=1) / (particles.radius + 1e-12)
			if np.max(move_in_radius) > tree_update_threshold:
				#Shuffle has very little effect as everything except overlapping points don't care about order
				#if rand != None:
				#	order = rand.permutation(len(particles))
				#	particles.shuffle(order)

				last_tree_pos = particles.pos.copy()

				n_pad_idxs, n_valid_idxs, neighbors_count = self.updateParticleNeighbors(particles, search_radius)


			## Neighbor forces
			#Neighbor distances can be re-calculated even with a stale tree
			#at worst the particles are beyond influence or barely touching points are ignored
			if np.any(n_valid_idxs):
				neighbors_dist_vec = particles.pos[:,None] - particles.pos[n_pad_idxs] #(point_count,neighbors_count,3)
				neighbors_dist_vec[~n_valid_idxs] = [0,0,0] #mark invalid

				neighbors_norm = np.zeros_like(neighbors_dist_vec) #init as invalid
				neighbors_dist = np.full(neighbors_dist_vec.shape[:2], np.inf)

				valid_n_dist_vec = neighbors_dist_vec[n_valid_idxs]
				neighbors_norm[n_valid_idxs], neighbors_dist[n_valid_idxs] = OkTools.vec3ArrayNorm( valid_n_dist_vec )

				#opposing norm if exactly on top of neighbor, order dependant
				zero_neighbor_dist = neighbors_dist==0 #if particle neighbor dists is 0
				zero_neighbor_dist[~n_valid_idxs] = False #prune padding
				inside_neighbor = np.any( zero_neighbor_dist, axis=1)
				if np.any( inside_neighbor ):
					inside_count = np.sum(zero_neighbor_dist)
					norm_offset = particles.pos[inside_neighbor][0]
					rand_norm = OkTools.sphereNormals(inside_count, norm_offset) #deterministic way deciding
					neighbors_norm[zero_neighbor_dist] = rand_norm

				#spring forces
				neighbor_force, neighbor_force_mag = self.calcSpringForce(particles.radius, particles.k, n_valid_idxs, neighbors_dist, neighbors_norm)

				particles.stress = neighbor_force_mag
				all_forces.append(neighbor_force)
				
				## Bounce fixed
				#if any, choose closest fixed neighbor of each particle and bounce of it
				fixed_neighbors = particles.fixed[n_pad_idxs] #(p_count,n_count) bool
				fixed_neighbors[~n_valid_idxs] = False

				fixed_neighbors &= (~particles.fixed)[:,None] #ignore fixed

				if np.any(fixed_neighbors):
					has_fixed_neighbor = np.any(fixed_neighbors,axis=1) #(particle_count) bool

					fixed_dist = np.where(fixed_neighbors, neighbors_dist, np.inf)
					closest_fixed = np.argmin(fixed_dist, axis=1)

					rows = np.where(has_fixed_neighbor)[0]
					cols = closest_fixed[rows]

					fixed_bounce_norm = neighbors_norm[rows, cols]
					
					particles.v[has_fixed_neighbor] = self.calcReflectVelocity(particles.v[has_fixed_neighbor], particles.COR[has_fixed_neighbor], fixed_bounce_norm)


			## Drag, opposing #Fd=0.5*p*A*Cd*v^2
			drag_factors = 0.5 * particles.rho * particles.A * particles.Cd
			drag_force = -1.0 * drag_factors[:,None] * particles.v_mag[:,None] * particles.v #approx (v_mag * v_norm)**2
			all_forces.append(drag_force)

			## Internal friction, opposing
			friction_force = particles.v * (-1.0*particles.mu)[:,None] #f=cf
			all_forces.append(friction_force)
			

			## Sum Fdt
			force_delta=np.sum(all_forces,axis=0)
			force_delta[particles.fixed] = [0,0,0] #ignore fixed
			acceleration = self.forceToAcceleration(particles.m, force_delta)
			particles.v = particles.v + acceleration * sim_dt
			particles.v_mag = OkTools.vec3Length(particles.v, axis=1)

			### Apply movement from velocity ###
			particles.pos = self.updateParticlePos(particles, sim_dt)

			## Gamut reflect and clip
			#lower bound prevents point being trapped behind darkest black
			particles.pos, clip_move = OkTools.clipToOklabGamut(particles.pos, lower_bound=OkTools.DARKEST_BLACK_LAB) 
	
			particles.clipped[:] = False
			if clip_move is not None:
				non_zero_move = np.any(clip_move, axis=1)
				particles.clipped = non_zero_move

				surface_norm, _ = OkTools.vec3ArrayNorm(clip_move[non_zero_move])
				particles.v[non_zero_move] = self.calcReflectVelocity(particles.v[non_zero_move], particles.COR[non_zero_move], surface_norm)

			## Adaptive radius scale
			if(
				iterations-tick >= relax_end_steps and #last trigger
				tick >= relax_end_steps and #first
				tick%anneal_steps==0 #periodic
			):
				particles.radius = self.calcParticleRadius(particles, n_pad_idxs, n_valid_idxs, neighbors_count, max_r_change, sim_dt)
				min_r = round(np.min(particles.radius),print_precision)

			## Update timestep
			sim_time_elapsed+=sim_dt
			sim_dt = self.calcTimeStep(
				particles.v_mag, 
				sim_dt, 
				max_move = min(max_global_move, max_radius_move * min_r),
				smoothing = sim_smoothing, 
				max_dt = sim_max_dt, 
				min_dt = sim_min_dt
			)
		

			## Logging
			if record_frame_path and (sim_time_elapsed - last_record_time >= record_frequency_dt):
				last_record_time+= record_frequency_dt #mitigate time overshooting by accumulation
				frame = particles.pos.astype(np.float32)
				particle_frames.append(frame)

			if log_frequency and (tick%log_frequency==0 or tick==iterations-1):
				total_energy=self.calcTotalEnergy(particles.m, particles.v_mag)
				out_str = "DT: " + str(round(sim_dt,print_precision)) + " Total energy["+str(round(sim_time_elapsed,print_precision))+"]:"
				out_str+= " "+str(total_energy)
				print(out_str)
			
				med_r = round(np.median(particles.radius),print_precision)
				max_r = round(np.max(particles.radius),print_precision)
				print("P: minr " + str(min_r) + " medr" + str(med_r) + " maxr" + str(max_r))
		
		if record_frame_path:
			if log_frequency:
				print("Writing particle_frames... ")
			npy_frames = np.array(particle_frames, dtype=np.float32)
			np.save(record_frame_path, npy_frames)

		#ensure order to match original point_list
		order = np.argsort(particles.pid)
		particles.shuffle(order)
		point_list.points["color"] = particles.pos

		if log_frequency:
			print("relaxCloud Done\n")
		return point_list