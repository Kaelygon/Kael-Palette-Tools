import math
from dataclasses import dataclass, fields
import numpy as np
from scipy.spatial import cKDTree
from numpy.typing import NDArray

from OkTools import *
from PointList import *

@dataclass
class ParticleType:
	pos: 		NDArray[[np.float64]*3]	= None
	fixed: 	NDArray[bool ]				= None
	v: 		NDArray[[np.float64]*3]	= None
	A: 		NDArray[float]				= None
	Cd: 		NDArray[float]				= None
	rho: 		NDArray[float]				= None
	m: 		NDArray[float]				= None
	k: 		NDArray[float]				= None
	COR: 		NDArray[float]				= None
	mu: 		NDArray[float]				= None
	radius: 	NDArray[float]				= None
	clipped: NDArray[bool ]				= None
	stress: 	NDArray[float]				= None

	def __getitem__(self, idxs):
		data = {}
		for f in fields(self):
			value = getattr(self, f.name)
			data[f.name] = value[idxs]

		return ParticleType(**data)

	def __len__(self):
		return len(self.pos)

class ParticleSim:
	"""Physics based spring/collision simulator to distribute points."""

	#Too far: attract, Too close: repel # f=-kx (single particle)
	def calcSpringForce(self, particle: ParticleType, neighbors_dist_sq: NDArray[[float]*3], neighbors_norm: NDArray[[float]*3]):
		beyond_radius = neighbors_dist_sq > (particle.radius**2)[:,None]
		spring_constant = np.zeros(beyond_radius.shape) + particle.k[:,None]
		spring_constant[beyond_radius] = 0.0 #attraction force (disabled)

		delta_x = np.sqrt(neighbors_dist_sq) - particle.radius[:,None]
		spring_mag = -1.0 * spring_constant * delta_x #f=-kx
		spring_force = neighbors_norm*spring_mag[:,:,None]
		spring_force = np.sum(spring_force,axis=1)
		return spring_force #(point_coint, 3)

	def moveToVelocity(self, move_delta: NDArray[[float]*3], time_delta: float):
		return move_delta / time_delta #v = dx/dt
	
	def velocityToForce(self, particles: ParticleType,velocity: NDArray[[float]*3], time_delta: float):
		return velocity * particles.m[:,None] / time_delta #f = dv*m/dt
	
	def forceToVelocity(self, particles: ParticleType,force: NDArray[[float]*3], time_delta: float):
		velocity = force * time_delta / particles.m[:,None] #dv = f*dt/m
		return velocity

	#force from distance moved in time step. f=m*((dx/dt)/dt)
	def moveToForce(self, particles: ParticleType,move_delta: NDArray[[float]*3], time_delta: float):
		velocity = self.moveToVelocity(move_delta, time_delta) #v = dx/dt
		force = self.velocityToForce(particles, velocity, time_delta)
		return force

	#returns reflected velocity  # f = - 2 * (vec . n) * n
	def calcReflectVelocity(self, particles: ParticleType,surface_norm: NDArray[[float]*3]):
		velocity = particles.v
		dot_v = np.sum(velocity*surface_norm, axis=1) #(vec . n)
		reflected_v = -1.0 * (particles.COR+1.0) * dot_v # - 2 * (vec . n)
		reflected_v = reflected_v[:,None] * surface_norm # - 2 * (vec . n) * n
		velocity = velocity + reflected_v
		return velocity

	def calcTotalEnergy(self, particles: ParticleType):
		kE = 0.5 * particles.m * np.linalg.norm(particles.v,axis=1)**2
		total_energy = np.sum(kE,axis=0)
		return total_energy

	def calcNeighborsForce(self,
		particles: ParticleType,
		n_pad_idxs: list[[int]], 
		n_valid_idxs: list[[int]],
		neighbors_dist_sq: NDArray[[float]*3],
		neighbors_norm: NDArray[[float]*3]
	):
		neighbor_force = np.zeros_like(particles.v)
		
		#random normal if exactly on top of neighbor
		zero_neighbor_dist = neighbors_dist_sq==0 #if particle neighbor dists is 0
		zero_neighbor_dist[~n_valid_idxs] = False #prune padding
		inside_neighbor = np.any( zero_neighbor_dist, axis=1)
		if np.any( inside_neighbor ):
			inside_count = np.sum(inside_neighbor)
			rand_norm = np.random.rand(np.sum(zero_neighbor_dist), 3) - 0.5
			neighbors_norm[zero_neighbor_dist] = rand_norm

		#spring forces
		spring_force = self.calcSpringForce(particles, neighbors_dist_sq, neighbors_norm)
		neighbor_force+= spring_force

		return neighbor_force

	#higher smoothing = bias toward prev_dt if dt increases
	def calcTimeStep(self, prev_dt: float, particles: ParticleType,unit_size: float, smoothing: float = 0.5, max_dt: float = 0.5):
		v_sq = np.sum(particles.v**2,axis=1)
		max_v_sq = np.max(v_sq)

		new_dt=max_dt
		if max_v_sq != 0.0:
			new_dt = unit_size**2 / max_v_sq #t=d/v #limit movement distance per tick

		if new_dt>prev_dt:
			new_dt = new_dt*(1.0-smoothing) + prev_dt*smoothing
		new_dt = max(min(new_dt,max_dt),1e-2)

		return new_dt

	#Scale radius by comparing neighbor states
	def calcParticleRadius(self, 
		particles: ParticleType,
		n_pad_idxs: NDArray[[int]], 
		n_valid_idxs: NDArray[[bool]], 
		neighbor_count: NDArray[int], 
		max_r_change: float=0.2, 
		time_delta: float=1.0
	):
		output_radii = np.zeros_like(particles.radius)
		particle_count = len(particles)

		scale_limit = particles.radius*max_r_change*time_delta #max radius change in time step

		none_valid = ~np.any(n_valid_idxs,axis=1) #all values are padding (invalid) = no neighbors
		no_neighbors = (neighbor_count==0) | none_valid

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

		n_stress = particles.stress[n_pad_idxs]
		n_radius = particles.radius[n_pad_idxs]
		n_stress[~n_valid_idxs] = np.nan
		n_radius[~n_valid_idxs] = np.nan

		avg_stress = np.zeros(particle_count)
		median_radius = particles.radius
		valid_rows = np.all(~np.isnan(n_stress), axis=1)
		avg_stress[valid_rows] 		= np.nanmedian(n_stress[valid_rows], axis=1)
		median_radius[valid_rows] 	= np.nanmedian(n_radius[valid_rows], axis=1)

		#Limit unstable and clipped particles radius growth
		is_unstable = particles.stress > 2.0*avg_stress
		unstable_or_clipping = is_unstable | particles.clipped
		if np.any(unstable_or_clipping):
			kissing_scale[unstable_or_clipping] = np.clip(kissing_scale[unstable_or_clipping], 0.0, 1.0)

		new_radius = median_radius*kissing_scale
		new_radius = np.clip(new_radius, particles.radius*0.5, particles.radius+scale_limit)

		has_neighbors = ~no_neighbors
		output_radii[has_neighbors] = new_radius[has_neighbors]
		output_radii = np.clip(output_radii,1e-20,2.0)
		return output_radii


	def createParticles(self, point_list: PointList, start_radius: float):
		p_count = point_list.length()
		particle_list = ParticleType()

		particle_list.pos		 = point_list.points["color"]
		particle_list.fixed	 = point_list.points["fixed"]
		particle_list.v		 = np.repeat([[0.0, 0.0, 0.0]], p_count, axis=0)
		particle_list.A		 = np.full(p_count ,math.pi * (0.1)**2 ) #m^2
		particle_list.Cd		 = np.full(p_count ,0.47 ) #sphere 
		particle_list.rho		 = np.full(p_count ,3.0 ) #kg/m^3 
		particle_list.m		 = np.full(p_count ,0.4 ) #kilograms 
		particle_list.k		 = np.full(p_count ,0.4 )
		particle_list.COR		 = np.full(p_count ,0.6 )
		particle_list.mu		 = np.full(p_count ,0.2 )
		particle_list.radius	 = np.full(p_count ,start_radius )
		particle_list.clipped = np.full(p_count ,False )
		particle_list.stress	 = np.full(p_count ,0.0 )

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
		if iterations == 0 or len(point_list.points) < 2:
			return point_list #Can't do anything with 0 or 1 points

		self.sim_dt = 0.1

		if anneal_steps == None:
			anneal_steps = max(1, min(iterations//16, 16))

		particles = self.createParticles(point_list, approx_radius)

		if record_frames:
			particle_frames=[]
			particle_frames.append(particles.pos.tolist())

		#min,median,max radius of particles.radius
		min_r,med_r,max_r = [approx_radius]*3

		### Physics loop ###
		for tick in range(iterations):
		
			## Apply movement from velocity
			movable_idxs = ~particles.fixed
			delta_dist = particles.v * self.sim_dt #dx = v * dt

			new_pos = particles.pos + delta_dist
			invalid_pos = np.any(np.isnan(new_pos), axis=1)

			particles.pos[movable_idxs] = new_pos[movable_idxs]
			particles.pos[invalid_pos] = [0.0]*3

			if record_frames:
				frame = particles.pos.tolist()
				particle_frames.append(frame)


			## Get neighbors of each point within its radius
			point_tree = cKDTree(particles.pos)
			neighbors_list = point_tree.query_ball_point(particles.pos, r=particles.radius*1.2)
			for i, neighbor_idxs in enumerate(neighbors_list):
				if i in neighbor_idxs:
					neighbor_idxs.remove(i) #remove self



			### Calc and sum forces ###
			all_forces=[]

			## Gamut reflect and clip
			particles.pos, clip_move = OkTools.clipToOklabGamut(particles.pos) # dx
	
			particles.clipped[:] = False
			if clip_move is not None:
				non_zero_move = np.any(clip_move, axis=1)
				particles.clipped = non_zero_move

				surface_norm = OkTools.vec3_array_norm(clip_move[non_zero_move])
				particles.v[non_zero_move] = self.calcReflectVelocity(particles[non_zero_move], surface_norm)


			## BOF padded neighbors
			#every point has different number of neighbors, vectorize with padding
			neighbors_count = np.fromiter((len(row) for row in neighbors_list), dtype=np.int64)
			n_pad_width = np.max(neighbors_count)
			n_pad_idxs = np.full((len(neighbors_list), n_pad_width), -1, dtype=int) #(p_count,n_count) int
			for i, row in enumerate(neighbors_list):
				n_pad_idxs[i, :len(row)] = row

			#invalid neighbors contain garbage, so make sure to clear invalid idxs
			n_valid_idxs = n_pad_idxs!=-1 #(p_count,n_count) bool


			## Adaptive radius scale
			if(
				iterations-tick >= 4*anneal_steps and #last trigger
				tick >= 4*anneal_steps and #first
				tick%anneal_steps==0 #periodic
			):
				particles.radius = self.calcParticleRadius(particles, n_pad_idxs, n_valid_idxs, neighbors_count, max_r_change, self.sim_dt)


			## Neighbor forces
			neighbor_force = np.zeros_like(particles.v)
			if np.any(n_valid_idxs):
				neighbors_dist_vec = particles.pos[:,None] - particles.pos[n_pad_idxs] #(point_count,neighbors_count,3)
				neighbors_dist_vec[~n_valid_idxs] = [0,0,0] #mark invalid

				neighbors_dist_sq = np.sum(neighbors_dist_vec**2,axis=2)
				neighbors_dist_sq[~n_valid_idxs] = 0 #mark invalid

				neighbors_norm = OkTools.vec3_array_norm(neighbors_dist_vec, axis=2)
				neighbors_norm[~n_valid_idxs] = [0,0,0] #mark invalid

				#jitter + spring
				neighbor_force = self.calcNeighborsForce(particles, n_pad_idxs, n_valid_idxs, neighbors_dist_sq, neighbors_norm)
				particles.stress = np.abs(np.sum(neighbor_force,axis=1))
				
				## Bounce fixed
				#if any, choose closest fixed neighbor of each particle and bounce of it
				fixed_neighbors = particles.fixed[n_pad_idxs] #(p_count,n_count) bool
				fixed_neighbors[~n_valid_idxs] = False

				fixed_neighbors &= (~particles.fixed)[:,None] #ignore fixed

				if np.any(fixed_neighbors):
					has_fixed_neighbor = np.any(fixed_neighbors,axis=1) #(particle_count) bool

					fixed_dist_sq = np.where(fixed_neighbors, neighbors_dist_sq, np.inf)
					closest_fixed = np.argmin(fixed_dist_sq, axis=1)

					rows = np.where(has_fixed_neighbor)[0]
					cols = closest_fixed[rows]

					fixed_bounce_norm = neighbors_norm[rows, cols]
					
					#TODO: could have some tangent bias or force that pushes a still point away
					particles.v[has_fixed_neighbor] = self.calcReflectVelocity(particles[has_fixed_neighbor], fixed_bounce_norm)
					
				all_forces.append(neighbor_force)

			## EOF padded neighbors


			## Drag, opposing #Fd=0.5*p*A*Cd*v^2
			drag_factors = 0.5 * particles.rho * particles.A * particles.Cd
			drag_force = -1.0 * drag_factors[:,None] * particles.v**2
			all_forces.append(drag_force)


			## Internal friction, opposing
			friction_force = particles.v * (-1.0*particles.mu)[:,None] #f=cf
			all_forces.append(friction_force)


			## Sum Fdt
			force_delta=np.sum(all_forces,axis=0)
			force_delta[particles.fixed] = [0,0,0] #ignore fixed
			delta_velocity = self.forceToVelocity(particles, force_delta, self.sim_dt)
			particles.v = particles.v + delta_velocity


			## Update timestep
			unit_size = min(0.25,min_r) * 0.5 #gamut dims is ~1x1x1
			self.sim_dt = self.calcTimeStep(self.sim_dt, particles, unit_size=unit_size, smoothing=0.5, max_dt = 0.5)
		

			## Logging
			if tick%log_frequency==0 or tick==iterations-1:
				total_energy=self.calcTotalEnergy(particles)
				out_str = "DT: " + str(round(self.sim_dt,4)) + " Total energy["+str(tick)+"]:"
				out_str+= " "+str(total_energy)
				print(out_str)
			
				min_r = round(np.min(particles.radius),4)
				med_r = round(np.median(particles.radius),4)
				max_r = round(np.max(particles.radius),4)
				print("P: minr " + str(min_r) + " medr" + str(med_r) + " maxr" + str(max_r))
				if abs(total_energy) < min_energy:
					break
		
		if record_frames:
			with open(record_frames, "w") as f:
				f.write("oklab_frame_list = ")
				f.write(str(particle_frames))
	

		print("relaxCloud Done\n")
		point_list.points["color"] = particles.pos
		return point_list