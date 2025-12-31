

import math
from dataclasses import dataclass, field

from RorrLcg import *
from KaelColor import *
from float3 import *
from PointGrid import *

class ParticleSim:
	"""Physics based spring/collision simulator to distribute points."""
	@dataclass
	class Particle:
		ref: KaelColor #simulated reference
		id: int #identifier
		fixed: bool #is immovable?
		v: [float]*3 #velocity
		A: float #Drag reference area
		Cd: float #Coefficient of drag
		rho: float #Drag (rho) fluid density
		m: float #Mass
		k: float #sprint constant
		COR: float #Coefficient of restitution
		mu: float #coefficient of friction
		radius: float #point radius
		clipped: bool #was near gamut boundary
		stress: float #spring crushing froce
	
	def __init__(self,
		_rand: RorrLCG,
		_point_grid: PointGrid
	):
		self.sim_dt = 0.1
		self.rand = _rand
		self.point_grid = _point_grid

	#Too far: attract, Too close: repel # f=-kx
	def calcSpringForce(self, p0: Particle, neighbor: NeighborList, neighbor_norm: [float]*3, spring_constant: float = 1.0):
		xd = math.sqrt(neighbor.dist_sq) - p0.radius
		spring_mag = -1.0 * spring_constant * xd #f=-kx
		spring_force = mul_vec3(neighbor_norm,[spring_mag]*3)

		return spring_force

	def moveToVelocity(self, p0: Particle, move_delta: [float]*3, time_delta: float):
		return div_vec3(move_delta, [time_delta]*3) #v = dx/dt
	
	def velocityToForce(self, p0: Particle, velocity: [float]*3, time_delta: float):
		return mul_vec3([p0.m]*3, div_vec3(velocity, [time_delta]*3) ) #
	
	def forceToVelocity(self, p0: Particle, force: [float]*3, time_delta: float):
		return div_vec3(mul_vec3(force,[time_delta]*3), [p0.m]*3) #dv = f*dt/m

	#force from distance moved in time step. f=m*((dx/dt)/dt)
	def moveToForce(self, p0:Particle, move_delta: [float]*3, time_delta: float):
		velocity = self.moveToVelocity(p0, move_delta, time_delta) #v = dx/dt
		return self.velocityToForce(p0, velocity, time_delta)

	#returns the force needed to reflect p0 velocity  # f = - 2 * (vec . n) * n
	def calcReflectForce(self, p0: Particle, surface_norm: [float]*3, time_delta: float):
		dot_v = dot_vec3(p0.v, surface_norm) #(vec . n)
		reflected_v = -1.0 * (p0.COR+1.0) * dot_v # - 2 * (vec . n)
		reflected_v = mul_vec3([reflected_v]*3, surface_norm) # - 2 * (vec . n) * n
		p0.v = add_vec3(p0.v, reflected_v)

	def calcTotalEnergy(self, p_list: list[float]):
		total_energy=0.0
		for particle in self.particles.values():
			kE = 0.5 * particle.m * length_vec3(particle.v)**2
			total_energy = total_energy + kE
		return total_energy

	#higher smoothing = bias toward prev_dt if dt increases
	def calcTimeStep(self, prev_dt: float, particles: dict, unit_size: float, scale: float = 0.5, smoothing: float = 0.5, max_dt: float = 0.5):
		max_v = 0.0
		velocity_list=[]
		for particle in particles.values():
			velocity_list.append(length_vec3(particle.v))
		max_v=max(velocity_list)

		new_dt=max_dt
		if max_v != 0.0:
			new_dt = scale * unit_size / max_v

		if new_dt>prev_dt:
			new_dt = new_dt*(1.0-smoothing) + prev_dt*smoothing
		new_dt = max(min(new_dt,max_dt),1e-4)

		return new_dt

	#update particle.radius adaptively
	def calcParticleRadius(self, particle: Particle, neighborhood: NeighborList, max_r_change: float=0.2, time_delta: float=1.0):
		radius = particle.radius
		scale_limit = radius*max_r_change*time_delta #max radius change in time step
		approx_speed = compSum_vec3(particle.v)
		almost_stopped = approx_speed < (0.1*radius/time_delta)
		if len(neighborhood.array)==0:
			if almost_stopped: #moving ~0.1x/dt, and no neighbors
				return radius + scale_limit
			return radius
			
		kissing_target=11 - particle.clipped*6
		kissing_number=0
		radius_list = []
		avg_stress = particle.stress
		for neighbor in neighborhood.array:
			if neighbor.dist_sq < (radius*1.1)**2:
				kissing_number+=1
			p1 = self.particles[neighbor.point.id]
			radius_list.append(p1.radius)
			avg_stress+=p1.stress

		median_radius = math_median(radius_list)
		kissing_scale = (kissing_target/(kissing_number+1))**(1/3)

		avg_stress = avg_stress/(len(neighborhood.array)+1)
		is_unstable = particle.stress > 2.0*avg_stress and almost_stopped
		if is_unstable or particle.clipped:
			kissing_scale = min(kissing_scale,1.0)

		new_radius = median_radius*kissing_scale
		new_radius = math_clip(new_radius, radius*0.5, radius+scale_limit)
		return new_radius

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
		iterations = 64,
		approx_radius = None,
		min_energy = 0.0,
		record_frames = None,
		log_frequency = 32,
		max_r_change = 0.1, #max change in radius per 1 sim_dt
		anneal_steps = None
	):
		if iterations == 0 or len(self.point_grid.cloud) == 0:
			return

		if record_frames:
			grid_frames=[]
			grid_frames.append(self.point_grid.cloudPosSnapshot())

		if approx_radius == None:
			approx_radius = self.point_grid.cell_size
		min_r,med_r,max_r = [approx_radius]*3

		if anneal_steps == None:
			anneal_steps = min(iterations//8, 64)

		self.particles = { p.id: None for p in self.point_grid.cloud }
		for p in self.point_grid.cloud:
			#softball in mud
			point = self.Particle(
				ref=p,
				id=p.id,
        		fixed=p.fixed,
				v=[0.0]*3,
          	A=math.pi * (0.1)**2, #m^2
           	Cd=0.47, #sphere
            rho=3.0, #kg/l
            m=0.4, #kilograms
            k=0.4,
            COR=0.6,
				mu=0.2,
				radius=approx_radius,
				clipped=False,
				stress=0.0
         )
			self.particles[p.id] = point

		for tick in range(iterations):
			self.particles = self.rand.shuffleDict(self.particles)

			self.sim_dt = self.calcTimeStep(self.sim_dt, self.particles, unit_size=min_r, scale=0.1, smoothing=0.5, max_dt = 10.0)
		
			#Calculate velocity synchronously
			for particle in self.particles.values():
				if particle.fixed:
					continue
				
				delta_dist = mul_vec3(particle.v, [self.sim_dt]*3) #dx = v * dt
				new_col = add_vec3(particle.ref.col, delta_dist)

				if valid_vec3(new_col):
					self.point_grid.setCol(particle.ref,new_col)
				else:
					particle.v=[0.0]*3

			#Add forces
			for particle in self.particles.values():
				neighborhood = self.point_grid.findNeighbors(particle.ref, particle.radius, particle.radius*0.2)
				if iterations-tick > anneal_steps and tick > anneal_steps:
					particle.radius = self.calcParticleRadius(particle, neighborhood, max_r_change, self.sim_dt)

				if particle.fixed:
					continue

				all_forces=[]

				#gamut reflect
				clip_move = particle.ref.clipToOklabGamut() # dx
				particle.clipped = clip_move !=[0.0]*3
				if particle.clipped and valid_vec3(clip_move):
					surface_norm = norm_vec3(clip_move)
					self.calcReflectForce(particle, surface_norm, self.sim_dt)
	
				#neighbor forces
				neighbor_force = [0.0,0.0,0.0]
				for neighbor in neighborhood.array:
					spring_mul = 1.0
					if neighbor.dist_sq > particle.radius**2:
						spring_mul = 0.0 #attraction force

					neighbor_particle = self.particles[neighbor.point.id]

					if neighbor.dist_sq == 0.0: #push out particles inside each other
						rand_norm = mul_vec3(self.rand.vec3(),[particle.radius]*3)
						push_move = mul_vec3([ particle.radius*self.sim_dt ]*3,rand_norm)
						push_force = self.moveToForce(particle, push_move, self.sim_dt)
						neighbor_force = add_vec3(neighbor_force, push_force)
						continue

					#spring influence
					neighbor_norm = norm_vec3(neighbor.dist_vec)
					spring_force = self.calcSpringForce(particle, neighbor, neighbor_norm, spring_mul*particle.k)
					neighbor_force = add_vec3(neighbor_force, spring_force)

					if neighbor_particle.fixed: #bounce if inside fixed
						surface_norm = norm_vec3(neighbor.dist_vec)
						self.calcReflectForce(particle, surface_norm, self.sim_dt)
	
				particle.stress = abs(compSum_vec3(neighbor_force))
				all_forces.append(neighbor_force)

				#drag, opposing #Fd=0.5*p*A*Cd*v^2
				drag_force = mul_vec3( [-1.0 * 0.5 * particle.rho * particle.A * particle.Cd]*3, mul_vec3(particle.v,particle.v) )
				all_forces.append(drag_force)

				#internal friction, opposing
				friction_force = mul_vec3( particle.v, [-1.0*particle.mu]*3 ) #f=cf
				all_forces.append(friction_force)

				#sum Fdt
				force_delta=[0.0]*3
				for force in all_forces:
					if valid_vec3(force):
						force_delta = add_vec3(force_delta,force)

				delta_velocity = self.forceToVelocity(particle, force_delta, self.sim_dt)
				particle.v = add_vec3(particle.v, delta_velocity)
		
			#logging
			if tick%log_frequency==0 or tick==iterations-1:
				total_energy=self.calcTotalEnergy(self.particles)
				prev_energy=total_energy
				out_str = "DT: " + str(round(self.sim_dt,4)) + " Total force["+str(tick)+"]:"
				out_str+= " "+str(total_energy)
				print(out_str)
			
				p_radius_list=[]
				p_velocity_list=[]
				for p in self.particles.values():
					p_radius_list.append(p.radius)
					p_velocity_list.append(p.v)
				min_r = round(min(p_radius_list),4)
				med_r = round(math_median(p_radius_list),4)
				max_r = round(max(p_radius_list),4)
				print("P: minr " + str(min_r) + " medr" + str(med_r) + " maxr" + str(max_r))
				if abs(total_energy) < min_energy:
					break

			if record_frames:
				grid_frames.append(self.point_grid.cloudPosSnapshot())
		
		if record_frames:
			with open(record_frames, "w") as f:
				f.write("oklab_frame_list = ")
				f.write(str(grid_frames))
	

		print("relaxCloud Done\n")
		return self.point_grid