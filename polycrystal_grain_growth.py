"""generate the evoulution process of polycrystal, 
   you can see the stucture solidify from liquid to solid state containing multiple grains, 
   and the subsequent grain growth, such as 
   big grain eating small grains, curve boundaries being straightened, and triple-junctions forming
   
   using multi-phase field model to describe polycrystal, 
   applying Steinbach's generalization for Ginzburg-Landau equation in multi-phase problems,
   for physical interpretation of this model, you can refer to Physica D 134 (1999) 385-393"""
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt 
from matplotlib import cm
import os


@ti.data_oriented
class Polycrystal():

    def __init__(self) -> None:
        self.phase_nums = 16
        self.nx = 128; self.ny = 128

        self.phi = ti.Vector.field(self.phase_nums, ti.f64, shape=(self.nx, self.ny))
        self.phi_old = ti.Vector.field(self.phase_nums, ti.f64, shape=(self.nx, self.ny))
        self.phi_rate = ti.Vector.field(self.phase_nums, dtype=ti.f64, shape=(self.nx, self.ny, 4))
        self.laplacian = ti.Vector.field(self.phase_nums, ti.f64, shape=(self.nx, self.ny))

        self.dt = 0.015  # maximum dt is 0.018 before going unstable
        self.dx = 0.1; self.dy = 0.1
        self.kappa = 4.  # 4.  # gradient coefficient
        self.u = 64.  # energy barrier, chemical energy coefficient
        self.u_3phase_penalty = 3. * self.u  # energy penalty for 3 phases coexistence
        self.fluctuation = 0.01
        self.mobility = 0.02

        ### variables for visualization
        self.phaseID = ti.field(ti.f64, shape=(self.nx, self.ny))
        self.grain_boundary = ti.field(ti.f64, shape=(self.nx, self.ny))

        ### parameters for RK4 method (use Runge-Kutta method for time integration)
        self.dtRatio_rk4 = ti.field(ti.f64, shape=(4)); self.dtRatio_rk4.from_numpy(np.array([0., 0.5, 0.5, 1.]))
        self.weights_rk4 = ti.field(ti.f64, shape=(4)); self.weights_rk4.from_numpy(np.array([1./6., 1./3., 1./3., 1./6.]))


    @ti.kernel
    def initialize(self, ):
        for I in ti.grouped(self.phi):
            for p in range(self.phi[I].n):
                self.phi[I][p] = 1. / self.phase_nums
            ### set some initial perturbation for phi
            for p in range(self.phi[I].n):
                self.phi[I][p] = self.phi[I][p] \
                    - self.fluctuation + 2.*self.fluctuation * ti.random(ti.f64)
            ### normalization
            self.phi[I] /= self.phi[I].sum()
        for I in ti.grouped(self.phi):
            self.phi_old[I] = self.phi[I]
    

    @ti.func
    def neighbor_index(self, i, j):
        """ use periodic boundary condition to get neighbor index"""
        im = i - 1 if i - 1 >= 0 else self.nx - 1
        jm = j - 1 if j - 1 >= 0 else self.ny - 1
        ip = i + 1 if i + 1 < self.nx else 0
        jp = j + 1 if j + 1 < self.ny else 0
        return im, jm, ip, jp


    @ti.kernel
    def rk4_intermediate_update(self, rk_loop: int):
        phi, phi_old, phi_rate, dt = ti.static(self.phi, self.phi_old, self.phi_rate, self.dt)
        for I in ti.grouped(phi):
            phi[I] = phi_old[I] + dt * self.dtRatio_rk4[rk_loop] * phi_rate[I, rk_loop - 1]
            ### normalization
            phi[I] /= phi[I].sum()
    

    @ti.kernel
    def rk4_total_update(self, ):
        """the final step in RK4 (Runge-Kutta) process"""
        phi, phi_old, phi_rate, dt = ti.static(self.phi, self.phi_old, self.phi_rate, self.dt)
        for I in ti.grouped(phi):
            for rk_loop in range(4):
                phi_old[I] = phi_old[I] + self.weights_rk4[rk_loop] * dt * phi_rate[I, rk_loop]
        for I in ti.grouped(phi):
            phi[I] = phi_old[I]


    @ti.kernel
    def get_rate(self, rk_loop: int):  # advance a time step
        phi, laplacian, dx, mobility = ti.static(
            self.phi, self.laplacian, self.dx, self.mobility)
        
        ### compute the laplacian
        for i, j in phi:
            im, jm, ip, jp = self.neighbor_index(i, j)
            laplacian[i, j] = (  # laplacian of phi
                2 * (phi[im, j] + phi[i, jm] + phi[ip, j] + phi[i, jp]) 
                + (phi[im, jm] + phi[im, jp] + phi[ip, jm] + phi[ip, jp]) 
                - 12 * phi[i, j]
            ) / (3. * dx * dx)

        ### compute evolution and advance a time step
        for I in ti.grouped(phi):
            forces = ti.Vector([0. for _ in range(phi[I].n)])  # forces for different phases
            for p in range(phi[I].n):

                ### penalty force
                penalty_force = 0.
                for k in range(phi[I].n):
                    for l in range(k+1, phi.n):
                        if k != p and l != p:
                            penalty_force = penalty_force + \
                                ti.abs(phi[I][k] * phi[I][l])
                penalty_force *= ti.math.sign(phi[I][p])
                penalty_force = -self.u_3phase_penalty * penalty_force

                ### chemical energy force
                chemical_force = 0.
                for k in range(phi[I].n):
                    if k != p:
                        chemical_force = chemical_force + ti.abs(phi[I][k])
                chemical_force *= ti.math.sign(phi[I][p])
                chemical_force = -self.u * chemical_force

                ### compute forces
                forces[p] = chemical_force + \
                            self.kappa * laplacian[I][p] + \
                            penalty_force
                
            ### update phi by Steinbach's equation
            for p in range(phi[I].n):
                effective_force = 0.
                for k in range(phi[I].n):
                    effective_force = effective_force + forces[p] - forces[k]
                effective_force /= phi[I].n 
                self.phi_rate[I, rk_loop][p] = effective_force * mobility
        

    def advance(self, ):  # advance a time step
        self.get_rate(rk_loop=0)
        for rk_loop in range(1, 4):
            self.rk4_intermediate_update(rk_loop)
            self.get_rate(rk_loop)
        self.rk4_total_update()
    

    @ti.kernel
    def update_phaseID(self, ):
        """for visualization of different phases at different position"""
        for I in ti.grouped(self.phi):
            id = -1; maxVal = -1.
            for p in range(self.phi[I].n):
                if self.phi[I][p] > maxVal:
                    id = p; maxVal = self.phi[I][p]
            self.phaseID[I] = id
    

    @ti.kernel
    def update_grain_boundary(self, ):
        """for visualization, grain boundary ≈ 1; grain's internal ≈ 0"""
        for I in ti.grouped(self.phi):
            self.grain_boundary[I] = 1. - self.phi[I].max()
    

if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    
    write_images = True
    if write_images:
        path1 = "./tests/polycrystal/"; path2 = "./tests/grain_boundary/"
        if not os.path.exists(path1):
            os.makedirs(path1)
        if not os.path.exists(path2):
            os.makedirs(path2)
    
    polycrys = Polycrystal()
    polycrys.initialize()      
    
    for time_frames in range(16000):  # 20000
        
        polycrys.advance()
        
        if (time_frames % 64 == 0):
            print('time_frames={}'.format(time_frames))
            polycrys.update_grain_boundary()
            polycrys.update_phaseID()
            grain_boundary = polycrys.grain_boundary.to_numpy()
            phaseID = polycrys.phaseID.to_numpy()

            # ==========plot the grain boundary================
            plt.figure(2); plt.clf()    
            plt.pcolor(grain_boundary, cmap=cm.jet)
            plt.title("grain boundary", size=16)
            plt.draw(); plt.pause(0.001) 
            if write_images and time_frames % 128 == 0:
                plt.savefig(path2 + "time_{:.4f}s.png".format(time_frames * polycrys.dt))

            # ==========plot different grains(phases)==========
            plt.figure(3); plt.clf()
            plt.pcolor(phaseID, cmap=cm.jet, vmin=0, vmax=polycrys.phase_nums-1)
            plt.title("grains morphology", size=16)
            plt.draw(); plt.pause(0.001) 
            if write_images and time_frames % 128 == 0:
                plt.savefig(path1 + "time_{:.4f}s.png".format(time_frames * polycrys.dt))
