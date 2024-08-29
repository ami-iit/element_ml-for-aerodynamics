import numpy as np
import scipy as sp
import scipy.optimize as sp_opt

class Optimizer:
    def __init__(self, robot):
        self.robot = robot
        self.u_des = np.zeros(robot.nDOF)
        
    def skew(self, v):
        S = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        return S
        
    def compute_self_collision_centers(self, w_H_i, centers_b):
        w_R_i = w_H_i[0:3,0:3]
        w_o_i = w_H_i[0:3,3]
        centers_w = np.zeros(centers_b.shape)
        for i in range(centers_b.shape[0]):
            centers_w[i,:] = w_o_i + np.dot(w_R_i, centers_b[i,:])
        return centers_w
    
    def compute_self_collision_gradient(self, w_H_i, w_H_j, c_i, c_j, b_c_i, b_c_j, J_i, J_j):
        w_R_i = w_H_i[0:3,0:3]
        w_R_j = w_H_j[0:3,0:3]
        Ja_i = J_i[:,6:]
        Ja_j = J_j[:,6:]
        Jc_i = Ja_i[0:3,:] - self.skew(np.dot(w_R_i,b_c_i)) @ Ja_i[3:6,:]
        Jc_j = Ja_j[0:3,:] - self.skew(np.dot(w_R_j,b_c_j)) @ Ja_j[3:6,:]
        gradient = 2*np.dot((c_i-c_j).T, (Jc_i-Jc_j))
        return gradient
    
    def check_collision_between_two_links(self, w_H_i, w_H_j, centers_i, centers_j, b_centers_i, b_centers_j, radiuses_i, radiuses_j, J_i, J_j):
        c_coll_ij    = np.empty((0,))
        grad_coll_ij = np.empty((0,))
        scaling      = 1.0
        is_colliding  = False
        for i in range(centers_i.shape[0]):
            b_c_i = b_centers_i[i,:]
            c_i = centers_i[i,:]
            r_i = radiuses_i[i]
            for j in range(centers_j.shape[0]):
                b_c_j = b_centers_j[j,:]
                c_j = centers_j[j,:]
                r_j = radiuses_j[j]
                if np.linalg.norm(c_i-c_j)**2 <= scaling*(r_i+r_j)**2:
                    is_colliding = True
                c_coll_ij = np.append(c_coll_ij, np.dot(c_i-c_j, c_i-c_j) - scaling*(r_i+r_j)**2)
                grad_coll_ij = np.append(grad_coll_ij, self.compute_self_collision_gradient(w_H_i, w_H_j, c_i, c_j, b_c_i, b_c_j, J_i, J_j))
        return c_coll_ij, grad_coll_ij, is_colliding

    def compute_self_collisions_and_gradient(self):
        c_collision = np.empty((0,))
        grad_c_collision = np.empty((0,))
        for i, frame_i in enumerate(self.robot.collisions_frame_list):
            J_i = self.robot.get_free_free_floating_jacobian(frame_i)
            w_H_i = self.robot.kinDyn.getWorldTransform(frame_i).asHomogeneousTransform().toNumPy()
            b_centers_i = self.robot.collisions[frame_i]["Centers"]
            centers_i = self.compute_self_collision_centers(w_H_i, b_centers_i)
            radiuses_i = self.robot.collisions[frame_i]["Radiuses"]
            for j, frame_j in enumerate(self.robot.collisions_frame_list):
                if i > j and not ( (i == 12 and j == 3) or (i == 12 and j == 4) ):
                    J_j = self.robot.get_free_free_floating_jacobian(frame_j)
                    w_H_j = self.robot.kinDyn.getWorldTransform(frame_j).asHomogeneousTransform().toNumPy()
                    b_centers_j = self.robot.collisions[frame_j]["Centers"]
                    centers_j = self.compute_self_collision_centers(w_H_j, b_centers_j)
                    radiuses_j = self.robot.collisions[frame_j]["Radiuses"]
                    c_ij, grad_ij, is_colliding = self.check_collision_between_two_links(w_H_i, w_H_j, centers_i, centers_j, b_centers_i, b_centers_j, radiuses_i, radiuses_j, J_i, J_j)
                    c_collision = np.append(c_collision, c_ij)
                    grad_c_collision = np.append(grad_c_collision, grad_ij)
        grad_c_collision = grad_c_collision.reshape((len(c_collision), self.robot.nDOF))
        return c_collision, grad_c_collision

    def compute_nonlinear_constraints(self, u):
        self.robot.set_state(0, 0, u)
        ceq = np.empty((0,))
        grad_ceq = np.empty((0,))
        c_collision, grad_c_collision = self.compute_self_collisions_and_gradient()
        c = c_collision
        grad_c = grad_c_collision
        return c, ceq, grad_c, grad_ceq
    
    def compute_cost_function(self, u):
        weights_matrix = np.eye(self.robot.nDOF)
        cost = (u-self.u_des).T @ weights_matrix @ (u-self.u_des)
        cost_gradient = 2 * (u-self.u_des).T @ weights_matrix
        return cost, cost_gradient
        
    def solve(self, u_desired):
        self.u_des = u_desired
        u_init = u_desired
        bounds = sp_opt.Bounds(self.robot.joint_limits[:,0]*np.pi/180, self.robot.joint_limits[:,1]*np.pi/180)
        cost_function = lambda u: self.compute_cost_function(u)[0]
        cost_function_gradient = lambda u: self.compute_cost_function(u)[1]
        inequality_constraint = lambda u: self.compute_nonlinear_constraints(u)[0]
        inequality_constraint_gradient = lambda u: self.compute_nonlinear_constraints(u)[2]
        if len(np.where(inequality_constraint(u_init)<=0)[0]) > 0:
            result = sp_opt.minimize(
                fun=cost_function,
                x0=u_init,
                jac=cost_function_gradient,
                method='trust-constr',
                bounds=bounds,
                constraints={
                    'type':'ineq',
                    'fun':inequality_constraint,
                    'jac':inequality_constraint_gradient,
                    },
                options={
                'maxiter': 7000,
                'gtol': 1e-6,
                'xtol': 1e-2,
                'initial_tr_radius':0.001,
                'disp': True,
                'verbose': 0,
                    },
                )
            optimized = True
            return optimized,result.x
        else:
            optimized = False
            return optimized, u_init
        