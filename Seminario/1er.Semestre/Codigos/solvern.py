
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Create classes for defining parts of the boundaries and the interior
# of the domain
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= epsilon*abs(sin(x[0]/epsilon)) + tol

# Initialize sub-domain instances
left = Left()
top = Top()
right = Right()
bottom = Bottom()
obstacle = Obstacle()

epsilon = 1


# Define mesh
mesh = RectangleMesh(Point(0,0),Point(1,1),8,8)

# Define tolerance
tol = 1E-14

# Initialize mesh function for interior domains
domains = MeshFunction('size_t',mesh,mesh.topology().dim())
domains.set_all(0)
obstacle.mark(domains, 1)

# Initialize mesh function for boundary domains
boundaries = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)

# Define input data
a0 = Constant(1)
a1 = Constant(1)
g_L = Constant(1*0.63662)
g_R = Constant(0)
f0 = Constant(0)
f1 = Constant(0)

dx = Measure("dx")[domains]
ds = Measure("ds")[boundaries]



p = 1
V_ele = FiniteElement("CG", mesh.ufl_cell(), p) # probably one needs to set dim=3?
R_ele = FiniteElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([V_ele, R_ele]))

# Define variational problem
(u, c) = TrialFunction(W)
(v, d) = TestFunctions(W)

F = (inner(a0*grad(u), grad(v))*dx(0) + inner(a1*grad(u), grad(v))*dx(1)
     + u*v*dx(0) + u*v*dx(1)
     + c*v*dx(0) + u*d*dx(0)
     + c*v*dx(1) + u*d*dx(1)
     - g_L*v*ds(4) - g_R*v*ds(3)
     - f0*v*dx(0) - f1*v*dx(1))

a, L = lhs(F), rhs(F)

# Compute solution
w = Function(W)
solve(a == L, w)
(u, c) = w.split()

# Plot solution and gradient
plot(u, title="u")

plt.show()