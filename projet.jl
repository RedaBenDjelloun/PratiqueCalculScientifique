using PyPlot
using LinearAlgebra
using Plots
using DifferentialEquations
pygui(true)

#################### Parameters ####################

# user parameters
p=2.0
q=0.05
d=0.01
T=10.0 # duration of the simulation
Δx = 0.1
Δt = 0.005

# initial state u0 = u_eq + A * [sin(k1*x),cos(k2*x)]
A = 1
k1 = 1
k2 = 2

# computed parameters
u_eq = [p/((p+q)^2),p+q]
D =[1. 0.;0. d]
nb_x = floor(Int,2pi/Δx)
nb_t = floor(Int,T/Δt)
Δx = 2pi/nb_x
Δt = T/nb_t
u0 = zeros(nb_x,2)
for i in 1:nb_x
    local x = i*Δx
    u0[i,:] = u_eq+A*[sin(k1*x),cos(k2*x)]
end

# M is the discrete operator for second derivative
M = zeros(nb_x,nb_x)
for i in 1:nb_x-1
    M[i,i]=-2
    M[i,i+1]=1
    M[i+1,i]=1
end
M[nb_x,nb_x]=-2
M[nb_x,1]=1
M[1,nb_x]=1
M = M/Δx^2

# test of M
begin
    x = [sin(k*Δx) for k in 1:nb_x]
    @assert(norm(M*x+x)<1e-2)
end

# M2 is the discrete operator for 2nd derivative when u is in the vectorial form
M2 = [M zeros(nb_x,nb_x); zeros(nb_x,nb_x) M]

D2 = zeros(2*nb_x,2*nb_x)
for i in 1:nb_x
    D2[i,i]=1
    D2[i+nb_x,i+nb_x]=d
end


#################### u representation ####################
# For the user u is always in the form u[i,c,t] or u[i,c] (form of reference)
# In the solvers u is in the form [u[:,1,:] u[:,2,:]] or [u[:,1] u[:,2]]

# reshape a matrix into a vector column by column
vect(u) = u[:]

#test de vect
begin
    x = rand(2,2)
    @assert(vect(x)==[x[1,1],x[2,1],x[1,2],x[2,2]])
end

# reshape a vector into a matrix with 2 columns
# input [u1, u2, ... u2N]
# output [u1 uN+1; u2 uN+2; ... ; uN u2N]
function devect(u)
    new_size = size(u)[1]÷2
    devect_u = zeros(new_size,2)
    for i in 1:new_size
        devect_u[i,:] = [u[i], u[i+new_size]]
    end
    devect_u
end

# test de devect
begin
    x = rand(4)
    @assert(devect(x)==[x[1] x[3]; x[2] x[4]])
end

# input : u of the form [u[:,1] u[:,2]]
# output : u of the form u[i,c,t]
function deflatten(u)
deflatten_u = zeros(size(u)[1]÷2,2,size(u)[2])
    for t in 1:nb_t
        deflatten_u[:,:,t] = devect(u[:,t])
    end
    deflatten_u
end

#################### f and its derivatives ####################

f(u) = [p-u[1]*u[2]^2;q-u[2]+u[1]*u[2]^2]

# Jacobienne de f
Jf(u) =[-u[2]^2 -2*u[1]*u[2] ;
         u[2]^2 -1+2*u[1]*u[2] ]

# test de la Jacobienne
begin
    u = rand(2)
    h = rand(2)
    eps = 1e-8
    @assert( norm( (f(u+eps*h)-f(u))/eps - Jf(u)*h ) < 1e-5 )
end

# input : u of the form u[i,c]
# output : [f(u[i,:])]
function F(u)
    Fu = zero(u)
    for i in 1:size(u)[1]
        Fu[i,:] = f(u[i,:])
    end
    Fu
end

# input : u of the form [u[:,1], u[:,2]]
# output : [f(u[i,:])]
function F2(u)
    F2u = zero(u)
    x_size = size(u)[1]÷2 # = nb_x in the simulation
    for i in 1:x_size
        fu = f([u[i],u[i+x_size]])
        F2u[i] = fu[1]
        F2u[x_size+i] =fu[2]
    end
    return F2u
end

#test of F2
begin
    u = rand(50,2)
    Fu1 = F(u)
    Fu2 = devect(F2(vect(u)))
    @assert(norm(Fu1-Fu2)<1e-10)
end

# input : u of the form [u[:,1], u[:,2]]
# output : Jacobian of F2 at u
function JF(u)
    x_size = size(u)[1]÷2 # = nb_x in the simulation
    JFu = zeros(2*x_size,2*x_size)
    for i in 1:x_size
        df = Jf([u[i];u[x_size+i]])
        JFu[i,i] = df[1,1]
        JFu[i,x_size+i] = df[1,2]
        JFu[x_size+i,i] = df[2,1]
        JFu[x_size+i,x_size+i] = df[2,2]
    end
    return JFu
end

# test of the jacobian
begin
    u = rand()
end

#################### Euler Explicite ####################

function generic_explicit_euler(u0, f)
    u = zeros(size(u0)[1],nb_t)

    # initialization
    u[:,1] = u0;

    # integration with implicit euler method
    for t in 1:(nb_t-1)
        u[:,t+1] = u[:,t] + Δt*f(u[:,t])
    end
    return u
end

# Intégration de l'edp avec la méthode d'Euler explicite
function explicit_euler(u0)
    u0 = vect(u0)
    f(u) = D2*M2*u + F2(u)
    sol = generic_explicit_euler(u0,f)
    return deflatten(sol)
end

#################### Euler implicite ####################

function newton(f,Jf,x0,u_t,eps,max_iter)
    x=x0
    i=0
    while(norm(f(x)-u_t)>eps && i<max_iter)
        x = x - inv(Jf(x))*(f(x)-u_t)
        i+=1
    end
    return x
end

function generic_implicit_euler(u0,f,Jf)
    u = zeros(size(u0)[1],nb_t)

    # initialization
    u[:,1] = u0;

    # integration
    for t in 1:(nb_t-1)
        u[:,t+1] = newton(f,Jf, u[:,t], u[:,t],1e-5*nb_x,20)
    end
    return u
end

function implicit_euler(u0)
    u0 = vect(u0)
    f(x) = x  - Δt*F2(x) - Δt*D2*M2*x
    Jf(x) = I-Δt*JF(x)-Δt*D2*M2

    sol = generic_implicit_euler(u0,f,Jf)

    return deflatten(sol)
end

#################### Boîte noire ####################
function boite_noire(u0)
    u0 = vect(u0)

    # definition du problème
    f(u,p,t) = F2(u) + D2*M2*u
    tspan = (0.0,T)
    prob = ODEProblem(f,u0,tspan)

    # résolution
    sol = solve(prob,saveat=Δt)

    return deflatten(reduce(hcat,sol.u))[:,:,1:nb_t]    # we make sure that the dimensions are (nb_x,2,nb_t)
end


#################### Etude théorique ####################

# ̇cn = J(n,d)*cn avec cn le n-ième coefficient de Fourrier de u
J(n,d) = [-(p+q)^2 -2*p/(p+q); (p+q)^2 -1+2*p/(p+q)] - n^2*[1 0; 0 d]

function display_max_eigen_values()
    X = range(0,0.2,100)
    Y = [[maximum(real.(eigen(J(n,d)).values)) for d in X] for n in 0:9]
    Plots.plot(X,Y, legend=false, ylim=(-2,1), xlabel = "d", ylabel = "highest eigen value of J(n,d)")
end

# dichotomy with f an increasing function and f(x_min)<f_target<f(x_max)
function generic_dichotomy(f,x_min,x_max,fx_target,eps)
    fx_min = f(x_min)
    fx_max = f(x_max)
    while(abs(x_max-x_min)>eps)
        new_x = (x_min+x_max)/2.0
        new_fx = f(new_x)
        if(new_fx<fx_target)
            fx_min = new_fx
            x_min = new_x
        else
            fx_max = new_fx
            x_max = new_x
        end
    end
    return (x_min+x_max)/2.0
end

# finding the absciss of the limit of apparition of the comportement
function dichotomy(eps)
    f(d) = - maximum([maximum(real.(eigen(J(n,d)).values)) for n in 0:20])
    generic_dichotomy(f,0,1,0,eps)
end

########################### Resolution #########################

u = explicit_euler(u0)
#u = implicit_euler(u0)
#u = boite_noire(u0)

# heatmap(range(0,T,nb_t),range(0,2pi,nb_x),u[:,1,:], xlabel = "Time", ylabel = "Space")
heatmap(range(0,T,nb_t),range(0,2pi,nb_x),u[:,2,:], xlabel = "Time", ylabel = "Space")
