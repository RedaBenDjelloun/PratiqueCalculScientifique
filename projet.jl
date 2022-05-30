using PyPlot
using LinearAlgebra
using Plots
using DifferentialEquations
pygui(true)

#################### Paramètres ####################

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
D =[1. 0.;0. d]
nb_x = floor(Int,2*pi/Δx)
nb_t = floor(Int,T/Δt)
Δx = 2*pi/nb_x
Δt = T/nb_t
u0 = zeros(nb_x,2)
for i in 1:nb_x
    x = i*Δx
    u0[i,:] = [p/((p+q)^2)+A*sin(k1*x),p+q+A*cos(k2*x)]
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

# M2 is the discrete operator for 2nd derivative when u is in the vectorial form
M2 = [M zeros(nb_x,nb_x); zeros(nb_x,nb_x) M]

D2 = zeros(2*nb_x,2*nb_x)
for i in 1:nb_x
    D2[i,i]=1
    D2[i+nb_x,i+nb_x]=d
end


###################### u representation ###################

# reshape a matrix into a vector column by column
function vect(u)
    u[:]
end

#test de vect
begin
    x = [1 3;2 4]
    @assert(vect(x)==[1,2,3,4])
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
    x = [1,2,3,4]
    @assert(devect(x)==[1 3; 2 4])
end

function deflatten(u)
    v = zeros(size(u)[1]÷2,2,size(u)[2])
    for i in 1:nb_x
        v[i,1,:]=u[i,:,:]
        v[i,2,:]=u[nb_x+i,:,:]
    end
    return v
end

#################### f and derivatives ####################

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
    return Fu
end

# input : u of the form [u[:,1], u[:,2]]
# output : [f(u[i,:])]
function F2(u)
    F2u = zero(u)
    new_size = size(u)[1]÷2
    for i in 1:new_size
        fu = f([u[i],u[i+new_size]])
        F2u[i] = fu[1]
        F2u[new_size+i] =fu[2]
    end
    return F2u
end

#test
begin
    u = rand(50,2)
    Fu1 = F(u)
    Fu2 = devect(F2(vect(u)))
    @assert(norm(Fu1-Fu2)<1e-10)
end

# input : u of the form [u[:,1], u[:,2]]
# output : Jacobian of F at u
function JF(u)
    JFu = zeros(2*nb_x,2*nb_x)
    for i in 1:nb_x
        df = Jf([u[i];u[nb_x+i]])
        JFu[i,i] = df[1,1]
        JFu[i,nb_x+i] = df[1,2]
        JFu[nb_x+i,i] = df[2,1]
        JFu[nb_x+i,nb_x+i] = df[2,2]
    end
    return JFu
end


##################### Euler Explicite ######################

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
function euler_explicite(u0)
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

function euler_implicite(u0)
    u0 = vect(u0)
    f(x) = x  - Δt*F2(x) - Δt*D2*M2*x
    Jf(x) = I-Δt*JF(x)-Δt*D2*M2

    sol = generic_implicit_euler(u0,f,Jf)

    return deflatten(sol)
end

#################### Boîte noire #######################
function boite_noire(u0)
    u0 = vect(u0)

    # definition du problème
    f(u,p,t) = F2(u) + D2*M2*u
    tspan = (0.0,T)
    prob = ODEProblem(f,u0,tspan)

    # résolution
    sol = solve(prob,saveat=Δt)

    return deflatten(reduce(hcat,sol.u))
end


#################### Etude théorique ####################

# ̇cn = J(n,d)*cn avec cn le n-ième coefficient de Fourrier de u
J(n,d) = [-(p+q)^2 -2*p/(p+q); (p+q)^2 -1+2*p/(p+q)] - n^2*[1 0; 0 d]

# Affichage
#X = [d for d in range(0,0.2,100)]
#Y = [[maximum(real.(eigen(J(n,d)).values)) for d in X] for n in 0:10]
#Plots.plot(X,Y, legend=false, ylim=(-2,1))

# simple dichotomie pour trouver la limite de changement de comportement
function dichotomie(eps)
    d_min = 1
    d_max = 0
    min = maximum([maximum(real.(eigen(J(n,d_min)).values)) for n in 0:20])
    max = maximum([maximum(real.(eigen(J(n,d_max)).values)) for n in 0:20])
    while(abs(d_min-d_max)>eps)
        new_d = (d_min + d_max)/2.0
        new_e = maximum([maximum(real.(eigen(J(n,new_d)).values)) for n in 0:20])
        if (new_e<0)
            min = new_e
            d_min = new_d
        else
            max = new_e
            d_max = new_d
        end
    end
    return (d_min+d_max)/2.0
end

########################### Calcul de la solution #########################
#u = euler_explicite(u0)
#u = euler_implicite(u0)
u = boite_noire(u0)

heatmap(u[:,1,:])
