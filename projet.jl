using PyPlot
using LinearAlgebra
using Plots
using DifferentialEquations
pygui(true)


# solveurs génériques
# vec + devec
# regrouper
# une seule représentation
# tester les fonctions dans un begin end

#################### Paramètres ####################

# user parameters
p=2.0
q=0.05
d=0.03
T=8.0 # durée de la simulation
Δx = 0.1
Δt = 0.005

# initial state u0 = u_eq + A * [sin(k1*x),cos(k2*x)]
A = 1
k1 = 1
k2 = 2

# computed parameters

u0 = zeros(nb_x,2)
u_ini(x) = [p/((p+q)^2)+A*sin(k1*x),p+q+A*cos(k2*x)]
for i in 1:nb_x
    u0[i,:] = f(i*Δx)
end

D =[1. 0.;0. d]
nb_x = floor(Int,2*pi/Δx)
nb_t = floor(Int,T/Δt)
Δx = 2*pi/nb_x
Δt = T/nb_t

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

#test
begin
    x = [1 3;2 4]
    @assert(vect(x)==[1,2,3,4])
end

# reshape a vector into a matrix with 2 columns
# input [u1, u2, ... u2N]
# output [u1 uN+1; u2 uN+2; ... ; uN u2N]
function devect(u)
    L = size(u)[1]÷2
    devect_u = zeros(L,2)
    for i in 1:L
        devect_u[i,:] = [u[i], u[i+L]]
    end
    devect_u
end

# test
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

function f(u)
    [p-u[1]*u[2]^2;q-u[2]+u[1]*u[2]^2]
end

# Jacobienne de f
function Jf(u)
      [-u[2]^2 -2*u[1]*u[2] ;
        u[2]^2 -1+2*u[1]*u[2] ]
end

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
    L = size(u)[1]÷2
    for i in 1:L
        fu = f([u[i],u[i+L]])
        F2u[i] = fu[1]
        F2u[L+i] =fu[2]
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
    u
end

# Intégration de l'edp avec la méthode d'Euler explicite
function euler_explicite(u0)
    u0 = vect(u0)
    f(u) = D2*M2*u + F2(u)
    sol = generic_implicit_euler(u0,f)
    return deflatten(sol)
end

#################### Euler implicite ####################

function newton(f,Jf,x0,u,t,eps,max_iter)
    x=x0
    i=0
    while(norm(f(x, parameter))>eps && i<max_iter)
        x = x - inv(Jf(x))*f(x, parameter)
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
        u[:,t+1] = newton(f,Jf, u[:,t], u, t,1e-5,50)
    end
end

function euler_implicite(u0)
    u0 = vect(u0)
    f(x, u_t) = x - u_t - Δt*F2(x) - Δt*D2*M2*x
    Jf(x) = I-Δt*JF(x)-Δt*D2*M2

    sol = generic_implicit_euler(u0,f,Jf)

    return deflatten(sol)
end

#################### Boîte noire #######################
function boite_noire(u0)
    u_ini = zeros(2*nb_x)

    # initialisation
    for k in 1:nb_x
        v = u0(k*Δx)
        u_ini[k,1] = v[1]
        u_ini[k+nb_x] = v[2]
    end

    # definition du problème
    f(u,p,t) = F2(u) + D2*M2*u
    tspan = (0.0,T)
    prob = ODEProblem(f,u_ini,tspan)

    # résolution
    sol = solve(prob,saveat=Δt)

    return deflatten(reduce(hcat,sol.u))
end


#################### Etude théorique ####################

# Matrice d'eq diff pour les coeff de Fourrier
function J(n,d)
    return [-(p+q)^2 -2*p/(p+q); (p+q)^2 -1+2*p/(p+q)] - n^2*[1 0; 0 d]
end

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

#################### Animation ####################

tps_affichage = T
fps = 25
n=floor(Int,fps*tps_affichage)  # nombre de frames
lst_t = [1/n*t*T for t in 1:n]  # liste des temps auxquels on affiche
lst_i = [floor(Int,t/Δt) for t in lst_t]    # liste des indices correspondants

# Calcul de la solution
#u = euler_explicite(u0)
#u = euler_implicite(u0)
u = boite_noire(u0)

# abscisses pour
X = [k*Δx for k in 1:nb_x]

#Etats stables
#Y1 = ones(nb_x)*p/((p+q)^2)
#Y2 = ones(nb_x)*(p+q)

# maximum et minimum de la solution pour l'affichage
#y1 = minimum(u)
#y2 = maximum(u)

#animation
#anim = @animate for i ∈ 1:n
#    Plots.plot(X,[Y1 Y2 u[:,lst_i[i],1] u[:,lst_i[i],2]],legend=false,ylim=(y1,y2))
#end

#gif(anim, fps = fps)

#Plots.plot(X,[Y1 Y2 u[:,nb_t,1] u[:,nb_t,2]],legend=false)

####################  ####################
