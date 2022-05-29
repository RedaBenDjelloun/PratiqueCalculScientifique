using PyPlot
using LinearAlgebra
using Plots
using DifferentialEquations
pygui(true)

#################### Paramètres ####################

p=2.0
q=0.05
d=0.1
T=3.0 # durée de la simulation
Δx = 0.1
Δt = 0.005
A = 1
ω1 = 1
ω2 = 2
D =[1. 0.;0. d]
nb_x = floor(Int,2*pi/Δx)
nb_t = floor(Int,T/Δt)
Δx = 2*pi/nb_x
Δt = T/nb_t

function norme(v)
    return sum([elt^2 for elt in v])
end

#################### Modélisation du problème ####################

function f(u)
    return [p-u[1]*u[2]^2;q-u[2]+u[1]*u[2]^2]
end

function ∇f(u)
    return [-u[2]^2 -2*u[1]*u[2];u[2]^2 -1+2*u[1]*u[2]]
end

# applique f à tous les éléments du vecteur de vecteurs
function F(u)
    v = zeros(nb_x,2)
    for i in 1:nb_x
        v[i,:] = f(u[i,:])
    end
    return v
end

#################### Euler explicite ####################

function ∇F(u)
    v = zeros(2*nb_x,2*nb_x)
    for i in 1:nb_x
        df = ∇f([u[i];u[nb_x+i]])
        v[i,i] = df[1,1]
        v[i,nb_x+i] = df[1,2]
        v[nb_x+i,i] = df[2,1]
        v[nb_x+i,nb_x+i] = df[2,2]
    end
    return v
end

# M permet de faire des dérivées secondes discrètes
M = zeros(nb_x,nb_x)
for i in 1:nb_x-1
    M[i,i]=-2
    M[i,i+1]=1
    M[i+1,i]=1
end
M[nb_x,nb_x]=-2
M[nb_x,1]=1
M[1,nb_x]=1

M2 = zeros(2*nb_x,2*nb_x)
for j in [0,nb_x]
    for i in 1:nb_x-1
        M2[i+j,i+j]=-2
        M2[i+j,i+j+1]=1
        M2[i+j+1,i+j]=1
    end
    M2[nb_x+j,nb_x+j]=-2
    M2[nb_x+j,1+j]=1
    M2[1+j,nb_x+j]=1
end

D2 = zeros(2*nb_x,2*nb_x)
for i in 1:nb_x
    D2[i,i]=1
    D2[i+nb_x,i+nb_x]=d
end

# Etat initial = état stable + fluctuation
function u0(x)
    return  [p/((p+q)^2)+A*sin(ω1*x),p+q+A*cos(ω2*x)]
end

# Intégration de l'edp avec la méthode d'Euler explicite
function euler_explicite(Δt,Δx,T, u0)
    nb_x = floor(Int,2*pi/Δx)
    Δx = 2*pi/nb_x # on arrondi Δx pour qu'il n'y ait pas de problème au bord
    nb_t = floor(Int,T/Δt)
    Δt = T/nb_t
    u = zeros(nb_x,nb_t,2)

    # initialisation
    for k in 1:nb_x
        u[k,1,:] = u0(k*Δx)
    end

    # intégration
    for t in 1:(nb_t-1)
        u[:,t+1,:] = u[:,t,:] + Δt*F(u[:,t,:]) + Δt/(Δx^2)*M*u[:,t,:]*D
    end

    return u
end

#################### Euler implicite ####################
# applique f à tous les éléments du vecteur applatit
function F2(u)
    v = zeros(2*nb_x)
    for i in 1:nb_x
        x = f([u[i],u[i+nb_x]])
        v[i] = x[1]
        v[nb_x+i] =x[2]
    end
    return v
end

function g(Δt,Δx, x, u, t)
    return x - u[:,t] - Δt*F2(x) - Δt/(Δx^2)*D2*M2*x
end

function ∇g(Δt,Δx,x)
    return I-Δt*∇F(x)-Δt/(Δx^2)*D2*M2
end

function flatten(u)
    v = zeros(2*nb_x)
    for  i in 1:nb_x
        v[i,:]=u[i,:,1]
        v[nb_x+i,:]=u[i,:,2]
    end
    return v
end

function deflatten(u)
    v = zeros(size(u)[1]÷2,size(u)[2],2)
    for i in 1:nb_x
        v[i,:,1]=u[i,:,:]
        v[i,:,2]=u[nb_x+i,:,:]
    end
    return v
end

function newton(g,∇g,Δt,Δx,x0,u,t,eps)
    x=x0
    i=0
    while(norme(g(Δt,Δx, x, u, t))>eps && i<100)
        x = x - inv(∇g(Δt,Δx, x))*g(Δt,Δx, x, u, t)
        i+=1
    end
    return x
end

function euler_implicite(Δt,Δx,T, u0)
    nb_x = floor(Int,2*pi/Δx)
    Δx = 2*pi/nb_x # on arrondi Δx pour qu'il n'y ait pas de problème au bord
    nb_t = floor(Int,T/Δt)
    Δt = T/nb_t
    u = zeros(2*nb_x,nb_t)

    # initialisation
    for k in 1:nb_x
        v = u0(k*Δx)
        u[k,1] = v[1]
        u[k+nb_x,1] = v[2]
    end

    # intégration
    for t in 1:(nb_t-1)
        u[:,t+1] = newton(g,∇g,Δt,Δx, u[:,t], u, t,1e-5)
    end

    return deflatten(u)
end

#################### Boîte noire #######################
function boite_noire(Δt,Δx,T,u0)
    nb_x = floor(Int,2*pi/Δx)
    Δx = 2*pi/nb_x # on arrondi Δx pour qu'il n'y ait pas de problème au bord
    nb_t = floor(Int,T/Δt)
    Δt = T/nb_t
    u_ini = zeros(2*nb_x)

    # initialisation
    for k in 1:nb_x
        v = u0(k*Δx)
        u_ini[k,1] = v[1]
        u_ini[k+nb_x] = v[2]
    end

    # definition du problème
    edp(u,p,t) = F2(u) + 1/(Δx^2)*D2*M2*u
    tspan = (0.0,T)
    prob = ODEProblem(edp,u_ini,tspan)

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
u = euler_explicite(Δt,Δx,T,u0)
#u = euler_implicite(Δt,Δx,T,u0)
#u = boite_noire(Δt,Δx,T,u0)

# abscisses pour
X = [k*Δx for k in 1:nb_x]

#Etats stables
Y1 = ones(nb_x)*p/((p+q)^2)
Y2 = ones(nb_x)*(p+q)

# maximum et minimum de la solution pour l'affichage
y1 = minimum(u)
y2 = maximum(u)

#animation
anim = @animate for i ∈ 1:n
    Plots.plot(X,[Y1 Y2 u[:,lst_i[i],1] u[:,lst_i[i],2]],legend=false,ylim=(y1,y2))
end

gif(anim, fps = fps)

#Plots.plot(X,[Y1 Y2 u[:,nb_t,1] u[:,nb_t,2]],legend=false)

####################  ####################
