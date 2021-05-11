using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack


push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
using EntropyStableEuler
using EntropyStableEuler.Fluxes1D

function wavespeed_1D(rho,rhou,E)
    p = pfun_nd(rho,(rhou,),E)
    cvel = @. sqrt(γ*p/rho)
    return @. abs(rhou/rho) + cvel
end
unorm(U) = sum(map((x->x.^2),U))
function pfun_nd(rho, rhoU, E)
    rhoUnorm2 = unorm(rhoU)./rho
    return @. (γ-1)*(E - .5*rhoUnorm2)
end

function primitive_to_conservative_hardcode(rho,U,p)
    rhoU = rho.*U
    Unorm = unorm(U)
    E = @. p/(γ-1) + .5*rho*Unorm
    return (rho,rhoU,E)
end

function euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,
    rhoR,uR,betaR,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))

    unorm = (@. uL*uR)
    pa    = (@. rhoavg/(betaL+betaR))
    f4aux = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

    FxS1 = (@. rholog*uavg)
    FxS2 = (@. FxS1*uavg + pa)
    FxS3 = (@. f4aux*uavg)

    return (FxS1,FxS2,FxS3)
end

function euler_fluxes(rhoL,uL,betaL,rhoR,uR,betaR)
    rhologL,betalogL,rhologR,betalogR = map(x->log.(x),(rhoL,betaL,rhoR,betaR))
    return euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,
                           rhoR,uR,betaR,rhologR,betalogR)
end

function dopri45_coeffs()
    rk4a = [0.0             0.0             0.0             0.0             0.0             0.0         0.0
            0.2             0.0             0.0             0.0             0.0             0.0         0.0
            3.0/40.0        9.0/40.0        0.0             0.0             0.0             0.0         0.0
            44.0/45.0      -56.0/15.0       32.0/9.0        0.0             0.0             0.0         0.0
            19372.0/6561.0 -25360.0/2187.0  64448.0/6561.0  -212.0/729.0    0.0             0.0         0.0
            9017.0/3168.0  -355.0/33.0      46732.0/5247.0  49.0/176.0      -5103.0/18656.0 0.0         0.0
            35.0/384.0      0.0             500.0/1113.0    125.0/192.0     -2187.0/6784.0  11.0/84.0   0.0 ]

    rk4c = vec([0.0 0.2 0.3 0.8 8.0/9.0 1.0 1.0 ])

    # coefficients to evolve error estimator = b1-b2
    rk4E = vec([71.0/57600.0  0.0 -71.0/16695.0 71.0/1920.0 -17253.0/339200.0 22.0/525.0 -1.0/40.0 ])

    return rk4a,rk4E,rk4c
end

const TOL = 1e-16
"Approximation parameters"
N = 4 # The order of approximation
K = 2^7
T = 0.1

# Becker viscous shocktube
const γ = 1.4
const M_0 = 3.0
const mu = 0.1
const lambda = 2/3*mu
const Pr = 3/4
const cp = γ/(γ-1)
const cv = 1/(γ-1)
const kappa = mu*cp/Pr

const v_inf = 0.2
const rho_0 = 1.0
const v_0 = 1.0
const m_0 = rho_0*v_0
const v_1 = (γ-1+2/M_0^2)/(γ+1)
const v_01 = sqrt(v_0*v_1)

# const uL = v_0+v_inf
# const uR = v_1+v_inf
# const rhoL = m_0/v_0
# const rhoR = m_0/v_1
# const eL = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_0^2)
# const eR = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_1^2)
# const pL = (γ-1)*rhoL*eL
# const pR = (γ-1)*rhoR*eR
# const EL = pL/(γ-1)+0.5*rhoL*uL^2
# const ER = pR/(γ-1)+0.5*rhoR*uR^2

const XL = -2.0#-1.0
const XR = 2.0
gr(size=(300,300),ylims=(0,5.0),legend=false,markerstrokewidth=1,markersize=2)
plot()

"Mesh related variables"
VX = LinRange(XL,XR,K+1)
EToV = transpose(reshape(sort([1:K; 2:K+1]),2,K))
rd = init_reference_interval(N)

# High order mesh
@unpack r,M,Dr,Vq,Pq,Vf,LIFT,wq,VDM = rd
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])
x = V1*VX[transpose(EToV)]
xf = Vf*x
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nrJ = [-1;1]
wf  = [1;1]
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

# construct hybridized SBP operators
Qr = Pq'*M*Dr*Pq
Ef = Vf*Pq
Br = diagm(wf.*nrJ)
Qrh = .5*[Qr-Qr' Ef'*Br;
         -Br*Ef  Br]

Vh = [Vq;Vf]
Ph = M\transpose(Vh)
VhP = Vh*Pq

# make sparse skew symmetric versions of the operators"
# precompute union of sparse ids for Qr, Qs
Qrh_skew = .5*(Qrh-transpose(Qrh))


"""Initial condition"""
function bisection_solve_velocity(x,max_iter,tol)
    v_L = v_1
    v_R = v_0
    num_iter = 0

    L_k = kappa/m_0/cv
    f(v) = -x + 2*L_k/(γ+1)*log((v_0-v)^((v_0/(v_0-v_1)))*(v-v_1)^(-(v_1)/(v_0-v_1)))

    v_new = (v_L+v_R)/2
    while num_iter < max_iter
        v_new = (v_L+v_R)/2

        if abs(f(v_new)) < tol
            return v_new
        elseif sign(f(v_L)) == sign(f(v_new))
            v_L = v_new
        else
            v_R = v_new
        end
        num_iter += 1
    end

    return v_new
end

const max_iter = 10000
const tol = 1e-16

function exact_sol_viscous_shocktube(x,t)
    u   = bisection_solve_velocity(x-v_inf*t,max_iter,tol)
    rho = m_0/u
    e   = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-u^2)
    return rho, rho*(v_inf+u), rho*(e+1/2*(v_inf+u)^2)
end


U = exact_sol_viscous_shocktube.(x,0.0)
U = ([x[1] for x in U], [x[2] for x in U], [x[3] for x in U])

function flux_differencing!(UF,Uh,Qrh_skew,rxJ,Nh,Nq,Nc,K)
    for k = 1:K
        for j = 1:Nh # col idx
            for i = j:Nh # row idx
                if i <= Nq || j <= Nq # Skip lower right block
                    Fx = euler_fluxes(Uh[1][i,k],Uh[2][i,k],Uh[3][i,k],Uh[1][j,k],Uh[2][j,k],Uh[3][j,k])
                    for c = 1:Nc
                        update_val = 2*rxJ*Qrh_skew[i,j]*Fx[c]
                        UF[c][i,k] += update_val
                        UF[c][j,k] -= update_val
                    end
                end
            end
        end
    end
end

function rhs_inviscid(U,K,VhP,Vq,LIFT,Ph,Qrh_skew,rxJ,mapP,nxJ,rhoL,uL,betaL,rhoR,uR,betaR)
    J = (XR-XL)/K/2 # assume uniform interval
    Nc = 3
    Nh,Nq = size(VhP)

    # Preallocate flux differencing array
    UF = [zeros(Float64,Nh,K) for i = 1:Nc]
    
    # entropy var projection
    VU = v_ufun((x->Vq*x).(U)...)
    VU = (x->VhP*x).(VU)
    Uh = u_vfun(VU...)

    # convert to rho,u,v,beta vars
    (rho,rhou,E) = Uh
    beta = betafun(rho,rhou,E)
    Ubh = (rho, rhou./rho, beta) # redefine Q = (rho,u,v,β)

    # compute face values
    UM = (x->x[Nq+1:Nh,:]).(Ubh)
    UP = (x->x[mapP]).(UM)
    UP[1][1] = rhoL
    UP[2][1] = uL
    UP[3][1] = betaL
    UP[1][end] = rhoR
    UP[2][end] = uR
    UP[3][end] = betaR

    # simple lax friedrichs dissipation
    Uf =  (x->x[Nq+1:Nh,:]).(Uh)
    (rhoM,rhouM,EM) = Uf
    rhoUM_n = @. (rhouM*nxJ)
    lam = abs.(wavespeed_1D(rhoM,rhoUM_n,EM))
    LFc = .25*max.(lam,lam[mapP])

    fSx = euler_fluxes(UM[1],UM[2],UM[3],UP[1],UP[2],UP[3])
    normal_flux(fx,u) = fx.*nxJ - LFc.*(u[mapP]-u)
    flux = normal_flux.(fSx,Uf)
    rhsU = (x->LIFT*x).(flux)

    flux_differencing!(UF,Ubh,Qrh_skew,rxJ,Nh,Nq,Nc,K)
    rhsU = (x->Ph*x).(UF) .+ rhsU
    rhsU = (x->-x./J).(rhsU)

    return rhsU
end

function rhs_viscous(U,K,N,Vq,Vf,Pq,Dr,LIFT,mapP,nxJ,v1L,v2L,v3L,v1R,v2R,v3R)
    J = (XR-XL)/K/2 # assume uniform interval
    Nc = 3

    # entropy var projection
    VU = v_ufun((x->Vq*x).(U)...)
    VU = (x->Pq*x).(VU)

    # compute and interpolate to quadrature
    VUf = (x->Vf*x).(VU)
    VUP = (x->x[mapP]).(VUf)
    VUP[1][1] = v1L
    VUP[2][1] = v2L
    VUP[3][1] = v3L
    VUP[1][end] = v1R
    VUP[2][end] = v2R
    VUP[3][end] = v3R

    surfx(uP,uf) = LIFT*(@. .5*(uP-uf)*nxJ)
    # Strong form, dv/dx
    VUx = (x->Dr*x).(VU)
    VUx = VUx .+ surfx.(VUP,VUf)
    VUx = VUx./J
    
    VUx = (x->Vq*x).(VUx)
    VUq = (x->Vq*x).(VU)

    # σ = K dv/dx
    sigma = zero.(VUx)
    for k = 1:K
        for i = 1:size(Vq,1)
            Kx = zeros(3,3)
            v1 = VUq[1][i,k]
            v2 = VUq[2][i,k]
            v4 = VUq[3][i,k]
            Kx[2,2] = -(2*mu-lambda)/v4
            Kx[2,3] = (2*mu-lambda)*v2/v4^2
            Kx[3,2] = Kx[2,3]
            Kx[3,3] = -(2*mu-lambda)*v2^2/v4^3+kappa/cv/v4^2

            sigma[2][i,k] += Kx[2,2]*VUx[2][i,k] + Kx[2,3]*VUx[3][i,k]
            sigma[3][i,k] += Kx[3,2]*VUx[2][i,k] + Kx[3,3]*VUx[3][i,k]
        end
    end
    sigma = (x->Pq*x).(sigma)

    sxf = (x->Vf*x).(sigma)
    sxP = (x->x[mapP]).(sxf)
    sxP[1][1] = sxf[1][1]
    sxP[2][1] = sxf[2][1]
    sxP[3][1] = sxf[3][1]
    sxP[1][end] = sxf[1][end]
    sxP[2][end] = sxf[2][end]
    sxP[3][end] = sxf[3][end]

    # strong form, dσ/dx
    penalization(uP,uf) = LIFT*(@. -.5*(uP-uf))
    sigmax = (x->Dr*x).(sigma)
    sigmax = sigmax .+ surfx.(sxP,sxf)
    sigmax = sigmax./J

    return sigmax
end

function rhs_ESDG(U,K,VhP,Vq,Vf,Pq,Dr,LIFT,Ph,Qrh_skew,rxJ,mapP,nxJ,t)

    rhoL,rhouL,EL = exact_sol_viscous_shocktube(XL,t)
    rhoR,rhouR,ER = exact_sol_viscous_shocktube(XR,t)
    uL = rhouL/rhoL
    uR = rhouR/rhoR
    pL = (γ-1)*(EL-.5*rhoL*uL^2)
    pR = (γ-1)*(ER-.5*rhoR*uR^2)
    betaL = rhoL/(2*pL)
    betaR = rhoR/(2*pR)
    v1L,v2L,v3L = v_ufun(rhoL,rhoL*uL,EL)
    v1R,v2R,v3R = v_ufun(rhoR,rhoR*uR,ER)

    rhsI = rhs_inviscid(U,K,VhP,Vq,LIFT,Ph,Qrh_skew,rxJ,mapP,nxJ,rhoL,uL,betaL,rhoR,uR,betaR)
    rhsV = rhs_viscous(U,K,N,Vq,Vf,Pq,Dr,LIFT,mapP,nxJ,v1L,v2L,v3L,v1R,v2R,v3R)
    return rhsI .+ rhsV
end




# Time stepping
"Time integration"
t = 0.0
U = collect(U)
resU = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resW = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resZ = [zeros(size(x)),zeros(size(x)),zeros(size(x))]

Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM


anim = Animation()
const ptL = XL+(XR-XL)/K/(N+1)/2
const ptR = XR-(XR-XL)/K/(N+1)/2
const hplot = (XR-XL)/K/(N+1)
i = 1

while t < T

    # SSPRK(3,3)
    # dt = min(1e-4,T-t)
    CN = (N+1)*(N+2)/2  # estimated trace constant
    dt = min(2/CN/K^2,T-t)
    rhsU = rhs_ESDG(U,K,VhP,Vq,Vf,Pq,Dr,LIFT,Ph,Qrh_skew,rxJ,mapP,nxJ,t)
    @. resW = U + dt*rhsU
    rhsU = rhs_ESDG(resW,K,VhP,Vq,Vf,Pq,Dr,LIFT,Ph,Qrh_skew,rxJ,mapP,nxJ,t)
    @. resZ = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resZ
    rhsU = rhs_ESDG(resW,K,VhP,Vq,Vf,Pq,Dr,LIFT,Ph,Qrh_skew,rxJ,mapP,nxJ,t)
    @. resZ = resW+dt*rhsU
    @. U = 1/3*U+2/3*resZ

    global t = t + dt
    global i = i + 1
    println("Current time $t with time step size $dt, and final time $T")
    if mod(i,1000) == 1
        plot(x,U[1])
        # for k = 1:K
        #     plot!(ptL+(k-1)*hplot*(N+1):hplot:ptL+k*hplot*(N+1), 1 .-L_plot[:,k],st=:bar,alpha=0.2)
        # end
        frame(anim)
    end

    # if i == 10
    #     break
    # end
end

# rka,rkE,rkc = dopri45_coeffs()
# bcopy!(x,y) = x .= y

# # DOPRI storage
# Utmp = similar.(U)
# rhsUrk = ntuple(x->zero.(U),length(rkE))

# errEst = 0.0
# prevErrEst = 0.0

# t = 0.0
# CN = (N+1)*(N+2)/2  # estimated trace constant
# dt = min(2/CN/K^2,T-t)
# dt0 = dt
# i = 0
# interval = 5

# dthist = Float64[dt]
# thist = Float64[0.0]
# errhist = Float64[0.0]
# wsJ = diagm(wf)

# rhsU = rhs_ESDG(U,K,VhP,Vq,Vf,Pq,Dr,LIFT,Ph,Qrh_skew,rxJ,mapP,nxJ,t)
# bcopy!.(rhsUrk[1],rhsU) # initialize DOPRI rhs (FSAL property)

# while t < T
#     # DOPRI step and
#     rhstest = 0.0
#     for INTRK = 2:7
#         k = zero.(Utmp)
#         for s = 1:INTRK-1
#             bcopy!.(k, @. k + rka[INTRK,s]*rhsUrk[s])
#         end
#         bcopy!.(Utmp, @. U + dt*k)
#         rhsU = rhs_ESDG(Utmp,K,VhP,Vq,Vf,Pq,Dr,LIFT,Ph,Qrh_skew,rxJ,mapP,nxJ,t)#rhsRK(Qtmp,rd,md,ops,euler_fluxes)
#         bcopy!.(rhsUrk[INTRK],rhsU)
#     end
#     errEstVec = zero.(Utmp)
#     for s = 1:7
#         bcopy!.(errEstVec, @. errEstVec + rkE[s]*rhsUrk[s])
#     end

#     errTol = 1e-5
#     errEst = 0.0
#     for field = 1:length(Utmp)
#         errEstScale = @. abs(errEstVec[field]) / (errTol*(1+abs(U[field])))
#         errEst += sum(errEstScale.^2) # hairer seminorm
#     end
#     errEst = sqrt(errEst/(length(U[1])*4))
#     if errEst < 1.0 # if err small, accept step and update
#             bcopy!.(U, Utmp)
#             global t += dt
#             bcopy!.(rhsUrk[1], rhsUrk[7]) # use FSAL property
#     end
#     order = 5
#     dtnew = .8*dt*(.9/errEst)^(.4/(order+1)) # P controller
#     if i > 0 # use PI controller if prevErrEst available
#             dtnew *= (prevErrEst/max(1e-14,errEst))^(.3/(order+1))
#     end
#     global dt = max(min(10*dt0,dtnew),1e-9) # max/min dt
#     global prevErrEst = errEst

#     push!(dthist,dt)
#     push!(thist,t)

#     global i = i + 1  # number of total steps attempted
#     if i%interval==0
#         println("i = $i, t = $t, dt = $dtnew, errEst = $errEst, rhstest = $rhstest")
#     end
# end

# plot(x,U[1])
# gif(anim,"~/Desktop/tmp.gif",fps=15)

xq = Vq*x
exact_U = @. exact_sol_viscous_shocktube.(xq,T)
exact_rho = [x[1] for x in exact_U]
exact_rhou = [x[2] for x in exact_U]
exact_u = exact_rhou./exact_rho
exact_E = [x[3] for x in exact_U]

rho = Vq*U[1]
u = Vq*(U[2]./U[1])
rhou = Vq*U[2]
E = Vq*U[3]
p = Vq*pfun_nd.(U[1],U[2],U[3])
J = (XR-XL)/K/2

# Linferr = maximum(abs.(exact_rho-rho))/maximum(abs.(exact_rho)) +
#           maximum(abs.(exact_u-u))/maximum(abs.(exact_u)) +
#           maximum(abs.(exact_E-E))/maximum(abs.(exact_E))

# L1err = sum(J*wq.*abs.(exact_rho-rho))/sum(J*wq.*abs.(rho)) +
#         sum(J*wq.*abs.(exact_u-u))/sum(J*wq.*abs.(u)) +
#         sum(J*wq.*abs.(exact_E-E))/sum(J*wq.*abs.(E))

Linferr = maximum(abs.(exact_rho-rho))/maximum(abs.(exact_rho)) +
          maximum(abs.(exact_rhou-rhou))/maximum(abs.(exact_rhou)) +
          maximum(abs.(exact_E-E))/maximum(abs.(exact_E))

L1err = sum(J*wq.*abs.(exact_rho-rho))/sum(J*wq.*abs.(rho)) +
        sum(J*wq.*abs.(exact_rhou-rhou))/sum(J*wq.*abs.(rhou)) +
        sum(J*wq.*abs.(exact_E-E))/sum(J*wq.*abs.(E))

L2err = sqrt(sum(J*wq.*abs.(exact_rho-rho).^2))/sqrt(sum(J*wq.*abs.(rho).^2)) +
        sqrt(sum(J*wq.*abs.(exact_rhou-rhou).^2))/sqrt(sum(J*wq.*abs.(rhou).^2)) +
        sqrt(sum(J*wq.*abs.(exact_E-E).^2))/sqrt(sum(J*wq.*abs.(E).^2))

println("N = $N, K = $K")
println("L1 error is $L1err")
println("L2 error is $L2err")
println("Linf error is $Linferr")