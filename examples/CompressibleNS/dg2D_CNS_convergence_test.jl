using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using BenchmarkTools
using StaticArrays
using UnPack
using DelimitedFiles

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

function euler_fluxes_2D(UL,UR)
    rhoL = first(UL); betaL = last(UL)
    rhoR = first(UR); betaR = last(UR)
    logL = (log.(rhoL),log.(betaL))
    logR = (log.(rhoR),log.(betaR))
    return euler_fluxes(UL,UR,logL,logR)
end


function euler_fluxes_2D(UL,UR,logL,logR)
    return euler_fluxes(UL...,UR...,logL...,logR...)
end

# 2d version
function euler_fluxes_2D(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR,
                      rhologL,betalogL,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))
    vavg   = (@. .5*(vL+vR))

    unorm = (@. uL*uR + vL*vR)
    pa    = (@. rhoavg/(betaL+betaR))
    f4aux = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

    FxS1 = (@. rholog*uavg)
    FxS2 = (@. FxS1*uavg + pa)
    FxS3 = (@. FxS1*vavg)
    FxS4 = (@. f4aux*uavg)

    FyS1 = (@. rholog*vavg)
    FyS2 = (@. FxS3)
    FyS3 = (@. FyS1*vavg + pa)
    FyS4 = (@. f4aux*vavg)
    return (FxS1,FxS2,FxS3,FxS4),(FyS1,FyS2,FyS3,FyS4)
end

function init_BC_funs(md::MeshData,BCTYPE::Int64)
    @unpack xf,yf,mapP,mapB,nxJ,nyJ,sJ = md
    xb,yb = (x->x[mapB]).((xf,yf))

    lid          = mapB[findall(@. abs(yb-1) < 1e-12)]
    wall         = mapB[findall(@. abs(yb-1) >= 1e-12)]
    bottomwall   = mapB[findall(@. abs(yb+1) < 1e-12)]
    leftwall     = mapB[findall(@. abs(xb+1) < 1e-12)]
    rightwall    = mapB[findall(@. abs(xb-1) < 1e-12)]
    vwall = [leftwall;rightwall]
    hwall = [bottomwall;lid]
    xlid = xf[lid]
    #vlid = ones(size(xlid))
    #vlid = @. exp(-1.0/(5.0*(1.0-xlid^2)))
    #vlid = @. xlid^2-1
    vlid = @. (1+cos(pi*xlid))/2

    boundary = [lid;wall]
    nxw = nxJ[wall]./sJ[wall]
    nyw = nyJ[wall]./sJ[wall]
    nxl = nxJ[lid]./sJ[lid]
    nyl = nyJ[lid]./sJ[lid]
    nx = nxJ[boundary]./sJ[boundary]
    ny = nyJ[boundary]./sJ[boundary]

    function impose_BCs_inviscid!(QP,Qf,md::MeshData)
        # No-slip at walls
        u_1 = Qf[2][boundary]
        u_2 = Qf[3][boundary]
        n_1 = nx
        n_2 = ny

        Un = @. u_1*n_1 + u_2*n_2
        Ut = @. u_1*n_2 - u_2*n_1

        # ρ^+ = ρ, p^+ = p (beta^+ = beta)
        @. QP[1][boundary] = Qf[1][boundary]
        @. QP[4][boundary] = Qf[4][boundary]

        # # u_n^+ = -u_n, u_t^+ = u_t
        # @. QP[2][boundary] = 1/(-n_1^2-n_2^2)*(n_1*Un-n_2*Ut)
        # @. QP[3][boundary] = 1/(-n_1^2-n_2^2)*(n_2*Un+n_1*Ut)
        @. QP[2][boundary] = Qf[2][boundary] - 2*Un*nx
        @. QP[3][boundary] = Qf[3][boundary] - 2*Un*ny
    end

    function impose_BCs_entropyvars!(VUP,VUf,md::MeshData)
        if BCTYPE == 1
            # Adiabatic no-slip BC
            @. VUP[2][wall] = -VUf[2][wall]
            @. VUP[3][wall] = -VUf[3][wall]
            @. VUP[4][wall] =  VUf[4][wall]

            @. VUP[2][lid] = -VUf[2][lid] - 2*vlid*VUf[4][lid]
            @. VUP[3][lid] = -VUf[3][lid]
            @. VUP[4][lid] =  VUf[4][lid]
        elseif BCTYPE == 2
            theta = 1.0/0.3^2/1.4/0.4#1.0
            # Isothermal BC
            @. VUP[2][wall] = -VUf[2][wall]
            @. VUP[3][wall] = -VUf[3][wall]
            @. VUP[4][wall] = -2.0/theta-VUf[4][wall]

            @. VUP[2][lid] = 2.0/theta-VUf[2][lid]
            @. VUP[3][lid] = -VUf[3][lid]
            @. VUP[4][lid] = -2.0/theta-VUf[4][lid]
        elseif BCTYPE == 3
            # Reflective BC
            v_1 = VUf[2][boundary]
            v_2 = VUf[3][boundary]
            n_1 = nx
            n_2 = ny

            VUn = @. v_1*n_1 + v_2*n_2
            VUt = @. v_1*n_2 - v_2*n_1

            # v_4^+ = v_4
            @. VUP[4][boundary] = VUf[4][boundary]
            # v_n^+ = -v_n, v_t^+ = v_t
            # @. VUP[2][boundary] = 1/(-n_1^2-n_2^2)*(n_1*VUn-n_2*VUt)
            # @. VUP[3][boundary] = 1/(-n_1^2-n_2^2)*(n_2*VUn+n_1*VUt)
            @. VUP[2][boundary] = VUf[2][boundary] - 2*VUn*nx
            @. VUP[3][boundary] = VUf[3][boundary] - 2*VUn*ny
        end
    end

    function impose_BCs_stress!(σxP,σyP,σxf,σyf,VUf,md::MeshData)
        if BCTYPE == 1
            # Adiabatic no-slip BC
            @. σxP[2][wall] = σxf[2][wall]
            @. σyP[2][wall] = σyf[2][wall]
            @. σxP[3][wall] = σxf[3][wall]
            @. σyP[3][wall] = σyf[3][wall]

            @. σxP[2][lid] = σxf[2][lid]
            @. σyP[2][lid] = σyf[2][lid]
            @. σxP[3][lid] = σxf[3][lid]
            @. σyP[3][lid] = σyf[3][lid]

            @. σxP[4][wall] = -σxf[4][wall]
            @. σyP[4][wall] = -σyf[4][wall]
            @. σxP[4][lid]  = -σxf[4][lid] + 2*vlid*σxf[2][lid]
            @. σyP[4][lid]  = -σyf[4][lid] + 2*vlid*σyf[2][lid]
        elseif BCTYPE == 2
            # Isothermal
            @. σxP[2][boundary] = σxf[2][boundary]
            @. σyP[2][boundary] = σyf[2][boundary]
            @. σxP[3][boundary] = σxf[3][boundary]
            @. σyP[3][boundary] = σyf[3][boundary]
            @. σxP[4][boundary] = σxf[4][boundary]
            @. σyP[4][boundary] = σyf[4][boundary]
        elseif BCTYPE == 3
            sigma_x_1 = σxf[2][boundary]
            sigma_x_2 = σxf[3][boundary]
            sigma_y_1 = σyf[2][boundary]
            sigma_y_2 = σyf[3][boundary]
            n_1 = nx
            n_2 = ny

            σn_x = @. sigma_x_1*n_1 + sigma_x_2*n_2
            σn_y = @. sigma_y_1*n_1 + sigma_y_2*n_2

            @. σxP[2][boundary] = -σxf[2][boundary]+2*n_1*σn_x
            @. σyP[2][boundary] = -σyf[2][boundary]+2*n_1*σn_y
            @. σxP[3][boundary] = -σxf[3][boundary]+2*n_2*σn_x
            @. σyP[3][boundary] = -σyf[3][boundary]+2*n_2*σn_y

            @. σxP[4][boundary] = -σxf[4][boundary]
            @. σyP[4][boundary] = -σyf[4][boundary]
        end
    end
    return impose_BCs_inviscid!,impose_BCs_entropyvars!,impose_BCs_stress!
end

"dense version - speed up by prealloc + transpose for col major "
function dense_hadamard_sum(Qhe,ops,vgeo,flux_fun)

    (Qr,Qs) = ops
    (rxJ,sxJ,ryJ,syJ) = vgeo
    # transpose for column-major evals
    QxTr = transpose(rxJ*Qr + sxJ*Qs)
    QyTr = transpose(ryJ*Qr + syJ*Qs)

    # precompute logs for logmean
    (rho,u,v,beta) = Qhe
    Qlog = (log.(rho), log.(beta))
    Qlogi = zeros(length(Qlog))
    Qlogj = zeros(length(Qlog))

    n = size(Qr,1)
    nfields = length(Qhe)

    QF = ntuple(x->zeros(n),nfields)
    QFi = zeros(nfields)
    for i = 1:n
        Qi = getindex.(Qhe,i)
        Qlogi = getindex.(Qlog,i)

        fill!(QFi,0)
        for j = 1:n
            Qj = getindex.(Qhe,j)
            Qlogj = getindex.(Qlog,j)

            Fx,Fy = flux_fun(Qi,Qj,Qlogi,Qlogj)
            @. QFi += QxTr[j,i]*Fx + QyTr[j,i]*Fy
        end

        for fld in eachindex(Qhe)
            QF[fld][i] = QFi[fld]
        end
    end

    return QF
end

function update_flux!(rhsQ,QM,QP,Uf,UP,LFc,nxJ,nyJ,mapP,Nfq,K,Nd,inviscid_dissp)
    for k = 1:K
        for i = 1:Nfq
            fx,fy = euler_fluxes_2D((x->x[i,k]).(QP),(x->x[i,k]).(QM))
            tmp_nxJ = nxJ[i,k]
            tmp_nyJ = nyJ[i,k]
            tmp_LF = LFc[i,k]
            for d = 1:Nd
                if inviscid_dissp
                    rhsQ[d][i,k] = fx[d]*tmp_nxJ+fy[d]*tmp_nyJ-tmp_LF*(UP[d][i,k]-Uf[d][i,k])
                else
                    rhsQ[d][i,k] = fx[d]*tmp_nxJ+fy[d]*tmp_nyJ
                end
            end
        end
    end
end

function flux_differencing!(QF,Qh,Qrh,Qsh,rxJ,ryJ,sxJ,syJ,Nh,Nq,Nd,K)
    for k = 1:K
        rx = rxJ[1,k]
        ry = ryJ[1,k]
        sx = sxJ[1,k]
        sy = syJ[1,k]
        for j = 1:Nh # col idx
            for i = j:Nh # row idx
                if i <= Nq || j <= Nq # Skip lower right block
                    Fx,Fy = euler_fluxes_2D((x->x[i,k]).(Qh),(x->x[j,k]).(Qh))
                    Qr = Qrh[i,j]
                    Qs = Qsh[i,j]
                    for d = 1:Nd
                        update_val = 2*((rx*Qr+sx*Qs)*Fx[d]+(ry*Qr+sy*Qs)*Fy[d])
                        QF[d][i,k] += update_val
                        QF[d][j,k] -= update_val
                    end
                end
            end
        end
    end
end

"Qh = (rho,u,v,beta), while Uh = conservative vars"
function rhs_inviscid(Q,md::MeshData,ops,flux_fun,compute_rhstest,inviscid_dissp)
    @unpack rxJ,sxJ,ryJ,syJ,sJ,J,wJq = md
    @unpack nxJ,nyJ,sJ,mapP,mapB,K = md
    Qrh,Qsh,VhP,Ph,Lf,Vq = ops
    Nh,Nq = size(VhP)
    Nfq = size(Lf,2)
    Nd = length(Q)

    VU = (zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K))
    tmp = zeros(Float64,Nh,K)
    tmp2 = zeros(Float64,Nh,K)
    Qh = (zeros(Float64,Nh,K),zeros(Float64,Nh,K),zeros(Float64,Nh,K),zeros(Float64,Nh,K))
    QF = (zeros(Float64,Nh,K),zeros(Float64,Nh,K),zeros(Float64,Nh,K),zeros(Float64,Nh,K))
    QM = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
    QP = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
    Uf = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
    UP = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
    LFc = zeros(Float64,Nfq,K)
    lam = zeros(Float64,Nfq,K)
    rhsQ = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))

    # entropy var projection
    # VU = v_ufun((x->Vq*x).(Q)...)
    Q = (x->Vq*x).(Q)
    vector_norm(U) = sum((x->x.^2).(U))
    @. VU[3] = Q[2]^2+Q[3]^2 # rhoUnorm
    @. VU[4] = Q[4]-.5*VU[3]/Q[1] # rhoe
    @. VU[1] = log(0.4*VU[4]/(Q[1]^1.4)) # sU #TODO: hardcoded gamma
    @. VU[1] = (-Q[4]+VU[4]*(2.4-VU[1]))/VU[4]
    @. VU[2] = Q[2]/VU[4]
    @. VU[3] = Q[3]/VU[4]
    @. VU[4] = -Q[1]/VU[4]

    # VU = (x->VhP*x).(VU)
    Uh = (x->VhP*x).(VU)

    # Uh = u_vfun(VU...)
    @. tmp = Uh[2]^2+Uh[3]^2 #vUnorm
    @. tmp2 = (0.4/((-Uh[4])^1.4))^(1/0.4)*exp(-(1.4 - Uh[1] + tmp/(2*Uh[4]))/0.4) # rhoeV
    @. Uh[1] = tmp2.*(-Uh[4])
    @. Uh[2] = tmp2.*Uh[2]
    @. Uh[3] = tmp2.*Uh[3]
    @. Uh[4] = tmp2.*(1-tmp/(2*Uh[4]))

    # # convert to rho,u,v,beta vars
    # (rho,rhou,rhov,E) = Uh
    # beta = betafun(rho,rhou,rhov,E)
    # Qh = (rho, rhou./rho, rhov./rho, beta) # redefine Q = (rho,u,v,β)
    @. tmp = Uh[1]/(2*0.4*(Uh[4]-.5*(Uh[2]^2+Uh[3]^2)/Uh[1])) #beta
    @. Qh[1] = Uh[1]
    @. Qh[2] = Uh[2]./Uh[1]
    @. Qh[3] = Uh[3]./Uh[1]
    @. Qh[4] = tmp

    # # compute face values
    # QM = (x->x[Nq+1:Nh,:]).(Qh)
    # QP = (x->x[mapP]).(QM)
    # impose_BCs_inviscid!(QP,QM,md)
    for i = 1:Nd
        QM[i] .= Qh[i][Nq+1:Nh,:]
        QP[i] .= QM[i][mapP]
    end
    impose_BCs_inviscid!(QP,QM,md)

    if inviscid_dissp
        # simple lax friedrichs dissipation
        for i = 1:Nd
            Uf[i] .= Uh[i][Nq+1:Nh,:]
        end
        (rhoM,rhouM,rhovM,EM) = Uf
        rhoUM_n = @. (rhouM*nxJ + rhovM*nyJ)/sJ
        #lam .= abs.(wavespeed(rhoM,rhoUM_n,EM))
        @. lam  = abs(sqrt(abs(rhoUM_n/rhoM))+sqrt(1.4*0.4*(EM-.5*rhoUM_n^2/rhoM)/rhoM))
        @. LFc = .25*max(lam,lam[mapP])*sJ
    end

    for i = 1:Nd
        UP[i] .= Uf[i][mapP]
    end
    update_flux!(rhsQ,QM,QP,Uf,UP,LFc,nxJ,nyJ,mapP,Nfq,K,Nd,inviscid_dissp)
    rhsQ = (x->Lf*x).(rhsQ)

    flux_differencing!(QF,Qh,Qrh,Qsh,rxJ,ryJ,sxJ,syJ,Nh,Nq,Nd,K)
    rhsQ = (x->Ph*x).(QF).+rhsQ
    rhsQ = (x -> -x./J).(rhsQ)

    rhstest = 0.0
    if compute_rhstest
        for fld in eachindex(rhsQ)
            VUq = VU[fld][1:Nq,:]
            rhstest += sum(wJq.*VUq.*(Vq*rhsQ[fld]))
        end
    end

    return rhsQ,rhstest
end

function rhs_inviscid!(Q,md::MeshData,ops,flux_fun,compute_rhstest,inviscid_dissp,VU,Qh,QF,QM,QP,Uf,UP,rhsQ,tmp,tmp2,lam,LFc,impose_BCs_inviscid!)
    @unpack rxJ,sxJ,ryJ,syJ,sJ,J,wJq = md
    @unpack nxJ,nyJ,sJ,mapP,mapB,K = md
    Qrh,Qsh,VhP,Ph,Lf,Vq = ops
    Nh,Nq = size(VhP)
    Nfq = size(Lf,2)
    Nd = length(Q)

    fill!.(QF,0.0)

    # entropy var projection
    # VU = v_ufun((x->Vq*x).(Q)...)
    Q = (x->Vq*x).(Q)
    vector_norm(U) = sum((x->x.^2).(U))
    @. VU[3] = Q[2]^2+Q[3]^2 # rhoUnorm
    @. VU[4] = Q[4]-.5*VU[3]/Q[1] # rhoe
    @. VU[1] = log(0.4*VU[4]/(Q[1]^1.4)) # sU #TODO: hardcoded gamma
    @. VU[1] = (-Q[4]+VU[4]*(2.4-VU[1]))/VU[4]
    @. VU[2] = Q[2]/VU[4]
    @. VU[3] = Q[3]/VU[4]
    @. VU[4] = -Q[1]/VU[4]

    # VU = (x->VhP*x).(VU)
    Uh = (x->VhP*x).(VU)

    # Uh = u_vfun(VU...)
    @. tmp = Uh[2]^2+Uh[3]^2 #vUnorm
    @. tmp2 = (0.4/((-Uh[4])^1.4))^(1/0.4)*exp(-(1.4 - Uh[1] + tmp/(2*Uh[4]))/0.4) # rhoeV
    @. Uh[1] = tmp2.*(-Uh[4])
    @. Uh[2] = tmp2.*Uh[2]
    @. Uh[3] = tmp2.*Uh[3]
    @. Uh[4] = tmp2.*(1-tmp/(2*Uh[4]))

    # # convert to rho,u,v,beta vars
    # (rho,rhou,rhov,E) = Uh
    # beta = betafun(rho,rhou,rhov,E)
    # Qh = (rho, rhou./rho, rhov./rho, beta) # redefine Q = (rho,u,v,β)
    @. tmp = Uh[1]/(2*0.4*(Uh[4]-.5*(Uh[2]^2+Uh[3]^2)/Uh[1])) #beta
    @. Qh[1] = Uh[1]
    @. Qh[2] = Uh[2]./Uh[1]
    @. Qh[3] = Uh[3]./Uh[1]
    @. Qh[4] = tmp

    # # compute face values
    # QM = (x->x[Nq+1:Nh,:]).(Qh)
    # QP = (x->x[mapP]).(QM)
    # impose_BCs_inviscid!(QP,QM,md)
    for i = 1:Nd
        QM[i] .= Qh[i][Nq+1:Nh,:]
        QP[i] .= QM[i][mapP]
    end
    impose_BCs_inviscid!(QP,QM,md)

    # simple lax friedrichs dissipation
    for i = 1:Nd
        Uf[i] .= Uh[i][Nq+1:Nh,:]
    end
    (rhoM,rhouM,rhovM,EM) = Uf
    rhoUM_n = @. (rhouM*nxJ + rhovM*nyJ)/sJ
    # lam = abs.(wavespeed(rhoM,rhoUM_n,EM))
    @. lam  = abs(sqrt(abs(rhoUM_n/rhoM))+sqrt(1.4*0.4*(EM-.5*rhoUM_n^2/rhoM)/rhoM))
    @. LFc = .25*max(lam,lam[mapP])*sJ

    for i = 1:Nd
        UP[i] .= Uf[i][mapP]
    end
    update_flux!(rhsQ,QM,QP,Uf,UP,LFc,nxJ,nyJ,mapP,Nfq,K,Nd,inviscid_dissp)
    rhsQ = (x->Lf*x).(rhsQ)

    flux_differencing!(QF,Qh,Qrh,Qsh,rxJ,ryJ,sxJ,syJ,Nh,Nq,Nd,K)
    rhsQ = (x->Ph*x).(QF).+rhsQ
    rhsQ = (x -> -x./J).(rhsQ)

    rhstest = 0.0
    if compute_rhstest
        for fld in eachindex(rhsQ)
            VUq = VU[fld][1:Nq,:]
            rhstest += sum(wJq.*VUq.*(Vq*rhsQ[fld]))
        end
    end
    return rhsQ,rhstest
end

function dg_grad(Q,Qf,QP,md::MeshData,rd::RefElemData)
    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,mapP,J = md
    @unpack Dr,Ds,LIFT,Vf = rd

    Np = size(Dr,1)
    rxj,sxj,ryj,syj = (x->@view x[1:Np,:]).((rxJ,sxJ,ryJ,syJ))
    Qr = (x->Dr*x).(Q)
    Qs = (x->Ds*x).(Q)

    volx(ur,us) = @. rxj*ur + sxj*us
    voly(ur,us) = @. ryj*ur + syj*us
    surfx(uP,uf) = LIFT*(@. .5*(uP-uf)*nxJ)
    surfy(uP,uf) = LIFT*(@. .5*(uP-uf)*nyJ)
    rhsx = volx.(Qr,Qs) .+ surfx.(QP,Qf)
    rhsy = voly.(Qr,Qs) .+ surfy.(QP,Qf)
    return (x->x./J).(rhsx),(x->x./J).(rhsy)
end

function dg_grad!(Qx,Qy,Q,Qf,QP,md::MeshData,rd::RefElemData)
    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,mapP,J = md
    @unpack Dr,Ds,LIFT,Vf = rd
    Np = size(Dr,1)
    Nd = size(Qx,1)

    rxj,sxj,ryj,syj = (x->@view x[1:Np,:]).((rxJ,sxJ,ryJ,syJ))
    Qr = (x->Dr*x).(Q)
    Qs = (x->Ds*x).(Q)

    volx(ur,us) = @. rxj*ur + sxj*us
    voly(ur,us) = @. ryj*ur + syj*us
    surfx(uP,uf) = LIFT*(@. .5*(uP-uf)*nxJ)
    surfy(uP,uf) = LIFT*(@. .5*(uP-uf)*nyJ)
    rhsx = volx.(Qr,Qs) .+ surfx.(QP,Qf)
    rhsy = voly.(Qr,Qs) .+ surfy.(QP,Qf)
    for i = 1:Nd
        Qx[i] .= rhsx[i]./J
        Qy[i] .= rhsy[i]./J
    end
    #return (x->x./J).(rhsx),(x->x./J).(rhsy)
end

function dg_div(Qx,Qxf,QxP,Qy,Qyf,QyP,md::MeshData,rd::RefElemData)

    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,mapP,J = md
    @unpack Dr,Ds,LIFT,Vf = rd

    Np = size(Dr,1)
    rxj,sxj,ryj,syj = (x->@view x[1:Np,:]).((rxJ,sxJ,ryJ,syJ))
    Qxr = (x->Dr*x).(Qx)
    Qxs = (x->Ds*x).(Qx)
    Qyr = (x->Dr*x).(Qy)
    Qys = (x->Ds*x).(Qy)

    vol(uxr,uxs,uyr,uys) = @. rxj*uxr + sxj*uxs + ryj*uyr + syj*uys
    surf(uxP,uxf,uyP,uyf) = LIFT*(@. .5*((uxP-uxf)*nxJ + (uyP-uyf)*nyJ))
    rhs = vol.(Qxr,Qxs,Qyr,Qys) .+ surf.(QxP,Qxf,QyP,Qyf)

    return (x->x./J).(rhs)
end

function dg_div!(rhs,Qx,Qxf,QxP,Qy,Qyf,QyP,md::MeshData,rd::RefElemData)

    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,mapP,J = md
    @unpack Dr,Ds,LIFT,Vf = rd
    Nd = length(Qx)

    Np = size(Dr,1)
    rxj,sxj,ryj,syj = (x->@view x[1:Np,:]).((rxJ,sxJ,ryJ,syJ))
    Qxr = (x->Dr*x).(Qx)
    Qxs = (x->Ds*x).(Qx)
    Qyr = (x->Dr*x).(Qy)
    Qys = (x->Ds*x).(Qy)

    vol(uxr,uxs,uyr,uys) = @. rxj*uxr + sxj*uxs + ryj*uyr + syj*uys
    surf(uxP,uxf,uyP,uyf) = LIFT*(@. .5*((uxP-uxf)*nxJ + (uyP-uyf)*nyJ))
    #rhs = vol.(Qxr,Qxs,Qyr,Qys) .+ surf.(QxP,Qxf,QyP,Qyf)
    for i = 1:Nd
        rhs[i] .= (vol(Qxr[i],Qxs[i],Qyr[i],Qys[i]) .+ surf(QxP[i],Qxf[i],QyP[i],Qyf[i]))./J
    end
    #rhs .= (x->x./J).(rhs)
    #return (x->x./J).(rhs)
end

function init_visc_fxn(λ,μ,Pr)
    let λ=λ,μ=μ,Pr=Pr
        function viscous_matrices!(Kxx,Kxy,Kyy,v)
            v1,v2,v3,v4 = v
            inv_v4_cubed = @. 1/(v4^3)
            λ2μ = (λ+2.0*μ)
            Kxx[2,2] = inv_v4_cubed*-λ2μ*v4^2
            Kxx[2,4] = inv_v4_cubed*λ2μ*v2*v4
            Kxx[3,3] = inv_v4_cubed*-μ*v4^2
            Kxx[3,4] = inv_v4_cubed*μ*v3*v4
            Kxx[4,2] = inv_v4_cubed*λ2μ*v2*v4
            Kxx[4,3] = inv_v4_cubed*μ*v3*v4
            Kxx[4,4] = inv_v4_cubed*-(λ2μ*v2^2 + μ*v3^2 - γ*μ*v4/Pr)

            Kxy[2,3] = inv_v4_cubed*-λ*v4^2
            Kxy[2,4] = inv_v4_cubed*λ*v3*v4
            Kxy[3,2] = inv_v4_cubed*-μ*v4^2
            Kxy[3,4] = inv_v4_cubed*μ*v2*v4
            Kxy[4,2] = inv_v4_cubed*μ*v3*v4
            Kxy[4,3] = inv_v4_cubed*λ*v2*v4
            Kxy[4,4] = inv_v4_cubed*(λ+μ)*(-v2*v3)

            Kyy[2,2] = inv_v4_cubed*-μ*v4^2
            Kyy[2,4] = inv_v4_cubed*μ*v2*v4
            Kyy[3,3] = inv_v4_cubed*-λ2μ*v4^2
            Kyy[3,4] = inv_v4_cubed*λ2μ*v3*v4
            Kyy[4,2] = inv_v4_cubed*μ*v2*v4
            Kyy[4,3] = inv_v4_cubed*λ2μ*v3*v4
            Kyy[4,4] = inv_v4_cubed*-(λ2μ*v3^2 + μ*v2^2 - γ*μ*v4/Pr)
        end
        return viscous_matrices!
    end
end

function rhs_viscous(Q,md::MeshData,rd::RefElemData,viscous_dissp)
    @unpack Pq,Vq,Vf,LIFT = rd
    @unpack K,mapP,mapB,J = md
    Nd = length(Q)
    Nq = size(Vq,1)
    Np = size(Pq,1)
    Nfq = size(Vf,1)

    rhs = (zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K))
    VU = (zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K))
    VUx = (zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K))
    VUy = (zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K))
    sigma_x = (zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K))
    sigma_y = (zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K))
    penalization = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
    Kxx = MMatrix{4,4,Float64}(zeros(Nd,Nd))
    Kyy = MMatrix{4,4,Float64}(zeros(Nd,Nd))
    Kxy = MMatrix{4,4,Float64}(zeros(Nd,Nd))

    # entropy var projection
    # VU = v_ufun((x->Vq*x).(Q)...)
    Q = (x->Vq*x).(Q)
    vector_norm(U) = sum((x->x.^2).(U))
    @. VU[3] = Q[2]^2+Q[3]^2 # rhoUnorm
    @. VU[4] = Q[4]-.5*VU[3]/Q[1] # rhoe
    @. VU[1] = log(0.4*VU[4]/(Q[1]^1.4)) # sU #TODO: hardcoded gamma
    @. VU[1] = (-Q[4]+VU[4]*(2.4-VU[1]))/VU[4]
    @. VU[2] = Q[2]/VU[4]
    @. VU[3] = Q[3]/VU[4]
    @. VU[4] = -Q[1]/VU[4]
    VU = (x->Pq*x).(VU)

    # compute and interpolate to quadrature
    VUf = (x->Vf*x).(VU)
    VUP = (x->x[mapP]).(VUf)
    impose_BCs_entropyvars!(VUP,VUf,md)

    dg_grad!(VUx,VUy,VU,VUf,VUP,md,rd)
    VUx = (x->Vq*x).(VUx)
    VUy = (x->Vq*x).(VUy)
    VUq = (x->Vq*x).(VU)

    # initialize sigma_x,sigma_y = viscous rhs
    vqi = zeros(Float64,Nd)
    for e = 1:K
        # mult by matrices and perform local projections
        for i = 1:Nq
            @. vqi = getindex(VUq,i,e)
            viscous_matrices!(Kxx,Kxy,Kyy,vqi)
            for col = 2:Nd
                vxi_col = VUx[col][i,e]
                vyi_col = VUy[col][i,e]
                for row = 2:Nd
                    sigma_x[row][i,e] += Kxx[row,col]*vxi_col + Kxy[row,col]*vyi_col
                    sigma_y[row][i,e] += Kxy[col,row]*vxi_col + Kyy[row,col]*vyi_col
                end
            end
        end
    end

    rhstest = 0.0
    for fld in 1:4
        VUq = VU[fld][1:Nq,:]
        rhstest += sum(wJq.*VUx[fld].*sigma_x[fld])
        rhstest += sum(wJq.*VUy[fld].*sigma_y[fld])
    end

    sigma_x = (x->Pq*x).(sigma_x)
    sigma_y = (x->Pq*x).(sigma_y)

    sxf = (x->Vf*x).(sigma_x)
    syf = (x->Vf*x).(sigma_y)
    sxP = (x->x[mapP]).(sxf)
    syP = (x->x[mapP]).(syf)
    impose_BCs_stress!(sxP,syP,sxf,syf,VUf,md)

    if viscous_dissp
        # add penalty
        tau = .5
        dV = ((xP,x)->xP-x).(VUP,VUf)
        @. penalization[2] = -tau*dV[2]
        @. penalization[3] = -tau*dV[3]
        @. penalization[4] = -tau*dV[4]
        penalization[4][mapB] .= zeros(size(mapB))
        # avgV = ((xP,x)->1/2*(xP+x)).(VUP,VUf)
        # @. penalization[4][mapB] = -tau*(avgV[2][mapB]*dV[2][mapB]
        #                                 +avgV[3][mapB]*dV[3][mapB]
        #                                 +  dV[4][mapB]*dV[4][mapB]/2)/VUf[4][mapB]

        penalization = (x->LIFT*x).(penalization)
    end

    dg_div!(rhs,sigma_x,sxf,sxP,sigma_y,syf,syP,md,rd)

    if viscous_dissp
        return rhs .+ penalization, rhstest
    else
        return rhs, rhstest
    end
end

function rhs_viscous!(Q,md::MeshData,rd::RefElemData,Re,BCTYPE,viscous_dissp,rhs,VU,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy,impose_BCs_entropyvars!,impose_BCs_stress!,viscous_matrices!)
    @unpack Pq,Vq,Vf,LIFT = rd
    @unpack K,mapP,mapB,J,wJq = md
    Nd = length(Q)
    Nq = size(Vq,1)
    Np = size(Pq,1)
    Nfq = size(Vf,1)

    fill!.(rhs,0.0)
    fill!.(sigma_x,0.0)
    fill!.(sigma_y,0.0)

    # entropy var projection
    # VU = v_ufun((x->Vq*x).(Q)...)
    Q = (x->Vq*x).(Q)
    vector_norm(U) = sum((x->x.^2).(U))
    @. VU[3] = Q[2]^2+Q[3]^2 # rhoUnorm
    @. VU[4] = Q[4]-.5*VU[3]/Q[1] # rhoe
    @. VU[1] = log(0.4*VU[4]/(Q[1]^1.4)) # sU #TODO: hardcoded gamma
    @. VU[1] = (-Q[4]+VU[4]*(2.4-VU[1]))/VU[4]
    @. VU[2] = Q[2]/VU[4]
    @. VU[3] = Q[3]/VU[4]
    @. VU[4] = -Q[1]/VU[4]
    VU = (x->Pq*x).(VU)

    # compute and interpolate to quadrature
    VUf = (x->Vf*x).(VU)
    VUP = (x->x[mapP]).(VUf)
    impose_BCs_entropyvars!(VUP,VUf,md)

    dg_grad!(VUx,VUy,VU,VUf,VUP,md,rd)
    VUx = (x->Vq*x).(VUx)
    VUy = (x->Vq*x).(VUy)
    VUq = (x->Vq*x).(VU)

    # initialize sigma_x,sigma_y = viscous rhs
    vqi = zeros(Float64,Nd)
    for e = 1:K
        # mult by matrices and perform local projections
        for i = 1:Nq
            @. vqi = getindex(VUq,i,e)
            viscous_matrices!(Kxx,Kxy,Kyy,vqi)
            for col = 2:Nd
                vxi_col = VUx[col][i,e]
                vyi_col = VUy[col][i,e]
                for row = 2:Nd
                    sigma_x[row][i,e] += Kxx[row,col]*vxi_col + Kxy[row,col]*vyi_col
                    sigma_y[row][i,e] += Kxy[col,row]*vxi_col + Kyy[row,col]*vyi_col
                end
            end
        end
    end

    rhstest = 0.0
    for fld in 1:4
        rhstest += sum(wJq.*VUx[fld].*sigma_x[fld])
        rhstest += sum(wJq.*VUy[fld].*sigma_y[fld])
    end

    sigma_x = (x->Pq*x).(sigma_x)
    sigma_y = (x->Pq*x).(sigma_y)

    sxf = (x->Vf*x).(sigma_x)
    syf = (x->Vf*x).(sigma_y)
    sxP = (x->x[mapP]).(sxf)
    syP = (x->x[mapP]).(syf)
    impose_BCs_stress!(sxP,syP,sxf,syf,VUf,md)

    if viscous_dissp
        # add penalty
        #tau = .5
        tau = @. -1/Re/VUf[4]
        dV = ((xP,x)->xP-x).(VUP,VUf)
        avgV = ((xP,x)->1/2*(xP+x)).(VUP,VUf)
        @. penalization[2] = tau*dV[2]
        @. penalization[3] = tau*dV[3]
        @. penalization[4] = tau*dV[4]

        tau = tau[mapB]
        @. penalization[2][mapB] = tau*dV[2][mapB]
        @. penalization[3][mapB] = tau*dV[3][mapB]
        if BCTYPE == 1
            @. penalization[4][mapB] = -tau*(avgV[2][mapB]*dV[2][mapB]
                                            +avgV[3][mapB]*dV[3][mapB])/VUf[4][mapB]
        else
            @. penalization[4][mapB] = -tau*(avgV[2][mapB]*dV[2][mapB]
                                            +avgV[3][mapB]*dV[3][mapB]
                                            +  dV[4][mapB]*dV[4][mapB]/2)/VUf[4][mapB]
        end

        penalization = (x->LIFT*x).(penalization)
    end

    dg_div!(rhs,sigma_x,sxf,sxP,sigma_y,syf,syP,md,rd)

    if viscous_dissp
        return rhs .+ penalization, rhstest
    else
        return rhs, rhstest
    end
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

function rhsRK(Q,rd,md,ops,euler_fluxes,inviscid_dissp,viscous_dissp)
    rhsQ,_ = rhs_inviscid(Q,md,ops,euler_fluxes,false,inviscid_dissp)
    visc_rhsQ,visc_test = rhs_viscous(Q,md,rd,viscous_dissp)
    bcopy!(x,y) = x .= y
    bcopy!.(rhsQ, @. rhsQ + visc_rhsQ)

    let Pq=rd.Pq, Vq=rd.Vq, wJq=md.wJq
        rhstest = 0.0
        rhstest_visc = 0.0
        VU = v_ufun((x->Vq*x).(Q)...)
        VUq = (x->Vq*Pq*x).(VU)
        for field in eachindex(rhsQ)
            rhstest += sum(wJq.*VUq[field].*(Vq*rhsQ[field]))
            rhstest_visc += sum(wJq.*VUq[field].*(Vq*visc_rhsQ[field]))
        end
        rhstest_visc += visc_test
        return rhsQ,rhstest,rhstest_visc
    end
end

function rhsRK!(Q,rd,md,Re,BCTYPE,ops,euler_fluxes,inviscid_dissp,viscous_dissp,rhs,VU,Qh,QF,QM,QP,Uf,UP,rhsQ,tmp,tmp2,lam,LFc,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy,impose_BCs_inviscid!,impose_BCs_entropyvars!,impose_BCs_stress!,viscous_matrices!)
    rhsQ,_ = rhs_inviscid!(Q,md,ops,euler_fluxes,false,inviscid_dissp,VU,Qh,QF,QM,QP,Uf,UP,rhsQ,tmp,tmp2,lam,LFc,impose_BCs_inviscid!)
    visc_rhsQ,visc_test = rhs_viscous!(Q,md,rd,Re,BCTYPE,viscous_dissp,rhs,VU,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy,impose_BCs_entropyvars!,impose_BCs_stress!,viscous_matrices!)
    bcopy!(x,y) = x .= y
    bcopy!.(rhsQ, @. rhsQ + visc_rhsQ)

    let Pq=rd.Pq, Vq=rd.Vq, wJq=md.wJq
        rhstest = 0.0
        rhstest_visc = 0.0
        VU = v_ufun((x->Vq*x).(Q)...)
        VUq = (x->Vq*Pq*x).(VU)
        for field in eachindex(rhsQ)
            rhstest += sum(wJq.*VUq[field].*(Vq*rhsQ[field]))
            rhstest_visc += sum(wJq.*VUq[field].*(Vq*visc_rhsQ[field]))
        end
        rhstest_visc = rhstest_visc + visc_test
        return rhsQ,rhstest,rhstest_visc
    end
end



################
####  Main  ####
################

Re_arr = [100]#[10;100;1000]#[10;100]
N_arr = [1;2;3;4] #[2;3;4]
K1D_arr = [32]#[2;4;8;16]#[4;8;16;32;64]
err_arr = zeros(length(K1D_arr),length(N_arr),2,2,length(Re_arr))
i_disp = [true]#[true; false]
v_disp = [true]#[true; false]

for n_idx in 1:length(N_arr)
    for k_idx in 1:length(K1D_arr)
        for id in 1:2
            for vd in 1:2
                for r_idx in 1:length(Re_arr)
                    "Approximation parameters"
                    N = N_arr[n_idx] # The order of approximation
                    K1D = K1D_arr[k_idx]

                    println("========= K:", K1D, " N:", N, " \n")

                    CFL = 0.01
                    T = 1.0 # endtime
                    BCTYPE = 1 # 1 - adiabatic, 2 - Isothermal, 3 - Slip
                    TESTCASE = 1 # 1 - lid driven caivty, 2 - wave diffusion, 3 - shocktube
                    inviscid_dissp = i_disp[id]
                    viscous_dissp = v_disp[vd]

                    "Viscous parameters"
                    Re = Re_arr[r_idx]
                    mu = 1/Re
                    lambda = -2/3*mu
                    Pr = .71
                    viscous_matrices! = init_visc_fxn(lambda,mu,Pr)

                    "Time integration"
                    rk4a,rk4b,rk4c = rk45_coeffs()
                    CN = (N+1)*(N+2)/2  # estimated trace constant for CFL
                    h  = 2/K1D
                    dt = CFL * h / CN
                    Nsteps = convert(Int,ceil(T/dt))
                    dt = T/Nsteps
                    dt0 = dt

                    "Mesh related variables"
                    VX, VY, EToV = uniform_tri_mesh(K1D, K1D)

                    # initialize ref element and mesh
                    rd = init_reference_tri(N)
                    md = init_mesh((VX,VY),EToV,rd)
                    impose_BCs_inviscid!,impose_BCs_entropyvars!,impose_BCs_stress! = init_BC_funs(md,BCTYPE)

                    # construct hybridized SBP operators
                    @unpack M,Dr,Ds,Vq,Pq,Vf,wf,nrJ,nsJ = rd
                    Qr = Pq'*M*Dr*Pq
                    Qs = Pq'*M*Ds*Pq
                    Ef = Vf*Pq
                    Br = diagm(wf.*nrJ)
                    Bs = diagm(wf.*nsJ)
                    Qrh = .5*[Qr-Qr' Ef'*Br;
                             -Br*Ef  Br]
                    Qsh = .5*[Qs-Qs' Ef'*Bs;
                             -Bs*Ef  Bs]

                    Vh = [Vq;Vf]
                    Ph = M\transpose(Vh)
                    VhP = Vh*Pq

                    # make sparse skew symmetric versions of the operators"
                    # precompute union of sparse ids for Qr, Qs
                    Qrhskew = .5*(Qrh-transpose(Qrh))
                    Qshskew = .5*(Qsh-transpose(Qsh))

                    # interpolate geofacs to both vol/surf nodes
                    @unpack rxJ,sxJ,ryJ,syJ = md
                    rxJ,sxJ,ryJ,syJ = (x->Vh*x).((rxJ,sxJ,ryJ,syJ)) # interp to hybridized points
                    @pack! md = rxJ,sxJ,ryJ,syJ

                    # pack SBP operators into tuple
                    @unpack LIFT = rd
                    ops = (Qrhskew,Qshskew,VhP,Ph,LIFT,Vq)


                    #####################
                    ### Time Stepping ###
                    #####################
                    bcopy!(x,y) = x .= y

                    # define initial conditions at nodes
                    @unpack x,y = md
                    Ma = .3
                    rho = ones(size(x))
                    u = zeros(size(x))
                    v = zeros(size(x))
                    p = (1/(Ma^2*γ))*ones(size(x))

                    Q = primitive_to_conservative(rho,u,v,p)
                    Q = collect(Q) # make Q,resQ arrays of arrays for mutability
                    resQ = [zeros(size(x)) for i in eachindex(Q)]

                    @unpack Pq,Vq,Vf,LIFT = rd
                    @unpack K,mapP,mapB,J = md
                    Qrh,Qsh,VhP,Ph,Lf,Vq = ops
                    Nh,Nq = size(VhP)
                    Nfq = size(Lf,2)
                    Nd = length(Q)
                    Nq = size(Vq,1)
                    Np = size(Pq,1)
                    Nfq = size(Vf,1)
                    rhs = (zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K))
                    VU = (zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K))
                    VUx = (zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K))
                    VUy = (zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K),zeros(Float64,Np,K))
                    sigma_x = (zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K))
                    sigma_y = (zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K),zeros(Float64,Nq,K))
                    penalization = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
                    Kxx = MMatrix{4,4,Float64}(zeros(Nd,Nd))
                    Kyy = MMatrix{4,4,Float64}(zeros(Nd,Nd))
                    Kxy = MMatrix{4,4,Float64}(zeros(Nd,Nd))
                    Qh = (zeros(Float64,Nh,K),zeros(Float64,Nh,K),zeros(Float64,Nh,K),zeros(Float64,Nh,K))
                    QF = (zeros(Float64,Nh,K),zeros(Float64,Nh,K),zeros(Float64,Nh,K),zeros(Float64,Nh,K))
                    QM = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
                    QP = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
                    Uf = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
                    UP = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
                    rhsQ_tmp = (zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K),zeros(Float64,Nfq,K))
                    tmp = zeros(Float64,Nh,K)
                    tmp2 = zeros(Float64,Nh,K)
                    lam = zeros(Float64,Nfq,K)
                    LFc = zeros(Float64,Nfq,K)
                    rka,rkE,rkc = dopri45_coeffs()

                    # DOPRI storage
                    Qtmp = similar.(Q)
                    rhsQrk = ntuple(x->zero.(Q),length(rkE))

                    errEst = 0.0
                    prevErrEst = 0.0

                    t = 0.0
                    i = 0
                    interval = 5

                    dthist = Float64[dt]
                    thist = Float64[0.0]
                    errhist = Float64[0.0]
                    vischist = Float64[0.0]
                    rhstesthist = Float64[0.0]
                    wsJ = diagm(rd.wf)*md.sJ

                    prevQ = [zeros(size(x)) for i in eachindex(Q)]
                    bcopy!(x,y) = x .= y

                    #rhsQ,_,_ = rhsRK(Q,rd,md,ops,euler_fluxes_2D,inviscid_dissp,viscous_dissp)
                    rhsQ,_,_ = rhsRK!(Q,rd,md,Re,BCTYPE,ops,euler_fluxes_2D,inviscid_dissp,viscous_dissp,rhs,VU,Qh,QF,QM,QP,Uf,UP,rhsQ_tmp,tmp,tmp2,lam,LFc,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy,impose_BCs_inviscid!,impose_BCs_entropyvars!,impose_BCs_stress!,viscous_matrices!)
                    bcopy!.(rhsQrk[1],rhsQ) # initialize DOPRI rhs (FSAL property)
                    @time begin
                    while t < T
                        # DOPRI step and
                        rhstest = 0.0
                        rhstest_visc = 0.0
                        for INTRK = 2:7
                            k = zero.(Qtmp)
                            for s = 1:INTRK-1
                                bcopy!.(k, @. k + rka[INTRK,s]*rhsQrk[s])
                            end
                            bcopy!.(Qtmp, @. Q + dt*k)
                            #rhsQ,rhstest,rhstest_visc = rhsRK(Qtmp,rd,md,ops,euler_fluxes_2D,inviscid_dissp,viscous_dissp)
                            rhsQ,rhstest,rhstest_visc = rhsRK!(Qtmp,rd,md,Re,BCTYPE,ops,euler_fluxes_2D,inviscid_dissp,viscous_dissp,rhs,VU,Qh,QF,QM,QP,Uf,UP,rhsQ_tmp,tmp,tmp2,lam,LFc,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy,impose_BCs_inviscid!,impose_BCs_entropyvars!,impose_BCs_stress!,viscous_matrices!)
                            bcopy!.(rhsQrk[INTRK],rhsQ)
                        end
                        errEstVec = zero.(Qtmp)
                        for s = 1:7
                            bcopy!.(errEstVec, @. errEstVec + rkE[s]*rhsQrk[s])
                        end

                        errTol = 1e-5
                        errEst = 0.0
                        for field = 1:length(Qtmp)
                            errEstScale = @. abs(errEstVec[field]) / (errTol*(1+abs(Q[field])))
                            errEst += sum(errEstScale.^2) # hairer seminorm
                        end
                        errEst = sqrt(errEst/(length(Q[1])*4))
                        if errEst < 1.0 # if err small, accept step and update
                                bcopy!.(Q, Qtmp)
                                t += dt
                                bcopy!.(rhsQrk[1], rhsQrk[7]) # use FSAL property
                        end
                        order = 5
                        dtnew = .8*dt*(.9/errEst)^(.4/(order+1)) # P controller
                        if i > 0 # use PI controller if prevErrEst available
                                dtnew *= (prevErrEst/max(1e-14,errEst))^(.3/(order+1))
                        end
                        dt = max(min(10*dt0,dtnew),1e-9) # max/min dt
                        prevErrEst = errEst

                        push!(dthist,dt)
                        push!(thist,t)
                        push!(vischist,rhstest_visc)
                        push!(rhstesthist,rhstest)

                        i = i + 1  # number of total steps attempted
                        if i%interval==0
                            #global prevQ
                            preverr = sum(norm.(Q .- prevQ))
                            bcopy!.(prevQ, Q)
                            println("i = $i, t = $t, dt = $dtnew, errEst = $errEst, \nrhstest = $rhstest, rhstest_visc = $rhstest_visc, preverr = $preverr")
                        end
                    end

                    end


                    #############
                    ### Error ###
                    #############

                    @unpack xf,yf,mapP,mapB,nxJ,nyJ,sJ = md
                    xb,yb = (x->x[mapB]).((xf,yf))

                    lid          = mapB[findall(@. abs(yb-1) < 1e-12)]
                    wall         = mapB[findall(@. abs(yb-1) >= 1e-12)]
                    boundary     = [lid;wall]
                    xlid = xf[lid]
                    #vlid = @. exp(-1.0/(5.0*(1.0-xlid^2)))
                    #vlid = @. xlid^2-1
                    vlid = @. (1+cos(pi*xlid))/2

                    u_1 = Vf*(Q[2]./Q[1])
                    u_2 = Vf*(Q[3]./Q[1])
                    Nqf = size(Vf,1)
                    u_1_lid = u_1[lid]
                    u_1_wall = u_1[wall]
                    u_2_boundary = u_2[boundary]

                    Jf = 2.0/K1D
                    WF = repeat(wf,1,K)
                    err = sum(Jf*WF[boundary].*u_2_boundary.^2)
                         +sum(Jf*WF[wall].*u_1_wall.^2)
                         +sum(Jf*WF[lid].*(u_1_lid-vlid).^2)
                    err = sqrt(err)
                    err_arr[k_idx,n_idx,id,vd,r_idx] = err
                    @show err
                end
            end
        end
    end
end

open("err_arr.txt","w") do io
    writedlm(io,err_arr)
end
