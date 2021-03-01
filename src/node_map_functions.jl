"""
build_node_maps(Xf,FToF)

Intialize the connectivity table along all edges and boundary node tables of all
elements. mapM - map minus (interior). mapP - map plus (exterior).

Xf = (xf,yf,zf) and FToF is size (Nfaces*K) and FToF[face] = face neighbor

mapM,mapP are size Nfp x (Nfaces*K)

# Examples
```jldoctest
mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
```
"""
function build_node_maps(xf,yf,FToF)
    build_node_maps((xf,yf),FToF)
end
function build_node_maps(xf,yf,zf,FToF)
    build_node_maps((xf,yf,zf),FToF)
end

function build_node_maps(Xf,FToF)

    NfacesK = length(FToF)
    NODETOL = 1e-10;
    dims = length(Xf)

    # number nodes consecutively
    Nfp  = convert(Int,length(Xf[1]) / NfacesK)
    mapM = reshape(collect(1:length(Xf[1])), Nfp, NfacesK);
    mapP = copy(mapM);

    ids = collect(1:Nfp)
    for (f1,f2) in enumerate(FToF)

        # find find volume node numbers of left and right nodes
        D = zeros(Nfp,Nfp)
        for i = 1:dims
            Xfi = reshape(Xf[i],Nfp,NfacesK)
            X1i = repeat(Xfi[ids,f1],1,Nfp)
            X2i = repeat(Xfi[ids,f2],1,Nfp)
            # Compute distance matrix
            D += abs.(X1i - transpose(X2i))
        end

        refd = maximum(D[:])
        idM = map(id->id[1], findall(@. D < NODETOL*refd))
        idP = map(id->id[2], findall(@. D < NODETOL*refd))
        mapP[idM,f1] = @. idP + (f2-1)*Nfp
    end

    mapB = map(x->x[1],findall(@. mapM[:]==mapP[:]))
    return mapM,mapP,mapB
end

"""
function build_periodic_boundary_maps(xf,yf,LX,LY,mapM,mapP,mapB)

    returns mapPB
    mapP[mapB] = mapPB modifies mapP to produce a periodic node map

"""

# version which mods FToF as well
function build_periodic_boundary_maps!(xf,yf,LX,LY,NfacesTotal,mapM,mapP,mapB,FToF)

        # find boundary faces (e.g., when FToF[f] = f)
        Flist = 1:length(FToF)
        Bfaces = findall(vec(FToF) .== Flist)

        xb = xf[mapB]
        yb = yf[mapB]
        Nfp = convert(Int,length(xf)/NfacesTotal)
        Nbfaces = convert(Int,length(xb)/Nfp)
        xb = reshape(xb,Nfp,Nbfaces)
        yb = reshape(yb,Nfp,Nbfaces)

        # compute centroids of faces
        xc = vec(sum(xb,dims=1)/Nfp)
        yc = vec(sum(yb,dims=1)/Nfp)
        mapMB = reshape(mapM[mapB],Nfp,Nbfaces)
        mapPB = reshape(mapP[mapB],Nfp,Nbfaces)

        xmax = maximum(xc)
        xmin = minimum(xc)
        ymax = maximum(yc)
        ymin = minimum(yc)

        "determine which faces lie on x and y boundaries"
        NODETOL = 1e-12
        yfaces = map(x->x[1],findall(@. (@. abs(yc-ymax)<NODETOL*LY) | (@. abs(yc-ymin)<NODETOL*LY)))
        xfaces = map(x->x[1],findall(@. (@. abs(xc-xmax)<NODETOL*LX) | (@. abs(xc-xmin)<NODETOL*LX)))

        # find matches in y faces
        for i = yfaces
            for j = yfaces
                if i!=j
                    if abs(xc[i]-xc[j])<NODETOL*LX && abs(abs(yc[i]-yc[j])-LY)<NODETOL*LY
                        Xa,Xb = meshgrid(xb[:,i],xb[:,j])
                        D = @. abs(Xa-Xb)
                        ids = map(x->x[1],findall(@.D < NODETOL*LX))
                        mapPB[:,i]=mapMB[ids,j]

                        FToF[Bfaces[i]] = Bfaces[j]
                    end
                end
            end
        end

        # find matches in x faces
        for i = xfaces
            for j = xfaces
                if i!=j
                    if abs(yc[i]-yc[j])<NODETOL*LY && abs(abs(xc[i]-xc[j])-LX)<NODETOL*LX
                        Ya,Yb = meshgrid(yb[:,i],yb[:,j])
                        D = @. abs(Ya-Yb)
                        ids = map(x->x[1],findall(@.D < NODETOL*LY))
                        mapPB[:,i]=mapMB[ids,j]

                        FToF[Bfaces[i]] = Bfaces[j]
                    end
                end
            end
        end

        return mapPB[:]
end

function build_periodic_boundary_maps(xf,yf,LX,LY,NfacesTotal,mapM,mapP,mapB)

    FToF = collect(1:NfacesTotal) # dummy argument
    mapPB = build_periodic_boundary_maps!(xf,yf,LX,LY,NfacesTotal,mapM,mapP,mapB,FToF)
    return mapPB[:]

end

"3D version of build_periodic_boundary_maps"
function build_periodic_boundary_maps(xf,yf,zf,LX,LY,LZ,NfacesTotal,mapM,mapP,mapB)

    xb = xf[mapB]
    yb = yf[mapB]
    zb = zf[mapB]
    Nfp = convert(Int,length(xf)/NfacesTotal)
    Nbfaces = convert(Int,length(xb)/Nfp)
    xb = reshape(xb,Nfp,Nbfaces)
    yb = reshape(yb,Nfp,Nbfaces)
    zb = reshape(zb,Nfp,Nbfaces)

    # compute centroids of faces
    xc = vec(sum(xb,dims=1)/Nfp)
    yc = vec(sum(yb,dims=1)/Nfp)
    zc = vec(sum(zb,dims=1)/Nfp)
    mapMB = reshape(mapM[mapB],Nfp,Nbfaces)
    mapPB = reshape(mapP[mapB],Nfp,Nbfaces)

    xmax = maximum(xc);  xmin = minimum(xc)
    ymax = maximum(yc);  ymin = minimum(yc)
    zmax = maximum(zc);  zmin = minimum(zc)

    "determine which faces lie on x and y boundaries"
    NODETOL = 1e-12
    xfaces = map(x->x[1],findall(@. (@. abs(xc-xmax)<NODETOL*LX) | (@. abs(xc-xmin)<NODETOL*LX)))
    yfaces = map(x->x[1],findall(@. (@. abs(yc-ymax)<NODETOL*LY) | (@. abs(yc-ymin)<NODETOL*LY)))
    zfaces = map(x->x[1],findall(@. (@. abs(zc-zmax)<NODETOL*LZ) | (@. abs(zc-zmin)<NODETOL*LZ)))

    # find matches in x faces
    for i = xfaces
        for j = xfaces
            if i!=j
                if abs(yc[i]-yc[j])<NODETOL*LY && abs(zc[i]-zc[j])<NODETOL*LZ && abs(abs(xc[i]-xc[j])-LX)<NODETOL*LX
                    Ya,Yb = meshgrid(yb[:,i],yb[:,j])
                    Za,Zb = meshgrid(zb[:,i],zb[:,j])
                    D = @. abs(Ya-Yb) + abs(Za-Zb)
                    ids = map(x->x[1],findall(@.D < NODETOL*LY))
                    mapPB[:,i]=mapMB[ids,j]
                end
            end
        end
    end

    # find matches in y faces
    for i = yfaces
        for j = yfaces
            if i!=j
                if abs(xc[i]-xc[j])<NODETOL*LX && abs(zc[i]-zc[j])<NODETOL*LZ && abs(abs(yc[i]-yc[j])-LY)<NODETOL*LY
                    Xa,Xb = meshgrid(xb[:,i],xb[:,j])
                    Za,Zb = meshgrid(zb[:,i],zb[:,j])
                    D = @. abs(Xa-Xb) + abs(Za-Zb)
                    ids = map(x->x[1],findall(@.D < NODETOL*LX))
                    mapPB[:,i]=mapMB[ids,j]
                end
            end
        end
    end

    # find matches in y faces
    for i = zfaces
        for j = zfaces
            if i!=j
                if abs(xc[i]-xc[j])<NODETOL*LX && abs(yc[i]-yc[j])<NODETOL*LY && abs(abs(zc[i]-zc[j])-LZ)<NODETOL*LZ
                    Xa,Xb = meshgrid(xb[:,i],xb[:,j])
                    Ya,Yb = meshgrid(yb[:,i],yb[:,j])
                    D = @. abs(Xa-Xb) + abs(Ya-Yb)
                    ids = map(x->x[1],findall(@.D < NODETOL*LX))
                    mapPB[:,i]=mapMB[ids,j]
                end
            end
        end
    end

    return mapPB[:]
end
