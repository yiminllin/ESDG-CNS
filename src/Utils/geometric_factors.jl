"""
    geometric_factors(x, y, Dr, Ds)

Compute metrics of mappings between "real" elements and reference elements,
outward pointing normals on faces of every elements, and Jacobian.

# Examples
```jldoctest

"""
function geometric_factors(x, y, Dr, Ds)
    "Transformation and Jacobian"

    xr = Dr*x;  xs = Ds*x;
    yr = Dr*y;   ys = Ds*y;

    J = -xs.*yr + xr.*ys;
    rxJ =  ys;  sxJ = -yr;
    ryJ = -xs;  syJ =  xr;

    return rxJ, sxJ, ryJ, syJ, J
end
