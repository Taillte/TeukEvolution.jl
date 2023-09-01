include("src/Sphere.jl")
import HDF5: h5read
import FastGaussQuadrature as FGQ
import Jacobi: jacobi

ny = 48

""" spin-weighted associated Legendre function Y^s_{lm}(y)"""
function swal(spin::Integer, m_ang::Integer, l_ang::Integer, y::Real)::Real
    @assert l_ang >= abs(m_ang)

    al = abs(m_ang - spin)
    be = abs(m_ang + spin)
    @assert((al + be) % 2 == 0)
    n = l_ang - (al + be) / 2

    if n < 0
        return convert(Float64, 0)
    end

    norm = sqrt(
        (2 * n + al + be + 1) * (2^(-al - be - 1.0)) * factorial(n + al + be) /
        factorial(n + al) * factorial(n) / factorial(n + be),
    )
    norm *= (-1)^(max(m_ang, -spin))

    return norm * ((1 - y)^(al / 2.0)) * ((1 + y)^(be / 2.0)) * jacobi(y, n, al, be)
end


# Gauss-Legendre points (y=-cosθ) over the interval [-1,1]
Yv, w = FGQ.gausslegendre(ny) 

a = 0.7
lin = 2

h5f = h5read(
         "qnm/0.7_samegrids2_m2_n0.h5",
         "[a=$(a),l=$(lin)]"
        )
lpoly = h5f["angular_coef"]
lmin = 2
max_val = 0.0
mv = 2
spin = 2

S = ones(Complex, ny)

for j = 1:ny
    S[j] *= sum([
        (-1)^l * lpoly[l+1] * swal(spin, mv, l + lmin, Yv[j]) for l = 0:(length(lpoly)-1)
        # CHANGED FOR -1^l FACTOR
	])
end

intsum = 0.0

rp = 1 + (1-a^2)^0.5

for j=1:ny
    global intsum += w[j] * S[j] * conj(S[j]) *( 4 * rp^2 /( rp^2 + a^2 * (-Yv[j]) )^2 )^2
end

println(intsum)







