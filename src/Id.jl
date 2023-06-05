module Id

using Interpolations

include("Sphere.jl")
import .Sphere: swal

import Polynomials: ChebyshevT
import HDF5: h5read

export set_gaussian!, set_qnm!, BHs, BHm

"""
    set_gaussian!(
            f,
            p,
            spin::Integer,
            mv::Integer,
            l_ang::Integer,
            ru::Real, 
            rl::Real, 
            width::Real,
            amp::Complex,
            cl::Real,
            Rv::Vector{<:Real},
            Yv::Vector{<:Real}
           )::Nothing

Initial gaussian initial data for the linear evolution variable (Psi_{0,1,2,3,4}).
"""
function set_gaussian!(
    f,
    p,
    spin::Integer,
    mv::Integer,
    l_ang::Integer,
    ru::Real,
    rl::Real,
    width::Real,
    amp::Complex,
    cl::Real,
    Rv::Vector{<:Real},
    Yv::Vector{<:Real},
)::Nothing
    @assert f.mv == mv
    @assert p.mv == mv

    nx, ny = f.nx, f.ny

    max_val = 0.0

    for j = 1:ny
        for i = 1:nx
            r = (cl^2) / Rv[i]

            bump = 0.0
            if ((r < ru) && (r > rl))
                bump = exp(-1.0 * width / (r - rl)) * exp(-2.0 * width / (ru - r))
            end

            f.n[i, j] = (((r - rl) / width)^2) * (((ru - r) / width)^2) * bump
            f.n[i, j] *= swal(spin, mv, l_ang, Yv[j])

            p.n[i, j] = 0.0

            max_val = max(abs(f.n[i, j]), max_val)
        end
    end

    ## rescale

    for j = 1:ny
        for i = 1:nx
            f.n[i, j] *= amp / max_val

            f.np1[i, j] = f.n[i, j]
            p.np1[i, j] = p.n[i, j]
        end
    end
    return nothing
end

"""
    set_qnm!(
            f,
            p,
            s::Integer,
            l::Integer,
            mv::Integer,
            n::Real,
            a::Real,
            amp::Real,
            Rv::Vector{<:Real},
            Yv::Vector{<:Real}
        )::Nothing

Initial qnm initial data for psi, read in from an HDF5 file
produced from TeukolskyQNMFunctions.jl
(see https://github.com/JLRipley314/TeukolskyQNMFunctions.jl).
"""

function _to_real(c::Vector{<:Number}, R::Vector{<:Number}, T::Type{<:Number} = ComplexF64)

    N = length(c) - 1
    lenf = N+1
    lenfin = length(R)
    f = zeros(T, lenf)
    rescale_factor = T(length(c))/T(lenf)

    for i = 1:(N+1)
        n = i - 1
        for j = 1:lenf
	    f[lenf+1-j] += c[i] * (cos(n * (j - 1) * pi / T(lenf-1)))
        end
    end
    xs_ID = Array([0.5833819533586022, 0.5830193584948604, 0.5819324753715973, 0.5801240061515092, 0.5775984469740232, 0.5743620767771875, 0.5704229416872406, 0.5657908350146749, 0.5604772729065228, 0.554495465715402, 0.5478602851564955, 0.5405882273341265, 0.5326973717298423, 0.5242073362539754, 0.5151392284724271, 0.505515593129933, 0.4953603561002752, 0.4846987649027904, 0.47355732593305905, 0.4619637385638288, 0.44994682628001026, 0.4375364650189516, 0.42476350889415126, 0.41165971348706965, 0.3982576568977512, 0.3845906587505339, 0.3706926973562134, 0.3565983252366087, 0.34234258322154887, 0.3279609133318486, 0.3134890706648608, 0.2989630345016702, 0.28441891885693205, 0.26989288269374145, 0.2554210400267536, 0.2410393701370534, 0.2267836281219935, 0.21268925600238886, 0.19879129460806833, 0.18512429646085105, 0.17172223987153262, 0.15861844446445114, 0.1458454883396505, 0.13343512707859195, 0.1214182147947735, 0.10982462742554322, 0.09868318845581184, 0.08802159725832706, 0.07786636022866922, 0.06824272488617511, 0.05917461710462679, 0.05068458162875997, 0.04279372602447573, 0.035521668202106615, 0.028886487643200287, 0.022904680452079274, 0.017591118343927403, 0.012959011671361653, 0.009019876581414732, 0.005783506384578974, 0.0032579472070931037, 0.001449477987004899, 0.0003625948637418497, 0.0])	
    xs_sorted = zeros(Float64,lenf)
    for i = 1:lenf
        xs_sorted[lenf+1-i] = xs_ID[i] 
    end
    interp_linear = linear_interpolation(xs_sorted, f, extrapolation_bc=Line())
    fin = zeros(T, lenfin)
    for i = 1:lenfin
	fin[i] = interp_linear(Real(R[i]))
    end

    #return fin
    return interp_linear
end

function set_qnm!(
    f,
    p,
    spin::Int64,
    mv::Int64,
    filename::String,
    amp::Real,
    idm::Int64,
    Rv::Vector{<:Real},
    Yv::Vector{<:Real},
)::Nothing

    println("Yv = ",Yv[Int64((f.ny+1)/2)])
    println("Rv = ",Rv[f.nx])
    @assert f.mv == mv
    @assert p.mv == mv
    nx, ny = f.nx, f.ny
    # TEMP SETTING a

    physical = 0
    if physical == 1
    	aval = 0.7
        Mval = 1.0
    end
    if physical == 0
        aval =  (0.7+0.0884848)/(1+0.020245)
        Mval = 1+0.020245
    end
    lin = 2

    # qnmpath = dirname(pwd()) * "/qnm/"
    # h5f = h5read(qnmpath * filename, "[a=0.0,l=2]")
    qnmpath = "./qnm/"
    h5f = h5read(
         qnmpath * filename,
         "[a=$(aval),l=$(lin)]"
        )
    rpoly = ChebyshevT(h5f["radial_coef"])
    #rpoly = _to_real(h5f["radial_coef"],Rv)
    lpoly = h5f["angular_coef"]
    lmin = max(abs(spin), abs(mv))
    max_val = 0.0
    
    # Prints twice for m = +-2
    #println("spherical harmonic ID = ",lpoly) 
    
    # only set the field if an evolution m matches the m mode in initial data
    if mv==idm
        for j = 1:ny
            for i = 1:nx
                f.n[i, j] = rpoly( (2 * Rv[i])/maximum(Rv) -1 )
                f.n[i, j] *= sum([
                    (-1)^l * lpoly[l+1] * swal(spin, mv, l + lmin, Yv[j]) for l = 0:(length(lpoly)-1)
		    # CHANGED FOR -1^l FACTOR
                ])
                max_val = max(abs(f.n[i, j]), max_val)
		if i==1
                    println(f.n[i, j])
                end
            end
        end
	println("max val = ",max_val)
	
        for j = 1:ny
            for i = 1:nx
                #f.n[i, j] *= amp / max_val
                f.np1[i, j] = f.n[i, j]
            end
        end
	#println("rs = ", Rv)
        ## p = f,t = -iωf  
        omega = h5f["omega"]/Mval
        for j = 1:ny
            for i = 1:nx
                p.n[i, j] = -im * omega * f.n[i, j]
                p.np1[i, j] = p.n[i, j]
            end
        end

    end
    return nothing
end

function BHm_test(t::Float64)
    #return 1.0 + 0.01 * t
    BHmi = Float64(1.0)
    BHmf = 1.1
    t1 = 0.0
    t2 = 10.0
    if t<t1
	return BHmi
    end
    if t>t2
	return BHmf
    end
    if t>=t1 
	if t<=t2
	    return connect_constants(BHmi, BHmf, t1, t2, t)
	end
    end
end

function BHs_test(t::Float64)
    BHmi = Float64(1.0)
    BHmf = 1.1
    t1 = 0.0
    t2 = 10.0
    if t<t1
        return 0.7*BHmi
    end
    if t>t2
        return 0.7*BHmf
    end
    if t>=t1 
	if t<=t2
            return 0.7*connect_constants(BHmi, BHmf, t1, t2, t)
	end
    end
end

function BHm(t::Float64)
    BHmi = 1.0
    #+0.020245
    return BHmi+ 0.020245021274727132 * physical_change(t)
end

function BHs(t::Float64)
    BHsi = 0.7
    #+0.0884848
    return BHsi + 0.08848482252508819 * physical_change(t)
end


function physical_change(x::Float64)
    return (1-exp(-0.1615859254814962*x))
end

function connect_constants(f1::Float64, f2::Float64, x1::Float64, x2::Float64, x::Float64)
    # Returns C2 polynomial connecting two constant mass functions
    a = f1
    d = (10* (f1 - f2))/(x1 - x2)^3
    e = (15 *(f1 - f2))/(x1 - x2)^4
    g = (6 *(f1 - f2))/(x1 - x2)^5
    return  a  + d* (x - x1)^3 + e* (x - x1)^4 + g* (x - x1)^5
end


end
