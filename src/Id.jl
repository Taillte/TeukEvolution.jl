module Id
using Interpolations
include("Sphere.jl")
import .Sphere: swal 

import Polynomials: ChebyshevT
import HDF5: h5read

export set_gaussian!, set_qnm!

"""
Initial gaussian initial data for psi.

function set_gaussian(
      f,
      p,
      spin::Int64,
      mv::Int64,
      l_ang::Int64,
      ru::Float64, 
      rl::Float64, 
      width::Float64,
      amp::ComplexF64,
      cl::Float64,
      Rv::Vector{Float64},
      Yv::Vector{Float64}
   )
"""
function set_gaussian!(
      f,
      p,
      spin::Int64,
      mv::Int64,
      l_ang::Int64,
      ru::Float64, 
      rl::Float64, 
      width::Float64,
      amp::ComplexF64,
      cl::Float64,
      Rv::Vector{Float64},
      Yv::Vector{Float64}
   )
   @assert f.mv == mv
   @assert p.mv == mv

   nx, ny = f.nx, f.ny

   max_val = 0.0

   for j=1:ny
      for i=1:nx
         r = (cl^2)/Rv[i] 

         bump = 0.0
         if ((r<ru) && (r>rl))
            bump = exp(-1.0*width/(r-rl))*exp(-2.0*width/(ru-r))
         end

         f.n[i,j]  = (((r-rl)/width)^2) * (((ru-r)/width)^2) * bump
         f.n[i,j] *= swal(spin,mv,l_ang,Yv[j])

         p.n[i,j] = 0.0

         max_val = max(abs(f.n[i,j]),max_val)
      end
   end

   ## rescale
  
   for j=1:ny
      for i=1:nx
         f.n[i,j] *= amp / max_val 
         
         f.np1[i,j] = f.n[i,j] 
         p.np1[i,j] = p.n[i,j] 
      end
   end
   return nothing
end

"""
Initial qnm initial data for psi, read in
from HDF5 file.

function set_qnm(
      f,
      p,
      spin::Int64,
      mv::Int64,
      l_ang::Int64,
      ru::Float64, 
      rl::Float64, 
      width::Float64,
      amp::ComplexF64,
      cl::Float64,
      Rv::Vector{Float64},
      Yv::Vector{Float64}
   )
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
    interp_linear = linear_interpolation(xs_sorted, f,extrapolation_bc=Line())
    fin = zeros(T, lenfin)
    for i = 1:lenfin
	fin[i] = interp_linear(Real(R[i]))
    end

    return fin
end


function set_qnm(
      f,
      p,
      s::Int64,
      l::Int64,
      mv::Int64,
      n::Int64,
      a::Float64,
      amp::Float64,
      Rv::Vector{Float64},
      Yv::Vector{Float64}
   )
   @assert f.mv == mv
   @assert p.mv == mv
   nx, ny = f.nx, f.ny
   
   aval = round(digits=12,a)
  
   qnmpath = dirname(pwd())*"/TeukEvolution.jl/qnm"
   #h5f = h5read(
   #      qnmpath*"/prec1024_nr$(nr)_s$(s)_m$(mv)_n$(n).h5",
   #      "[a=$(aval),l=$(l)]"
   #     )
   h5f = h5read(
         qnmpath*"/s2_m2_n0.h5",
         "[a=$(aval),l=$(l)]"
        )
   # Instead of Polynomials etc, using to_real
   #rpoly = ChebyshevT(h5f["radial_coef"])
   rpoly = _to_real(h5f["radial_coef"],Rv)
   lpoly = h5f["angular_coef"]
   lmin  = max(abs(s),abs(mv))
   # changed sum to: from 0 to length-1
   lvals = [i+lmin for i in range(0,length(lpoly)-1,step=1)]
   
   for j=1:ny
      for i=1:nx 
         #f.n[i,j]  = 1.0
	 # Changed to be i-th component instead of rpoly(Rv[i])
	 f.n[i,j]  = rpoly[i] 
         # Here took spin=s, changed sum to be over l instead of i again
	 f.n[i,j] *= sum(
            [lpoly[l]*swal(s,mv,(l-1)+lmin,Yv[j]) 
             for l in 1:length(lpoly)
            ]
         )
	if j==1
		println(rpoly[i].re)
	end
      end
   end
   println("rs = ", Rv)
   ## rescale
   max_val = maximum(abs.(f.n))
   for j=1:ny
      for i=1:nx
         f.n[i,j] *= amp / max_val    
         f.np1[i,j] = f.n[i,j] 
      end
   end
   ## p = f,t = -iωf  
   omega = h5f["omega"]
   for j=1:ny
      for i=1:nx
         p.n[i,j]   = -im*omega*f.n[i,j]
         p.np1[i,j] = p.n[i,j] 
      end
   end
   return nothing
end

end
