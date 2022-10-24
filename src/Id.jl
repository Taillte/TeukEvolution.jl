module Id

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
   rpoly = ChebyshevT(h5f["radial_coef"])
   lpoly = ChebyshevT(h5f["angular_coef"])
   lmin  = max(abs(s),abs(mv))
   # Should be from 0 to length-1?
   lvals = [i+lmin for i in range(1,length(lpoly),step=1)]
   
   for j=1:ny
      for i=1:nx 
         f.n[i,j]  = rpoly(Rv[i]) 
         # Here took spin=s, changed sum to be over k instead of i again
	 #println(mv, (i-1)+lmin)
	 f.n[i,j] *= sum(
            [lpoly[k]*swal(s,mv,(k-1)+lmin,Yv[j]) 
             for k in 1:length(lpoly)
            ]
         )
      end
   end
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
