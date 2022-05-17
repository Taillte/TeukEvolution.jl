module TeukEvolution

include("Fields.jl")
include("Io.jl")
include("Radial.jl")
include("Sphere.jl")
include("Id.jl")
include("Evolution.jl")
include("GHP.jl")
include("BackgroundNP.jl")

using .Fields: Field, Initialize_Field
import .Io
import .Radial
import .Sphere
import .Id
using .Evolution: Evo_psi4, Initialize_Evo_psi4, Evolve_psi4!
using .GHP: GHP_ops, Initialize_GHP_ops 
import .BackgroundNP

import TOML

#import MPI
#MPI.Init()
#const comm = MPI.COMM_WORLD
#const nm = MPI.Comm_size(comm) ## number of m angular numbers

#const mi = 1 + MPI.Comm_rank(comm) ## m index
#const m_val = params["m_vals"][mi] ## value of m angular number

function launch(paramfile::String)
   println("Launching run, params=$paramfile")
   params = TOML.parsefile(paramfile)

   nx = convert(Int64,params["nx"])
   ny = convert(Int64,params["ny"])
   nt = convert(Int64,params["nt"])
   ts = convert(Int64,params["ts"])

   psi_spin    = convert(Int64,params["psi_spin"])
   psi_falloff = convert(Int64,params["psi_falloff"])

   cl  = convert(Float64,params["cl"])
   cfl = convert(Float64,params["cfl"])
   bhs = convert(Float64,params["bhs"])
   bhm = convert(Float64,params["bhm"])

   outdir = params["outdir"]
   ##===================
   ## Derived parameters
   ##===================
   nm   = length(params["m_vals"])
   minr = bhm*(
      1.0 + sqrt(1.0+(bhs/bhm))*sqrt(1.0-(bhs/bhm))
     ) # horizon (uncompactified)
   maxR = (cl^2)/minr
   dr   = maxR/(nx-1.0)
   dt   = min(cfl*dr,6.0/ny^2)
   
   println("Number of threads: $(Threads.nthreads())")

   println("Setting up output directory")
   if !isdir(outdir)
      mkdir(outdir)
   else
      rm(outdir,recursive=true)
      mkdir(outdir)
   end
   println("Initializing constant fields")
   Rv = Radial.R_vals(nx, dr)
   Yv = Sphere.Y_vals(ny)
   Cv = Sphere.cos_vals(ny)
   Sv = Sphere.sin_vals(ny)
   Mv = params["m_vals"]
   time = 0.0

   ##=================
   ## Dynamical fields 
   ##=================
   println("Initializing linear psi4")
   psi4_lin_f = Initialize_Field(name="psi4_lin_f",spin=psi_spin,boost=psi_spin,falloff=psi_falloff,Mvals=Mv,nx=nx,ny=ny)
   psi4_lin_p = Initialize_Field(name="psi4_lin_p",spin=psi_spin,boost=psi_spin,falloff=psi_falloff,Mvals=Mv,nx=nx,ny=ny)
   
   println("Initializing metric reconstruction fields")
   psi3_f = Initialize_Field(name="psi3",spin=-1,boost=-1,falloff=2,Mvals=Mv,nx=nx,ny=ny)
   psi2_f = Initialize_Field(name="psi2",spin= 0,boost= 0,falloff=3,Mvals=Mv,nx=nx,ny=ny)

   la_f = Initialize_Field(name="la",spin=-2,boost=-1,falloff=1,Mvals=Mv,nx=nx,ny=ny)
   pi_f = Initialize_Field(name="pi",spin=-1,boost= 0,falloff=2,Mvals=Mv,nx=nx,ny=ny)

   muhll_f = Initialize_Field(name="muhll",spin= 0,boost=1,falloff=3,Mvals=Mv,nx=nx,ny=ny)
   hlmb_f  = Initialize_Field(name="hlmb" ,spin=-1,boost=1,falloff=2,Mvals=Mv,nx=nx,ny=ny)
   hmbmb_f = Initialize_Field(name="hmbmb",spin=-2,boost=0,falloff=1,Mvals=Mv,nx=nx,ny=ny)
  
   println("Initializing independent residuals")
   res_bianchi3_f = Initialize_Field(name="res_bianchi3",spin=-2,boost=-1,falloff=2,Mvals=Mv,nx=nx,ny=ny)
   res_bianchi2_f = Initialize_Field(name="res_bianchi2",spin=-1,boost= 0,falloff=2,Mvals=Mv,nx=nx,ny=ny)
   res_hll_f      = Initialize_Field(name="res_hll",     spin= 0,boost= 2,falloff=2,Mvals=Mv,nx=nx,ny=ny)
  
   println("Initializing 2nd order psi4")
   psi4_scd_f = Initialize_Field(name="psi4_scd_f",spin=psi_spin,boost=psi_spin,falloff=psi_falloff,Mvals=Mv,nx=nx,ny=ny)
   psi4_scd_p = Initialize_Field(name="psi4_scd_p",spin=psi_spin,boost=psi_spin,falloff=psi_falloff,Mvals=Mv,nx=nx,ny=ny)
   
   ##=======================================
   ## Fixed fields (for evolution equations) 
   ##=======================================
   println("Initializing psi4 evolution operators")
   evo_psi4 = Initialize_Evo_psi4(Rvals=Rv,Cvals=Cv,Svals=Sv,Mvals=Mv,bhm=bhm,bhs=bhs,cl=cl,spin=psi_spin)
   
   println("Initializing GHP operators")
   ghp = Initialize_GHP_ops(Rvals=Rv,Cvals=Cv,Svals=Sv,Mvals=Mv,bhm=bhm,bhs=bhs,cl=cl)
   
   println("Initializing Background NP operators")
   bkgrd_np = BackgroundNP.NP_0(Rvals=Rv,Yvals=Yv,Cvals=Cv,Svals=Sv,bhm=bhm,bhs=bhs,cl=cl)
   ##=============
   ## Initial data
   ##=============
   println("Initial data")
 
   if params["id_kind"]=="gaussian"
      for (mi,mv) in enumerate(Mv) 
         Id.set_gaussian!(psi4_lin_f[mv], psi4_lin_p[mv], 
            psi_spin,
            mv,
            params["id_l_ang"][mi],
            params["id_ru"][mi], 
            params["id_rl"][mi], 
            params["id_width"][mi],
            params["id_amp"][mi][1] + params["id_amp"][mi][2]*im,
            cl, Rv, Yv
         )
         Io.save_csv(tc=0,mv=mv,Rv=Rv,Yv=Yv,outdir=outdir,f=psi4_lin_f[mv])
      end
   elseif params["id_kind"]=="qnm"
      Id.set_qnm!()
   else
      throw(DomainError(params["id_kind"],"Unsupported `id_kind`")) 
   end
   
   ##===================
   ## Time evolution 
   ##===================
   println("Beginning evolution")
 
   for tc=1:nt
      Threads.@threads for mv in Mv
         Evolve_psi4!(psi4_lin_f[mv],psi4_lin_p[mv],evo_psi4[mv],dr,dt) 
        
         lin_f_n   = psi4_lin_f[mv].n
         lin_p_n   = psi4_lin_p[mv].n
         lin_f_np1 = psi4_lin_f[mv].np1
         lin_p_np1 = psi4_lin_p[mv].np1
         
         for j=1:ny
            for i=1:nx
               lin_f_n[i,j] = lin_f_np1[i,j] 
               lin_p_n[i,j] = lin_p_np1[i,j] 
            end
         end
      end
      
      if tc%ts==0
         println("time/bhm ", tc*dt/bhm)
         Threads.@threads for mv in Mv 
            Io.save_csv(tc=tc,mv=mv,Rv=Rv,Yv=Yv,outdir=outdir,f=psi4_lin_f[mv])
         end 
      end
   end
   println("Finished evolution")
   return nothing
end

end
