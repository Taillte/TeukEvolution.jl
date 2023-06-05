module TeukEvolution

include("Fields.jl")
include("Io.jl")
include("Radial.jl")
include("Sphere.jl")
include("Id.jl")
include("Evolution.jl")
include("GHP.jl")
include("BackgroundNP.jl")
include("LinearEvolution.jl")

using Interpolations
using CSV
using DataFrames
using .Fields: Field, Initialize_Field
import .Io
import .Radial
import .Sphere
import .Id
import .BackgroundNP
using .GHP: GHP_ops, Initialize_GHP_ops
using .Evolution: Evo_lin_f, Initialize_Evo_lin_f, Evolve_lin_f!
using .LinearEvolution: Linear_evolution!, Set_independent_residuals!

import TOML

#import MPI
#MPI.Init()
#const comm = MPI.COMM_WORLD
#const nm = MPI.Comm_size(comm) ## number of m angular numbers

#const mi = 1 + MPI.Comm_rank(comm) ## m index
#const m_val = params["m_vals"][mi] ## value of m angular number

"""
    launch(params::Dict{String,Any})

    Launches time evolution of Teukolsky code, reading in paramter
    dictionary file (see examples).
"""
function launch(params::Dict{String,Any})::Nothing
    println("Launching run")
    println("params=$params")

    nx = params["nx"]
    ny = params["ny"]
    nt = params["nt"]
    ts = params["ts"]

    psi_spin = params["psi_spin"]

    runtype = params["runtype"]

    prec = params["precision"]

    cl = convert(prec, params["cl"])
    cfl = convert(prec, params["cfl"])
    bhs = convert(prec, params["bhs"])
    bhm = convert(prec, params["bhm"])
    #bhs = 
    #bhm = 
    
    outdir = params["outdir"]
    ##===================
    ## Derived parameters
    ##===================
    minr = Id.BHm(0.0) * (1 + sqrt(1 - (Id.BHs(0.0) /Id.BHm(0.0))^2) )
    #bhm * (1 + sqrt(1 - (bhs / bhm)^2) ) # horizon (uncompactified)
    maxR = 1 / minr # dt should not depend on cl
    # NOTE: Included extra point at null infinity -- see also Radial.R_vals
    dr = maxR / (nx - 1)
    #dr = maxR / (nx )
    println("maxR = ",maxR)
    dt = min(cfl * dr * bhm^2, 6 / ny^2) # make the time step roughly proportional to mass instead of inversely proportional
    # added for convergence tests
    #dt = cfl * dr * bhm^2
    println("dt = ",dt)
    println("Number of threads: $(Threads.nthreads())")

    println("Setting up output directory")
    if !isdir(outdir)
        mkdir(outdir)
    else
        rm(outdir, recursive = true)
        mkdir(outdir)
    end
    println("Initializing constant fields")
    Rv = Radial.R_vals(nx, dr)
    Yv = Sphere.Y_vals(ny)
    Cv = Sphere.cos_vals(ny)
    Sv = Sphere.sin_vals(ny)
    Mv = params["m_vals"]
    
    # Setting up mass and spin fields
    time = Float64(0.0)

    ##=================
    ## Dynamical fields 
    ##=================
    println("Initializing linear psi4")
    lin_f = Initialize_Field(
        name = "lin_f",
        spin = psi_spin,
        boost = psi_spin,
        falloff = 1,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )
    lin_p = Initialize_Field(
        name = "lin_p",
        spin = psi_spin,
        boost = psi_spin,
        falloff = 1,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )

    println("Initializing metric reconstruction fields")
    psi3_f = Initialize_Field(
        name = "psi3",
        spin = -1,
        boost = -1,
        falloff = 2,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )
    psi2_f = Initialize_Field(
        name = "psi2",
        spin = 0,
        boost = 0,
        falloff = 3,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )

    lam_f = Initialize_Field(
        name = "lam",
        spin = -2,
        boost = -1,
        falloff = 1,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )
    pi_f = Initialize_Field(
        name = "pi",
        spin = -1,
        boost = 0,
        falloff = 2,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )

    muhll_f = Initialize_Field(
        name = "muhll",
        spin = 0,
        boost = 1,
        falloff = 3,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )
    hlmb_f = Initialize_Field(
        name = "hlmb",
        spin = -1,
        boost = 1,
        falloff = 2,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )
    hmbmb_f = Initialize_Field(
        name = "hmbmb",
        spin = -2,
        boost = 0,
        falloff = 1,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )

    println("Initializing independent residuals")
    res_bianchi3_f = Initialize_Field(
        name = "res_bianchi3",
        spin = -2,
        boost = -1,
        falloff = 2,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )
    res_bianchi2_f = Initialize_Field(
        name = "res_bianchi2",
        spin = -1,
        boost = 0,
        falloff = 2,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )
    res_hll_f = Initialize_Field(
        name = "res_hll",
        spin = 0,
        boost = 2,
        falloff = 2,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )

    println("Initializing 2nd order psi4")
    scd_f = Initialize_Field(
        name = "scd_f",
        spin = psi_spin,
        boost = psi_spin,
        falloff = 1,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )
    scd_p = Initialize_Field(
        name = "scd_p",
        spin = psi_spin,
        boost = psi_spin,
        falloff = 1,
        Mvals = Mv,
        nx = nx,
        ny = ny,
    )

    ##=======================================
    ## Fixed fields (for evolution equations) 
    ##=======================================
    println("Skipped initializing psi4 evolution operators")
    "
    evo_psi4 = Initialize_Evo_lin_f(
        Rvals = Rv,
        Cvals = Cv,
        Svals = Sv,
        Mvals = Mv,
        bhm = BHm(time),
        bhs = BHs(time),
        cl = cl,
        spin = psi_spin,
    )
    "

    println("time type = ",typeof(time))
    
    println("Initializing GHP operators")
    ghp = Initialize_GHP_ops(
        Rvals = Rv,
        Cvals = Cv,
        Svals = Sv,
        Mvals = Mv,
        bhm = Id.BHm(time),
        bhs = Id.BHs(time),
        cl = cl,
    )

    println("Initializing Background NP operators")
    bkgrd_np = BackgroundNP.NP_0{prec}(
        Rvals = Rv,
        Yvals = Yv,
        Cvals = Cv,
        Svals = Sv,
        bhm = Id.BHm(time),
        bhs = Id.BHs(time),
        cl = cl,
    )
    ##=============
    ## Initial data
    ##=============
    println("Initial data")

    if params["id_kind"] == "gaussian"
        for (mi, mv) in enumerate(Mv)
            Id.set_gaussian!(
                lin_f[mv],
                lin_p[mv],
                psi_spin,
                mv,
                params["id_l_ang"][mi],
                params["id_ru"][mi],
                params["id_rl"][mi],
                params["id_width"][mi],
                params["id_amp"][mi][1] + params["id_amp"][mi][2] * im,
                cl,
                Rv,
                Yv,
            )
            Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = lin_f[mv])
            Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = lin_p[mv])
            if runtype == "reconstruction"
                Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = res_bianchi3_f[mv])
                Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = lam_f[mv])
                Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = psi3_f[mv])
            end
        end
    elseif params["id_kind"] == "qnm"
        for (mi, mv) in enumerate(Mv)
	    #println("initiating mi,mv = ",mi,mv)
            Id.set_qnm!(
                lin_f[mv],
                lin_p[mv],
                psi_spin,
                mv,
                params["id_filename"],
                params["id_amp"],
                params["id_m"],
                Rv,
                Yv,
            )
	    if mv==2
            	Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = lin_f[mv])
            	Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = lin_p[mv])
	    end
            if runtype == "reconstruction"
                Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = res_bianchi3_f[mv])
                Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = lam_f[mv])
                Io.save_csv(t = 0.0, mv = mv, outdir = outdir, f = psi3_f[mv])
            end
        end
    else
        throw(DomainError(params["id_kind"], "Unsupported `id_kind` in parameter file"))
    end

    ##===================
    ## Time evolution 
    ##===================
    println("Beginning evolution")

    Io.save_csv(t = 0.0, mv = 2, outdir = outdir, f = lin_f[2])
    for tc = 1:nt
	step1_evo_psi4 = Initialize_Evo_lin_f(
        	Rvals = Rv,
      		Cvals = Cv,
        	Svals = Sv,
        	Mvals = Mv,
        	bhm = Id.BHm((tc-1) * dt),
        	bhs = Id.BHs((tc-1) * dt),
        	cl = cl,
        	spin = psi_spin,
    	)
	step23_evo_psi4 = Initialize_Evo_lin_f(
                Rvals = Rv,
                Cvals = Cv,
                Svals = Sv,
                Mvals = Mv,
                bhm = Id.BHm((tc-0.5) * dt),
                bhs = Id.BHs((tc-0.5) * dt),
                cl = cl,
                spin = psi_spin,
        )
	step4_evo_psi4 = Initialize_Evo_lin_f(
                Rvals = Rv,
                Cvals = Cv,
                Svals = Sv,
                Mvals = Mv,
                bhm = Id.BHm((tc) * dt),
                bhs = Id.BHs((tc) * dt),
                cl = cl,
                spin = psi_spin,
        )
        if runtype == "reconstruction"
            for mv in Mv
                if mv >= 0
                    Linear_evolution!(
                        psi4_f_pm = lin_f[mv],
                        psi4_f_nm = lin_f[-mv],
                        psi4_p_pm = lin_p[mv],
                        psi4_p_nm = lin_p[-mv],
                        psi3_pm = psi3_f[mv],
                        psi3_nm = psi3_f[-mv],
                        psi2_pm = psi2_f[mv],
                        psi2_nm = psi2_f[-mv],
                        lam_pm = lam_f[mv],
                        lam_nm = lam_f[-mv],
                        pi_pm = pi_f[mv],
                        pi_nm = pi_f[-mv],
                        hmbmb_pm = hmbmb_f[mv],
                        hmbmb_nm = hmbmb_f[-mv],
                        hlmb_pm = hlmb_f[mv],
                        hlmb_nm = hlmb_f[-mv],
                        muhll_pm = muhll_f[mv],
                        muhll_nm = muhll_f[-mv],
                        Evo_pm = evo_psi4[mv],
                        Evo_nm = evo_psi4[-mv],
                        Op_pm = ghp[mv],
                        Op_nm = ghp[-mv],
                        NP = bkgrd_np,
                        R = Rv,
                        m_ang = mv,
                        bhm = bhm,
                        cl = cl,
                        dr = dr,
                        dt = dt,
                    )
                end
            end
            Threads.@threads for mv in Mv
                lin_f_n = lin_f[mv].n
                lin_f_np1 = lin_f[mv].np1
                lin_p_n = lin_p[mv].n
                lin_p_np1 = lin_p[mv].np1
                psi3_n = psi3_f[mv].n
                psi3_np1 = psi3_f[mv].np1
                psi2_n = psi2_f[mv].n
                psi2_np1 = psi2_f[mv].np1
                lam_n = lam_f[mv].n
                lam_np1 = lam_f[mv].np1
                pi_n = pi_f[mv].n
                pi_np1 = pi_f[mv].np1
                hmbmb_n = hmbmb_f[mv].n
                hmbmb_np1 = hmbmb_f[mv].np1
                hlmb_n = hlmb_f[mv].n
                hlmb_np1 = hlmb_f[mv].np1
                muhll_n = muhll_f[mv].n
                muhll_np1 = muhll_f[mv].np1
                for j = 1:ny
                    for i = 1:nx
                        lin_f_n[i, j] = lin_f_np1[i, j]
                        lin_p_n[i, j] = lin_p_np1[i, j]
                        psi3_n[i, j] = psi3_np1[i, j]
                        psi2_n[i, j] = psi2_np1[i, j]
                        lam_n[i, j] = lam_np1[i, j]
                        pi_n[i, j] = pi_np1[i, j]
                        hmbmb_n[i, j] = hmbmb_np1[i, j]
                        hlmb_n[i, j] = hlmb_np1[i, j]
                        muhll_n[i, j] = muhll_np1[i, j]
                    end
                end
            end
        elseif runtype == "linear_field"
            Threads.@threads for mv in Mv
		# HERE PASS SUBSTEP MATRICES For evolution on changing background
                Evolve_lin_f!(lin_f[mv], lin_p[mv], step1_evo_psi4[mv], step23_evo_psi4[mv], step4_evo_psi4[mv], dr, dt)

                lin_f_n = lin_f[mv].n
                lin_p_n = lin_p[mv].n
                lin_f_np1 = lin_f[mv].np1
                lin_p_np1 = lin_p[mv].np1

                for j = 1:ny
                    for i = 1:nx
                        lin_f_n[i, j] = lin_f_np1[i, j]
                        lin_p_n[i, j] = lin_p_np1[i, j]
                    end
                end
            end
        else
            throw(DomainError(runtype, "Unsupported `runtype` in parameter file"))
        end
        if tc % ts == 0
	    # Choosing units where initial black hole mass is 1
            t = tc * dt / bhm
	    #Id.BHm(Float64(0.0))
            println("time/bhm ", t)
	    #println(lin_f[2].sph_lap)
            Threads.@threads for mv in Mv
		# For some reason getting a problem w/ Threads command when accessing sph_lap
                #print(lin_fi[mv].sph_lap)
		#println("f")
		Io.save_csv(t = t, mv = mv, outdir = outdir, f = lin_f[mv])
                #println("p")
		Io.save_csv(t = t, mv = mv, outdir = outdir, f = lin_p[mv])

                if runtype == "reconstruction"
                    #=Set_independent_residuals!(
                        res_bianchi3_f=res_bianchi3_f[mv],
                        res_bianchi2_f=res_bianchi2_f[mv],
                        res_hll_f=res_hll_f[mv],
                        psi4_f=lin_f[mv],
                        psi3_f=psi3_f[mv],
                        psi2_f=psi2_f[mv],
                        lam_f=lam_f[mv],
                        pi_f=pi_f[mv],
                        hmbmb_f=hmbmb_f[mv],
                        hlmb_f=hlmb_f[mv],
                        muhll_f=muhll_f[mv],
                        Op=ghp[mv],
                        NP=bkgrd_np,
                        R=Rv,
                        m_ang=mv
                      )
                    =#
                    Io.save_csv(t = t, mv = mv, outdir = outdir, f = res_bianchi3_f[mv])
                    Io.save_csv(t = t, mv = mv, outdir = outdir, f = psi3_f[mv])
                    Io.save_csv(t = t, mv = mv, outdir = outdir, f = lam_f[mv])
                end
            end
        end
    end
    println("Finished evolution")
    return nothing
end

end
