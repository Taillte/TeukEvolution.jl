include("../src/TeukEvolution.jl")

import .TeukEvolution as TE

# write the parameter file
# "nt" => 40000
params = Dict(
    "outdir" => "a=0.7_0.5evol_reltoextremal",
    "nx" => 121,   # number of x radial grid points
    "ny" => 51,    # number of y collocation points
    "nt" => 200000, # number of time steps
    "ts" => 1,   # save every ts time steps
    "psi_spin" => -2, # spin-weight of linear evolution scalar
    "id_kind" => "qnm",
    "runtype" => "linear_field",
    "m_vals" => [-2, 2],   # m angular values
    "id_l_ang" => [2, 2],
    "id_ru" => [3.0, 3.0],
    "id_rl" => [-3.0, -3.0],
    "id_width" => [6.0, 6.0],
    "id_m" => 2,
    "id_amp" => 1.0,
    "id_spin" => 0.7, # dimensionless
    "id_filename" => "0.7_reltoextremal_s-2_m2_n0.h5",

    # format: for each m value: [real part, imaginary part]
    # "id_amp" => [[0.0, 0.0], [0.4, 0.0]],
    "cl" => 1.0, # compactification scale
    "cfl" => 0.5, # CFL number
    "bhs" => 1.0, #0.6374524336205508,#884848, # black hole spin -- sets grid
    "bhm" => 1.0, #0.9856893331997661,#20245,  #black hole mass -- sets grid
    "precision" => Float64, # precision the code is compiled at
)

@time TE.launch(params)
