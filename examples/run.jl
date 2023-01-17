include("../src/TeukEvolution.jl")

import .TeukEvolution as TE

@time TE.launch("examples/params_qnm.toml")
