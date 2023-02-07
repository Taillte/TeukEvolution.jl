import FastGaussQuadrature as FGQ
import Jacobi: jacobi

using CSV
using DataFrames

#df = CSV.read("./medpsi0/lin_f_re_2.csv", DataFrame)
df = CSV.File("./evol_high/lin_f_re_2.csv") |> Tables.matrix

println(size(df))
times_local = df[:,1]
ny = trunc(Int, df[1,3])
nx = trunc(Int, df[1,2])
nt = trunc(Int,10000)
spin = 2
println("nx = ",nx)
println("ny = ",ny)
println("nt = ",size(times_local))

"""Gives number of ls that can be consistent with other mode numbers ?"""
function num_l(ny::Integer,max_s::Integer=2,max_m::Integer=6)::Integer
    @assert ny > (2 * max_s + 2 * max_m) + 4
    return ny - 2 * max_s - 2 * max_m
end

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



""" Transformation matrix taking R -> Ylm space"""
yv, wv = FGQ.gausslegendre(trunc(Int, ny))
nl = num_l(ny)
"""Choosing here single m mode m_ang instead of ?summing """
m_ang = 2
lmin = max(abs(spin), abs(m_ang))
"""
for j = 1:ny
    for i = 1:ny
        for k = 1:nl
            l = k - 1 + lmin
            lap[j, i] -= swal(spin, m_ang, l, yv[i]) 
        end
        lap[j, i] *= wv[i]
    end
end
"""

"""Now taking data from saved files

data = zeros(Float64, trunc(Int,nt))
for i=1:nt
	#for j=1:ny
	for j in [7]
		data[trunc(Int, i)] += df[trunc(Int, i),trunc(Int,3 + ny +j)]
	end
	data[trunc(Int, i)] /= ny
end
"""

""" 
Now want to take data but projected into harmonic space

data_Ylm = zeros(Float64, trunc(Int,nt))

for tim = 1:nt
    f_Ylm = zeros(2,ny)
    for j = 1:ny
        for i in [2]
            f_Ylm[i, j] = 0.0
            for k = 1:ny
                f_Ylm[i, j] +=df[tim, 3+ (i-1)*k+ k ] * swal(2, 2, 2, yv[k])*wv[k]
		#* lap[k, j]
            end
        end
	data_Ylm[tim] +=f_Ylm[j]
    end
    #println(f_Ylm)
    #println(" ")
end
"""

"""Testing with pure harmonic -- s=l=m=2

test_harm = zeros(ny)
for i = 1:ny
    test_harm[i] = swal(2, 2, 2, yv[i])
end

f_Ylm = zeros(ny)
for j = 1:ny
    f_Ylm[j] = 0.0
    for k = 1:ny
        f_Ylm[j] += swal(2, 2, 3, yv[k]) * swal(2, 2, 2, yv[k])*wv[k]
    end
end
#println("modes = ",f_Ylm)
"""

function save_mode( times::Vector{Union{Missing, Float64}}, nt::Int64, ny::Int64, mv::Int64, lv::Int64, filename::String, f::Vector{<:Real})
    nx = 2
 
    for tim in 2:nt
	t = times[tim]
	ft = f[tim]
	#println(t,ft,"   ")
        open("./$(filename)_$(mv)_$(lv).csv", "a") do out
            write(out, "$t,")
	    write(out, "$ft")
	    write(out, "\n")
        end
    end
    return nothing
end

function modes_from_source(filename::String,mv::Integer,lv::Integer)
    df = CSV.File("./$(filename)") |> Tables.matrix
    times = df[:,1]
    data_Ylm = zeros(Float64, trunc(Int,nt))

    for tim = 1:nt
        f_Ylm = zeros(2,ny)
        for j = 1:ny
            for i in [2]
                 f_Ylm[i, j] = 0.0
                for k = 1:ny
                    f_Ylm[i, j] +=df[tim, 3+ (i-1)*k+ k ] * swal(spin, mv, lv, yv[k])*wv[k]
                end
            end
            data_Ylm[tim] +=f_Ylm[j]
        end
        #println(f_Ylm)
        #println(" ")
    end
    return times,data_Ylm
end

#filenames = ["evol_med/lin_f_re_2.csv","evol_med/lin_f_im_2.csv","evol_high/lin_f_re_2.csv","evol_high/lin_f_im_2.csv","evol_xhigh/lin_f_re_2.csv","evol_xhigh/lin_f_im_2.csv"]
#outputs = ["evol_med/2_Harm_re", "evol_med/2_Harm_im","evol_high/2_Harm_re", "evol_high/2_Harm_im","evol_xhigh/2_Harm_re", "evol_xhigh/2_Harm_im"]

filenames = ["evol_low/lin_f_re_2.csv","evol_low/lin_f_im_2.csv"]
outputs = ["evol_low/2_Harm_re","evol_low/2_Harm_im"]

for i =1:2
    times,mode = modes_from_source(filenames[i],2,2)
    save_mode(times,nt,ny,2,2,outputs[i],mode)
end

#print(times)
#println(" ")
#print(data_Ylm)


#print(df)
#println(describe(df))
#println(ncol(df))
