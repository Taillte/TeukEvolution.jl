import FastGaussQuadrature as FGQ
import Jacobi: jacobi

using CSV
using DataFrames

#df = CSV.read("./medpsi0/lin_f_re_2.csv", DataFrame)
df = CSV.File("./evol_med/lin_f_re_2.csv") |> Tables.matrix

println(size(df))
times_local = df[:,1]
ny = trunc(Int, df[1,3])
nx = trunc(Int, df[1,2])
nt = trunc(Int,5000)
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
    ny_here = trunc(Int, df[1,3])
    println("index = ", ny_here)

    yv, wv = FGQ.gausslegendre(ny_here)
    nl = num_l(ny_here)
    for tim = 1:nt
        f_Ylm = zeros(2,ny_here)
        for j = 1:ny_here
            for i in [2]
                 f_Ylm[i, j] = 0.0
                for k = 1:ny_here
                    f_Ylm[i, j] +=df[tim, 3+ (i-1)*k+ k ] * swal(spin, mv, lv, yv[k])*wv[k]
                end
            end
            data_Ylm[tim] +=f_Ylm[j]/ny_here
        end
        #println(f_Ylm)
        #println(" ")
    end
    return ny,times,data_Ylm
end

function compare_modes(filename1::String,filename2::String,mv::Integer)
    df = CSV.File("./$(filename1)") |> Tables.matrix
    times = df[:,1]
    data_Ylm = zeros(Float64, trunc(Int,nt))
    ny_here = trunc(Int, df[1,3])

    dftwo = CSV.File("./$(filename2)") |> Tables.matrix

    yv, wv = FGQ.gausslegendre(ny_here)
    #nl = num_l(ny_here)
    nl = 3
    lmodes = [2,3,4]
    mmodes = [2,2,2]
    println("nl = ",nl)
    #lmodes = zeros(nl)
    #data_Ylm = zeros(nl, trunc(Int,nt))

    for lv_index = 1:nl
	data_Ylm = zeros(Float64,nt)
        lv_here = lmodes[lv_index]
	mv_here = mmodes[lv_index]
	println("mode ",lv_here," ",mv_here)
        for tim = 1:nt
            f_Ylm = zeros(2,ny_here)
            for j = 1:ny_here
                for i in [2]
                    f_Ylm[i, j] = 0.0
                    for k = 1:ny_here
			f_Ylm[i, j] +=(df[tim, 3+ (i-1)*k+ k ] ^2 + dftwo[tim, 3+ (i-1)*k+ k ]^2) * swal(spin, mv_here, lv_here, yv[k])*wv[k]
                        #f_Ylm[i, j] +=df[tim, 3+ (i-1)*k+ k ] * swal(spin, mv_here, lv_here, yv[k])*wv[k]
                    end
                end
                data_Ylm[tim] +=f_Ylm[j]/ny_here
            end
            #println(f_Ylm)
            #println(" ")
        end
	save_mode( times, nt, ny_here, mv_here, lv_here, "mode_comparison2/2_Harm", data_Ylm)
    end
   
end

function test_mode_projection(lv1::Integer,mv1::Integer,lv2::Integer,mv2::Integer)
	f_Ylm = zeros(ny)
	result = 0
	yv, wv = FGQ.gausslegendre(ny)
        for j = 1:ny
                for k = 1:ny
                        f_Ylm[j] +=swal(spin, mv1, lv1, yv[k])* swal(spin, mv2, lv2, yv[k])*wv[k]
                end
                result +=f_Ylm[j]/ny
        end
        println(result)
end


#filenames = ["evol_med/lin_f_re_2.csv","evol_med/lin_f_im_2.csv","evol_high/lin_f_re_2.csv","evol_high/lin_f_im_2.csv","evol_xhigh/lin_f_re_2.csv","evol_xhigh/lin_f_im_2.csv"]
#outputs = ["evol_med/2_Harm_re", "evol_med/2_Harm_im","evol_high/2_Harm_re", "evol_high/2_Harm_im","evol_xhigh/2_Harm_re", "evol_xhigh/2_Harm_im"]

filenames = ["evol_med/lin_f_re_2.csv","evol_med/lin_f_im_2.csv","evol_high/lin_f_re_2.csv","evol_high/lin_f_im_2.csv"]
outputs = ["evol_med/2_Harm_re","evol_med/2_Harm_im","evol_high/2_Harm_re","evol_high/2_Harm_im"]

#for i =1:4
#    ny_save,times,mode = modes_from_source(filenames[i],2,2)
#    save_mode(times,nt,ny_save,2,2,outputs[i],mode)
#end

compare_modes("evol_med/lin_f_re_2.csv","evol_med/lin_f_im_2.csv",2)

test_mode_projection(2,2,2,2)

#test_mode_projection(3,3,3,3)
#test_mode_projection(3,2,3,2)

test_mode_projection(2,2,3,2)
test_mode_projection(2,2,4,2)

#test_mode_projection(2,2,3,3)
#test_mode_projection(3,3,3,2)
#test_mode_projection(2,2,4,2)

#print(times)
#println(" ")
#print(data_Ylm)


#print(df)
#println(describe(df))
#println(ncol(df))
