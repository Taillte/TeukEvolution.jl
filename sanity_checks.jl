import FastGaussQuadrature as FGQ
import Jacobi: jacobi
import Polynomials: ChebyshevT
import HDF5: h5read

using Interpolations
using CSV
using DataFrames

#df = CSV.read("./medpsi0/lin_f_re_2.csv", DataFrame)
#df = CSV.File("./evol_low/lin_f_re_2.csv") |> Tables.matrix

#println(size(df))
#times_local = df[:,1]
#ny = trunc(Int, df[1,3])
#nx = trunc(Int, df[1,2])
nt = trunc(Int,1001)
spin = trunc(Int,-2)
#println("nx = ",nx)
#println("ny = ",ny)
#println("nt = ",size(times_local))

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
    n = trunc(Int,l_ang - (al + be) / 2)

    if n < 0
        return convert(Float64, 0)
    end
    
    #println(n," ",al," ",be)
    #println(typeof(n)," ",typeof(al)," ",typeof(be))
    norm = sqrt(
		(2 * n + al + be + 1) * (2^(-al - be - 1.0)) * factorial(big(n + al + be)) /
		factorial(big(n + al)) * factorial(n) / factorial(big(n + be)),
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
    for tim in 1:nt
	t = times[tim]
	ft = f[tim]
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
    println("df shape = ",size(df))
    println("time10 = ",times[10])
    data_Ylm = zeros(Float64, trunc(Int,nt))
    ny_here = trunc(Int, df[1,3])
    println("index = ", ny_here)
    
    yv_t,wv_t = FGQ.gausslegendre(25)
    yv, wv = FGQ.gausslegendre(ny_here)
    #angular_slice =  Vector{Float64}(df[1000, 3+ (2-1)*ny_here + 1 : 3+ (2-1)*ny_here + ny_here])
    #println("t = ",times[1000])
    #nl = num_l(ny_here)
    for tim = 1:nt
        f_Ylm = zeros(2)
	# NOTE i=1 -- null inf, i=2 -- horizon
	for i in [1]
            #f_Ylm[i] = 0.0
            for k = 1:ny_here
                data_Ylm[tim] +=df[tim, 3+ (i-1)*ny_here+ k ] * swal(spin, mv, lv, yv[k])*wv[k]
            end
	    #println(yv[Int64((ny_here+1)/2)])
	    #data_Ylm[tim] =df[tim, 3+ (i-1)*ny_here + Int64((ny_here+1)/2)]
            #df[tim, 3+ (i-1)*ny_here + 1 + Int64((ny_here-1)/4)]
	    #angular_slice =  Vector{Float64}(df[tim, 3+ (i-1)*ny_here + 1:3+ (i-1)*ny_here + ny_here])
	    #println("ang_slice = ",angular_slice)
	    #println("yv = ",yv)
	    # Note: no higher than linear interp for not evenly separated points
	    #interp = Interpolations.linear_interpolation(yv, angular_slice, extrapolation_bc=Line())
            #data_Ylm[tim] = interp(yv_t[10]) 
	    #df[tim, 3+ (i-1)*ny_here + 1]
        end
    end
    return ny_here,times,data_Ylm
end

function save_all_modes( times::Vector{Union{Missing, Float64}}, nt::Int64, ny::Int64, mv::Int64, filename::String, f::Matrix{<:Real})
    nx = 2
    for tim in 1:nt
        t = times[tim]
        open("./$(filename)_$(mv).csv", "a") do out
	    write(out, "$t, ")
	end
	for mode in 1:18
	    ft = f[tim,mode]
	    open("./$(filename)_$(mv).csv", "a") do out
		write(out, "$ft, ")
	    end
	end
	ft = f[tim,19]
        open("./$(filename)_$(mv).csv", "a") do out
            write(out, "$ft")
	    write(out, "\n")
        end
    end
    return nothing
end

function all_modes_from_source(filename::String,mv::Integer)
    df = CSV.File("./$(filename)") |> Tables.matrix
    times = df[:,1]
    data_Ylm = zeros(Float64, trunc(Int,nt),19)
    ny_here = trunc(Int, df[1,3])
    ells = [ 2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    yv_t,wv_t = FGQ.gausslegendre(25)
    yv, wv = FGQ.gausslegendre(ny_here)
    for tim = 1:nt
        # NOTE i=1 -- null inf, i=2 -- horizon
        i = 1
	for ell =1:19
            for k = 1:ny_here
                data_Ylm[tim,ell] +=df[tim, 3+ (i-1)*ny_here+ k ] * swal(spin, mv, ells[ell], yv[k])*wv[k]
            end
        end
    end
    return ny_here,times,data_Ylm
end


function compare_modes(filename1::String,filename2::String,mv::Integer)
    df = CSV.File("./$(filename1)") |> Tables.matrix
    times = df[:,1]
    data_Ylm = zeros(Float64, trunc(Int,nt))
    ny_here = trunc(Int, df[1,3])
    nx_here = trunc(Int, df[1,2])

    dftwo = CSV.File("./$(filename2)") |> Tables.matrix

    yv, wv = FGQ.gausslegendre(ny_here)
    #nl = num_l(ny_here)
    nl = 3
    lmodes = [2,3,4]
    mmodes = [2,2,2]    

    println("times = ",times)
    #lmodes = zeros(nl)
    #data_Ylm = zeros(nl, trunc(Int,nt))

    """ NEED TO INTERPOLATE ONTO yv"""
	
    xs_sorted = [-0.9982377097105593, -0.990726238699457, -0.9772599499837743, -0.9579168192137917, -0.9328128082786765, -0.9020988069688742, -0.8659595032122595, -0.8246122308333117, -0.7783056514265194, -0.7273182551899271, -0.6719566846141796, -0.6125538896679802, -0.5494671250951282, -0.4830758016861787, -0.413779204371605, -0.3419940908257585, -0.2681521850072537, -0.1926975807013711, -0.11608407067525521, -0.03877241750605083, 0.03877241750605083, 0.11608407067525521, 0.1926975807013711, 0.2681521850072537, 0.3419940908257585, 0.413779204371605, 0.4830758016861787, 0.5494671250951282, 0.6125538896679802, 0.6719566846141796, 0.7273182551899271, 0.7783056514265194, 0.8246122308333117, 0.8659595032122595, 0.9020988069688742, 0.9328128082786765, 0.9579168192137917, 0.9772599499837743, 0.990726238699457, 0.9982377097105593]
    for lv_index = 1:nl
	data_Ylm = zeros(nx_here)
        lv_here = lmodes[lv_index]
	mv_here = mmodes[lv_index]
	println("mode ",lv_here," ",mv_here)
        for tim in [1]
            f_Ylm = zeros(nx_here,ny_here)
	    #f_Ylm = zeros(ny_here)
            for i =1:nx_here
		f = zeros(ny_here)
		for j=1:ny_here
			f[j] = df[tim, 3+ (i-1)*ny_here+ j ]
			#(df[tim, 3+ (i-1)*ny_here+ j ] ^2 + dftwo[tim, 3+ (i-1)*ny_here+ j ]^2)
		end
		interp_linear = linear_interpolation(xs_sorted, f, extrapolation_bc=Line())
	
		for j = 1:ny_here
                    f_Ylm[i,j] = 0.0
                    for k = 1:40
			f_Ylm[i,j] += interp_linear(yv[k]) * swal(spin, mv_here, lv_here, yv[k])*wv[k]
                        #f_Ylm[i, j] +=df[tim, 3+ (i-1)*k+ k ] * swal(spin, mv_here, lv_here, yv[k])*wv[k]
                    end
		    data_Ylm[i] += f_Ylm[i,j]/(ny_here)
                end
                #data_Ylm[tim] +=f_Ylm[j]/ny_here
            end
            #println(f_Ylm)
            #println(" ")
        end
	#save_mode( times, nt, ny_here, mv_here, lv_here, "mode_comparison_full_xrange/2_Harm", data_Ylm)
        save_mode( times, nx_here, ny_here, mv_here, lv_here, "mode_comparison_full_xrange/2_Harm", data_Ylm)
    end
   
end

function test_mode_projection(ny::Integer,lv1::Integer,mv1::Integer,lv2::Integer,mv2::Integer)
	result = 0
	yv, wv = FGQ.gausslegendre(ny)
        for k = 1:ny
        	result +=swal(spin, mv1, lv1, yv[k])* swal(spin, mv2, lv2, yv[k])*wv[k]
        end
        println(result)
end

function test_mode_projection_data(lv::Integer,mv::Integer)
	qnmpath = "./qnm/"
	aval = 0.7
	l = 2
	nx_here = 124
	ny_here = 24

	h5f = h5read(
        	 qnmpath *"s2_m2_n0.h5",
      	   	"[a=$(aval),l=$(l)]"
      		  )
	rpoly = ChebyshevT(h5f["radial_coef"])
	lpoly = h5f["angular_coef"]
	lmin = 2
	
	nYv = 40
	println("Yv = ",size(Rv))
        
	yv, wv = FGQ.gausslegendre(ny_here)
	#println("lpoly = ",lpoly)
	fn = zeros(nx_here,nYv)

	for j = 1:ny_here
    		for i = 1:nx_here
			fn[i, j] = 1
			#rpoly( (2 * Rv[i])/maximum(Rv) -1 )
        		fn[i, j] *= sum([
           		 (-1)^l * lpoly[l+1].re * swal(2, 2, l + lmin, yv[j]) for l = 0:(length(lpoly)-1)
			#fn[i,j] = swal(2, 2, 2, yv[j])
            		])
       		end
   	end

	f_Ylm = zeros(ny_here)
        result = zeros(nx_here)

	for i = 1:nx_here
		f_Ylm = 0
                for k = 1:ny_here
                       	f_Ylm +=fn[i,k] * swal(spin, mv, lv, yv[k])*wv[k]
               	end
               	result[i] = f_Ylm
	end
	println(" ")
        println(result)
end

function testing_read_in()
	aval = 0.7
        lin = 2
	h5f = h5read(
                 "./qnm/s2_m2_n0.h5",
                "[a=$(aval),l=$(lin)]"
                  )
        rpoly = ChebyshevT(h5f["radial_coef"])
        lpoly = h5f["angular_coef"]
	
	Yv = [-0.9982377097105593, -0.990726238699457, -0.9772599499837743, -0.9579168192137917, -0.9328128082786765, -0.9020988069688742, -0.8659595032122595, -0.8246122308333117, -0.7783056514265194, -0.7273182551899271, -0.6719566846141796, -0.6125538896679802, -0.5494671250951282, -0.4830758016861787, -0.413779204371605, -0.3419940908257585, -0.2681521850072537, -0.1926975807013711, -0.11608407067525521, -0.03877241750605083, 0.03877241750605083, 0.11608407067525521, 0.1926975807013711, 0.2681521850072537, 0.3419940908257585, 0.413779204371605, 0.4830758016861787, 0.5494671250951282, 0.6125538896679802, 0.6719566846141796, 0.7273182551899271, 0.7783056514265194, 0.8246122308333117, 0.8659595032122595, 0.9020988069688742, 0.9328128082786765, 0.9579168192137917, 0.9772599499837743, 0.990726238699457, 0.9982377097105593]
	Rv = [0.009723032555976702, 0.019446065111953403, 0.029169097667930105, 0.03889213022390681, 0.04861516277988351, 0.05833819533586021, 0.06806122789183691, 0.07778426044781361, 0.08750729300379032, 0.09723032555976702, 0.10695335811574372, 0.11667639067172042, 0.12639942322769712, 0.13612245578367382, 0.14584548833965053, 0.15556852089562723, 0.16529155345160393, 0.17501458600758063, 0.18473761856355733, 0.19446065111953403, 0.20418368367551074, 0.21390671623148744, 0.22362974878746414, 0.23335278134344084, 0.24307581389941754, 0.25279884645539424, 0.2625218790113709, 0.27224491156734765, 0.2819679441233244, 0.29169097667930105, 0.3014140092352777, 0.31113704179125445, 0.3208600743472312, 0.33058310690320786, 0.34030613945918453, 0.35002917201516126, 0.359752204571138, 0.36947523712711466, 0.37919826968309134, 0.38892130223906807, 0.3986443347950448, 0.40836736735102147, 0.41809039990699814, 0.4278134324629749, 0.4375364650189516, 0.4472594975749283, 0.45698253013090495, 0.4667055626868817, 0.4764285952428584, 0.4861516277988351, 0.49587466035481176, 0.5055976929107885, 0.5153207254667652, 0.5250437580227418, 0.5347667905787186, 0.5444898231346953, 0.554212855690672, 0.5639358882466488, 0.5736589208026254, 0.5833819533586021]

	nx_here = 60
	ny_here = 40
	
	fn = zeros(ComplexF64,nx_here,ny_here)
	f_final = zeros(nx_here,ny_here)
	max_val = 0

	for j = 1:ny_here
                for i = 1:nx_here
                        fn[i, j] = rpoly( (2 * Rv[i])/maximum(Rv) -1 )
                        fn[i, j] *= sum([
                         (-1)^l * lpoly[l+1] * swal(2, 2, l + lmin, Yv[j]) for l = 0:(length(lpoly)-1)
                        ])
			max_val = max(abs(fn[i, j]), max_val)
                end
        end
	for j = 1:ny_here
            for i = 1:nx_here
                fn[i, j] *= 1 / max_val
            end
        end
	csv_file = zeros(nx_here,ny_here)
	df = CSV.File("./evol_full_xrange/lin_f_re_2.csv") |> Tables.matrix
	for i = 1:nx_here
		for j=1:ny_here
			csv_file[i,j] = df[1, 3+ (i-1)*ny_here+ j ]
			f_final[i,j] = fn[i,j].re - df[1, 3+ (i-1)*j+ j ]
		end
	end
	
	times = df[:,1]
	
	#println("time = ",times[1])
	#println("hdf = ",fn[1,:])
	println("csv = ",csv_file[1,:])
	println("csv = ",csv_file[3,:])
	#println(f_final)
	
end


function test_mode_convergence()
    mv = 2
    lv = 2
    spin = 2
    data_Ylm = zeros(10)
    points = [12,24,48,96,Int64(24*4),Int64(24*5),Int64(24*6),Int64(24*7),Int64(24*8),Int64(24*1000)]/3 .+1
    for res = 1:10
	n_points= points[res]
        data = zeros(Float64, trunc(Int,n_points))
	yv, wv = FGQ.gausslegendre(Int(n_points))
        for i= 1:Int(n_points)
	    data[i] = exp(yv[i])*swal(2, 2, 2, yv[i])
	    #exp(acos(yv[i])) * swal(2, 2, 2, yv[i])
	end
405355	#println(data[n_points])
        for k = 1:Int(n_points)
            data_Ylm[res] +=data[k] * swal(spin, mv, lv, yv[k])*wv[k]
        end
	test_mode_projection(Int(n_points),2,2,3,2)
    end
    return data_Ylm
end

#println(test_mode_convergence())

#filenames = ["evol_low/lin_f_re_2.csv","evol_low/lin_f_im_2.csv","evol_med/lin_f_re_2.csv","evol_med/lin_f_im_2.csv","evol_high/lin_f_re_2.csv","evol_high/lin_f_im_2.csv"]
#outputs = ["evol_low/2_Harm_re", "evol_low/2_Harm_im","evol_med/2_Harm_re","evol_med/2_Harm_im","evol_high/2_Harm_re","evol_high/2_Harm_im"]

#filenames = ["evol_physical_low/lin_f_re_2.csv","evol_physical_low/lin_f_im_2.csv","evol_physical_med/lin_f_re_2.csv","evol_physical_med/lin_f_im_2.csv","evol_physical_high/lin_f_re_2.csv","evol_physical_high/lin_f_im_2.csv"]
#outputs = ["evol_physical_low/2_Harm_re","evol_physical_low/2_Harm_im","evol_physical_med/2_Harm_re","evol_physical_med/2_Harm_im","evol_physical_high/2_Harm_re","evol_physical_high/2_Harm_im"]

filenames = ["a=0.0_proj_reltoextremal/lin_f_re_0.csv","a=0.0_proj_reltoextremal/lin_f_im_0.csv"]
outputs = ["a=0.0_proj_reltoextremal/2_Harm_re","a=0.0_proj_reltoextremal/2_Harm_im"]

#filenames = ["evol_med/lin_f_re_2.csv","evol_med/lin_f_im_2.csv","evol_high/lin_f_re_2.csv","evol_high/lin_f_im_2.csv","evol_xhigh/lin_f_re_2.csv","evol_xhigh/lin_f_im_2.csv"]
#outputs = ["evol_med/2_Harm_re","evol_med/2_Harm_im","evol_high/2_Harm_re","evol_high/2_Harm_im","evol_xhigh/2_Harm_re","evol_xhigh/2_Harm_im"]


#yv, wv = FGQ.gausslegendre(25)
#println(yv[1])
#yv, wv = FGQ.gausslegendre(49)
#println(yv[1])

#nt = 28724
#nt = 5611
#nt = 15000
nt = 87079
#nt = 190583

for i =1:2
    #ny_save,times,mode = all_modes_from_source(filenames[i],2)
    #save_all_modes(times,nt,ny_save,2,outputs[i],mode)
    #print(filenames[i])
    ny_save,times,mode = modes_from_source(filenames[i],2,2)
    save_mode(times,nt,ny_save,2,2,outputs[i],mode)
    #ny_save,times,mode = modes_from_source(filenames[i],2,3)
    #save_mode(times,nt,ny_save,2,3,outputs[i],mode)
    #ny_save,times,mode = modes_from_source(filenames[i],2,4)
    #save_mode(times,nt,ny_save,2,4,outputs[i],mode)
    #ny_save,times,mode = modes_from_source(filenames[i],2,5)
    #save_mode(times,nt,ny_save,2,5,outputs[i],mode)
end

#testing_read_in()

#compare_modes("evol_full_xrange/lin_f_re_2.csv","evol_full_xrange/lin_f_im_2.csv",2)

#test_mode_projection(2,2,3,3)

#include("src/Id.jl")

#for i in 1:nt
#    t = times_local[i]
#    bhs = Id.BHs(t)
#    bhm = Id.BHm(t)
#    open("./test.csv", "a") do out
#        write(out, "$t,")
#        write(out, "$bhm,")
#        write(out, "$bhs")
#        write(out, "\n")
#    end
#
#end

#print(times)
#println(" ")
#print(data_Ylm)


#print(df)
#println(describe(df))
#println(ncol(df))
