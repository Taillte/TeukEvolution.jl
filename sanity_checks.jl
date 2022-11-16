using CSV
using DataFrames

#df = CSV.read("./medpsi0/lin_f_re_2.csv", DataFrame)
df = CSV.File("./hrpsi0/lin_f_re_2.csv") |> Tables.matrix

println(size(df))
times = df[:,1]
nx = df[1,2]
ny = df[1,3]
nt = 100
println("nx = ",nx)
println("ny = ",ny)
println("nt = ",size(times))

data = zeros(Float64, trunc(Int,nt))
for i=1:nt
	# j=100
	for j=30
		data[trunc(Int, i)] = df[trunc(Int, i),trunc(Int, 3+ (ny-1)*nx +j)]
	end
end

print(times)
println(" ")
print(data)

#print(df)
#println(describe(df))
#println(ncol(df))
