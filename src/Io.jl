module Io 

export save_csv

"""
Save the real and imaginary parts of level np1
"""
function save_csv(;
      tc::Int64,
      mv::Int64,
      Rv::Vector{Float64},
      Yv::Vector{Float64},
      outdir::String,
      f)
      
   nx, ny = f.nx, f.ny 

   open("$(outdir)/$(f.name)_$(mv)_re_$tc.csv","a") do out
      write(out,"R,Y,val\n")
      for i=1:nx
         for j=1:ny
            if abs(f.np1[i,j].re)>1e-16
               write(out,"$(Rv[i]),$(Yv[j]),$(f.np1[i,j].re)\n")
            end
         end
      end
   end

   open("$(outdir)/$(f.name)_$(mv)_im_$tc.csv","a") do out
      write(out,"R,Y,val\n")
      for i=1:nx
         for j=1:ny
            if abs(f.np1[i,j].im)>1e-16
               write(out,"$(Rv[i]),$(Yv[j]),$(f.np1[i,j].im)\n")
            end
         end
      end
   end

   return nothing
end

end
