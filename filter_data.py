import numpy as np
import qnm_filter
import qnm

spin = 0.7

file = np.loadtxt("a=0.7_proj_HRresstudy_30/proj2_Harm_re_2_2.csv",delimiter=",", dtype=float)
times = file[:,0]
RePsi4 = file[:,1]
ImPsi4 = np.loadtxt("a=0.7_proj_HRresstudy_30/proj2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

savedir = 'a=0.7_proj_HRresstudy'

#0.9 -- 
#tinit = 15.4

#0.8 --
#tinit = 10.9638

#0.7 
tinit = 8.91205

#0.6 -- 
#tinit = 7.64677

#0.5 -- 
#tinit = 6.7579

#0.4 -- 
#tinit = 6.08387

#0.3 -- 
#tinit = 5.54614

#0.2 -- 
#tinit = 5.10129

#0.1 -- 
#tinit = 4.72304

#0.0 -- 
#tinit = 4.39445

model_list_list=[]
for l in range(2,5):
    for m in range(-l,l+1):
        for n in range(0,3):
            if m>0:
                p='p' # p refers to prograde mode
            else:
                p='r' # r refers to retrograde mode
            model_list_list.extend([(l,m,n,p)])

freq_list = qnm_filter.Filter(chi=spin, mass=1,
                              model_list=model_list_list).get_freq_list

Psi4 = RePsi4 + 1j * ImPsi4

time1_cat=np.concatenate((-np.flip(times)[:-2],times))
Psi4_cat=np.concatenate((np.flip(Psi4)[:-2],Psi4))

complex_data=qnm_filter.ComplexData(Psi4_cat, index=time1_cat)
complex_data_padded=complex_data.pad_complex_data_for_fft(2,2)

def impose_filter(signalH_no_noise_padded, model_list):
    freq=signalH_no_noise_padded.fft_freq


    filter_in_freq = qnm_filter.Filter(mass=1,chi=0.7,model_list=model_list).NR_filter(freq)

    ifft = np.fft.fft(
        filter_in_freq * signalH_no_noise_padded.fft_data, norm="ortho"
    )

    return qnm_filter.ComplexData(ifft, index=signalH_no_noise_padded.index, ifo=signalH_no_noise_padded.ifo)


modelist=[(2,2,0,'p'),(2,-2,0,'r'),(3,2,0,'p'),(3,-2,0,'r'),(2,2,4,'p'),(2,2,5,'p'),(2,2,6,'p'),(2,2,7,'p'),(2,2,2,'p'),(2,2,3,'p'),(2,2,1,'p'),(2,-2,1,'r'),(2,-2,2,'r')]
filtered1=impose_filter(complex_data_padded, modelist).truncate_data(before=tinit, after=2000).pad_complex_data_for_fft(2, 2)

to_save = np.zeros((filtered1.time.size,3))
to_save[:,0] = filtered1.time
to_save[:,1] = np.real(filtered1.values)
to_save[:,2] = np.imag(filtered1.values)

#np.savetxt('%/allmodes_filtered_data.csv'%s, to_save, delimiter=',')

modelist=[(2,2,0,'p'),(2,-2,0,'r'),(3,2,0,'p'),(3,-2,0,'r'),(2,2,4,'p'),(2,2,5,'p'),(2,2,6,'p'),(2,2,7,'p'),(2,-2,1,'r'),(2,-2,2,'r')]
filtered1=impose_filter(complex_data_padded, modelist).truncate_data(before=tinit, after=2000).pad_complex_data_for_fft(2, 2)

to_save = np.zeros((filtered1.time.size,3))
to_save[:,0] = filtered1.time
to_save[:,1] = np.real(filtered1.values)
to_save[:,2] = np.imag(filtered1.values)

#np.savetxt('%/except3overtones_filtered_data.csv'%s, to_save, delimiter=',')

modelist=[(2,2,0,'p'),(2,-2,0,'r'),(3,-2,0,'r'),(2,2,4,'p'),(2,2,5,'p'),(2,2,6,'p'),(2,2,7,'p'),(2,2,2,'p'),(2,2,3,'p'),(2,2,1,'p'),(2,-2,1,'r'),(2,-2,2,'r')]
filtered1=impose_filter(complex_data_padded, modelist).truncate_data(before=tinit, after=2000).pad_complex_data_for_fft(2, 2)

to_save = np.zeros((filtered1.time.size,3))
to_save[:,0] = filtered1.time
to_save[:,1] = np.real(filtered1.values)
to_save[:,2] = np.imag(filtered1.values)

#np.savetxt('%/except320_filtered_data.csv'%s, to_save, delimiter=',')

modelist=[(2,2,0,'p'),(2,-2,0,'r'),(3,2,0,'p'),(2,2,4,'p'),(2,2,5,'p'),(2,2,6,'p'),(2,2,7,'p'),(2,2,2,'p'),(2,2,3,'p'),(2,2,1,'p'),(2,-2,1,'r'),(2,-2,2,'r')]
filtered1=impose_filter(complex_data_padded, modelist).truncate_data(before=tinit, after=2000).pad_complex_data_for_fft(2, 2)

to_save = np.zeros((filtered1.time.size,3))
to_save[:,0] = filtered1.time
to_save[:,1] = np.real(filtered1.values)
to_save[:,2] = np.imag(filtered1.values)

#np.savetxt('%/except320R_filtered_data.csv'%s, to_save, delimiter=',')

modelist=[(2,2,0,'p'),(3,2,0,'p'),(3,-2,0,'r'),(2,2,4,'p'),(2,2,5,'p'),(2,2,6,'p'),(2,2,7,'p'),(2,2,2,'p'),(2,2,3,'p'),(2,2,1,'p'),(2,-2,1,'r'),(2,-2,2,'r')]
filtered1=impose_filter(complex_data_padded, modelist).truncate_data(before=tinit, after=2000).pad_complex_data_for_fft(2, 2)

to_save = np.zeros((filtered1.time.size,3))
to_save[:,0] = filtered1.time
to_save[:,1] = np.real(filtered1.values)
to_save[:,2] = np.imag(filtered1.values)

#np.savetxt('%/except220R_filtered_data.csv'%s, to_save, delimiter=',')

modelist=[(2,2,0,'p'),(2,-2,0,'r'),(3,2,0,'p'),(3,-2,0,'r'),(2,2,4,'p'),(2,2,5,'p'),(2,2,6,'p'),(2,2,7,'p'),(2,2,2,'p'),(2,2,3,'p'),(2,2,1,'p')]
filtered1=impose_filter(complex_data_padded, modelist).truncate_data(before=tinit, after=2000).pad_complex_data_for_fft(2, 2)

to_save = np.zeros((filtered1.time.size,3))
to_save[:,0] = filtered1.time
to_save[:,1] = np.real(filtered1.values)
to_save[:,2] = np.imag(filtered1.values)

#np.savetxt('%/except22Rovertones_filtered_data.csv'%s, to_save, delimiter=',')






