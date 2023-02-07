import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

file1 = np.loadtxt("evol_med/2_Harm_re_2_2.csv",delimiter=",", dtype=float)
times = file1[:,0]
Ref_MR = file1[:,1]
Imf_MR = np.loadtxt("evol_med/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]
Ref_HR = np.loadtxt("evol_high/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
Imf_HR = np.loadtxt("evol_high/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]
Ref_XHR = np.loadtxt("evol_xhigh/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
Imf_XHR = np.loadtxt("evol_xhigh/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]
Ref_LR = np.loadtxt("evol_low/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
Imf_LR = np.loadtxt("evol_low/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

print(Ref_LR.shape)

def _fit_func(data,Amp,wR,wI,phase):
    total =  Amp * np.sin(wR*times+phase)* np.exp(wI*times) 
    return total

from scipy.optimize import curve_fit
import cmath

def omega_from_f(dataR,dataI,times):
    f = dataR + 1j*dataI
    phases = np.zeros_like(dataR)
    for i in range(times.size):
        phases[i] = cmath.phase(f[i])
    return phases

def find_period(data,times):
    saw_points = []
    saw_times = []
    saw_indices = []
    max_slope = 0
    min_slope = 0
    dtinv = (times[2]-times[1])**-1
    min_slope = (data[1]-data[0])*dtinv
    for i in range(times.size-1):
        slope = (data[i+1]-data[i])*dtinv
        if np.abs(slope) - np.abs(min_slope) > 2*np.abs(min_slope) :
            max_slope = slope
            saw_points.append(data[i])
            saw_times.append(times[i])
            saw_indices.append(i)
        else:
            min_slope = slope
    #print('saw indices = ',saw_indices)
    for i in saw_indices:
        for j in range(times.size):
            if j>i:
                #x = 1
                data[j]-=2*np.pi
    #delta_t = saw_times[2]-saw_times[1]
    #omegaR = 2 * np.pi/(delta_t)
    #print('omegaR = ',omegaR)
    #np.array([saw_points,saw_times])
    slope, intercept, r_value, p_value, std_err = stats.linregress(times,data)
    print('wr = ',slope," with err ",std_err)
    return data,slope


def find_omegaI(dataR,dataI,times):
    f = dataR + 1j*dataI
    Amp = abs(f)
    wI = np.log(Amp)/times
    slope, intercept, r_value, p_value, std_err = stats.linregress(times,np.log(Amp))
    half_num_points = int(wI.size/2)
    print('omegaI = ', slope)
    return Amp,slope

phases_XHR = omega_from_f(Ref_XHR,Imf_XHR,times)
data_XHR,slope_XHR = find_period(phases_XHR,times)
amp_XHR,wi_XHR = find_omegaI(Ref_XHR,Imf_XHR,times)

phases_HR = omega_from_f(Ref_HR,Imf_HR,times)
data_HR,slope_HR = find_period(phases_HR,times)
amp_HR,wi_HR = find_omegaI(Ref_HR,Imf_HR,times)

phases_MR = omega_from_f(Ref_MR,Imf_MR,times)
data_MR,slope_MR = find_period(phases_MR,times)
amp_MR,wi_MR = find_omegaI(Ref_MR,Imf_MR,times)

phases_LR = omega_from_f(Ref_LR,Imf_LR,times*0.5)
data_LR,slope_LR = find_period(phases_LR,times*0.5)
amp_LR,wi_LR = find_omegaI(Ref_LR,Imf_LR,times*0.5)

#phasesHR = omega_from_f(RefHR,ImfHR,timesHR)
#dataHR,slope2 = find_period(phasesHR,timesHR)
#ampHR = find_omegaI(RefHR,ImfHR,timesHR)

# convergence factor for result
#wr =  -0.5321879751341276  with err  0.0006461247390380805

print('convergence = ',(slope_MR+ 0.5326002435510183)/(slope_HR+0.5326002435510183))
print((np.log(np.abs(slope_MR - slope_HR)**-1 *np.abs(slope_XHR-slope_HR)))/np.log(2))
print((np.log(np.abs(wi_HR - wi_MR)**-1 *np.abs(wi_XHR-wi_HR)))/np.log(2))

print((np.log(np.abs(slope_MR - slope_LR)**-1 *np.abs(slope_MR-slope_HR)))/np.log(2))
print((np.log(np.abs(wi_LR - wi_MR)**-1 *np.abs(wi_MR-wi_HR)))/np.log(2))

from matplotlib.backends.backend_pdf import PdfPages

#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 12})


pp = PdfPages('./plots/phases.pdf')
plt.figure()
#plt.plot(rs_ID,cheb_ID,label='cheb reconstructed')
#plt.plot(rs_ID,ID,label='vr solved')
#plt.plot(times,modelPredictions)
#plt.plot(times,phases,'o',label='logf')
#plt.plot(times,data,'o')
#plt.plot(timesLRn,data2,'o',label='LR')
#plt.plot(timesLRn,ImfLRn,label='Imf')
#plt.plot(times,Imf)
plt.plot(0.5*times,np.log(amp_LR),label='LR')
plt.plot(times,np.log(amp_HR),label='HR')
plt.plot(times,np.log(amp_XHR),label='XHR')
plt.plot(times,np.log(amp_MR),label='MR')
#plt.plot(timesHR,np.log(ampHR),label='HR')
plt.plot(times,-5-times*0.08,label='slope wi')
plt.legend()
plt.ylabel('Phase')
plt.xlabel('t')
pp.savefig()
pp.close()

pp = PdfPages('./plots/logAmp.pdf')
plt.figure()
plt.plot(0.5*times,data_LR,'o',label='LR')
plt.plot(times,data_MR,'o',label='MR')
plt.plot(times,data_HR,'o',label='HR')
plt.plot(times,data_XHR,'o',label='XHR')
plt.legend()
plt.ylabel('Log Amp(psi0)')
plt.xlabel('t')
pp.savefig()
pp.close()

pp = PdfPages('./plots/Repsi0.pdf')
plt.figure()
plt.plot(0.5*times,Ref_LR,label='lr')
plt.plot(times,Ref_MR,label='mr')
plt.plot(times,Ref_HR,label='hr')
plt.plot(times,Ref_XHR,label='xhr')
plt.legend()
plt.ylabel('Re(psi0) - 2Y22')
plt.xlabel('t')
pp.savefig()
pp.close()


plt.show()
