from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import stats
from scipy import interpolate
from matplotlib import pyplot as plt

#times = np.loadtxt("test.csv",delimiter=",", dtype=float)[:,0]
#BHm = np.loadtxt("test.csv",delimiter=",", dtype=float)[:,1]
#BHs = np.loadtxt("test.csv",delimiter=",", dtype=float)[:,2]

file1 = np.loadtxt("evol_med_t/2_Harm_re_2_2.csv",delimiter=",", dtype=float)
times_MR = file1[:,0]
Ref_MR = file1[:,1]
Imf_MR = np.loadtxt("evol_med_t/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

times_HR = np.loadtxt("evol_high_t/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,0]
Ref_HR = np.loadtxt("evol_high_t/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
Imf_HR = np.loadtxt("evol_high_t/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

times_XLR = np.loadtxt("evol_xlow/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,0]
Ref_XLR = np.loadtxt("evol_xlow/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
Imf_XLR = np.loadtxt("evol_xlow/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

times_LR = np.loadtxt("evol_low_t/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,0]
Ref_LR = np.loadtxt("evol_low_t/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
Imf_LR = np.loadtxt("evol_low_t/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

file1 = np.loadtxt("mode_comparison_full_xrange/2_Harm_2_2.csv",delimiter=",", dtype=float)
f_22 = file1[:,1]
f_23 = np.loadtxt("mode_comparison_full_xrange/2_Harm_2_3.csv",delimiter=",", dtype=float)[:,1]
f_24 = np.loadtxt("mode_comparison_full_xrange/2_Harm_2_4.csv",delimiter=",", dtype=float)[:,1]


#Imf_34 = np.loadtxt("mode_comparison/2_Harm_im_3_4.csv",delimiter=",", dtype=float)[:,1]


print(Ref_LR.shape)
print(Ref_MR.shape)
print(Ref_HR.shape)

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

def self_convergence(dataR1,dataR2,dataR3,dataI1,dataI2,dataI3):
    nt = dataR1.size
    #nt =1
    if dataR1.size != dataR2.size or dataR1.size != dataR3.size:
        print('Error: times not aligned')
    diffR12 = np.zeros(nt-1)
    diffR23 = np.zeros(nt-1)
    diffI12 = np.zeros(nt-1)
    diffI23 = np.zeros(nt-1)
    for i in range(nt-1):
        diffR12[i] = np.abs(dataR1[i+1]-dataR2[i+1])
        diffR23[i] = np.abs(dataR2[i+1]-dataR3[i+1])
        diffI12[i] = np.abs(dataI1[i+1]-dataI2[i+1])
        diffI23[i] = np.abs(dataI3[i+1]-dataI2[i+1])
    convR = np.log(diffR12 /diffR23)/np.log(2)
    convI = np.log(diffI12 /diffI23)/np.log(2)
    print('self convergence = ',np.mean(convR)," ",np.mean(convI))
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('./plots/convergence.pdf')
    plt.figure()
    #plt.plot(times,convR,label='R')
    #plt.plot(times,convI,label='I')
    plt.plot(times_MR[1:],diffI12,label='Im(LR->MR)')
    plt.plot(times_MR[1:],((-2**4+4**4)/(2**4-1))*diffI23,label='A*Im(MR->HR)')
    plt.plot(times_MR[1:],diffR12,label='Re(LR->MR)')
    plt.plot(times_MR[1:],((-2**4+4**4)/(2**4-1))*diffR23,label='A*Re(MR->HR)')
    #plt.plot(times,dataR3,label='HR')
    plt.legend()
    plt.ylabel('Re(w-wf)')
    plt.xlabel('t')
    pp.savefig()
    pp.close()


def frequency_evolution(time,dataR,dataI):
    npoint = time.size
    wr = []
    wi_list = []
    time_list = []
    for i in range(npoint-4000):
        phases = omega_from_f(dataR[i:i+4000],dataI[i:i+4000],time[i:i+4000])
        data,slope = find_period(phases,time[i:i+4000])
        amp,wi = find_omegaI(dataR[i:i+4000],dataI[i:i+4000],time[i:i+4000])
        wr.append(slope)
        wi_list.append(wi)
        time_list.append(time[i+2000])

    pp = PdfPages('./plots/freq_evol.pdf')
    plt.figure()
    #plt.plot(times,convR,label='R')
    #plt.plot(times,convI,label='I')
    plt.plot(time_list,np.log10(np.array(wr)+0.48418178028940395),label='wr')
    plt.plot(time_list,np.log10(np.array(wi_list)+0.07344814794613463),label='wi')
    #plt.plot(times,dataR3,label='HR')
    plt.legend()
    plt.ylabel('log(w-wf)')
    plt.xlabel('t')
    pp.savefig()
    pp.close()

    print('finished tracking frequency')

def convergence_with_interp():
    yv_LR = [-0.9955569697904981, -0.9766639214595175, -0.9429745712289743, -0.8949919978782754, -0.833442628760834, -0.7592592630373576, -0.6735663684734683, -0.577662930241223, -0.473002731445715, -0.3611723058093878, -0.24386688372098844, -0.1228646926107104, 0.0, 0.1228646926107104, 0.24386688372098844, 0.3611723058093878, 0.473002731445715, 0.577662930241223, 0.6735663684734683, 0.7592592630373576, 0.833442628760834, 0.8949919978782754, 0.9429745712289743, 0.9766639214595175, 0.9955569697904981]
    angular_slice_LR = [2.9620109539167647e-8, 8.153787342387705e-7, 4.850020986092783e-6, 1.635026536260084e-5, 4.0810699624049724e-5, 8.439351146466565e-5, 0.00015319752725534175, 0.00025246680020984567, 0.0003858155000032404, 0.0005545577385330061, 0.0007572341950667718, 0.000989418459333886, 0.001243862754461075, 0.0015110051367828918, 0.0017798111399828195, 0.0020388679426781363, 0.0022775972008362536, 0.002487414403501295, 0.0026626488341369905, 0.002801057254733289, 0.0029038190124653774, 0.0029749851681427557, 0.003020455598693936, 0.0030466553183033105, 0.0030591538930230906]
    yv_MR = [-0.998866404420071, -0.9940319694320907, -0.9853540840480058, -0.972864385106692, -0.9566109552428079, -0.936656618944878, -0.9130785566557919, -0.8859679795236131, -0.8554297694299461, -0.821582070859336, -0.7845558329003993, -0.7444943022260685, -0.7015524687068222, -0.6558964656854394, -0.6077029271849502, -0.5571583045146501, -0.5044581449074642, -0.44980633497403877, -0.39341431189756515, -0.33550024541943735, -0.276288193779532, -0.21600723687604176, -0.1548905899981459, -0.09317470156008614, -0.031098338327188873, 0.031098338327188873, 0.09317470156008614, 0.1548905899981459, 0.21600723687604176, 0.276288193779532, 0.33550024541943735, 0.39341431189756515, 0.44980633497403877, 0.5044581449074642, 0.5571583045146501, 0.6077029271849502, 0.6558964656854394, 0.7015524687068222, 0.7444943022260685, 0.7845558329003993, 0.821582070859336, 0.8554297694299461, 0.8859679795236131, 0.9130785566557919, 0.936656618944878, 0.9566109552428079, 0.972864385106692, 0.9853540840480058, 0.9940319694320907, 0.998866404420071]
    angular_slice_MR = [1.7923128927178684e-9, 4.9650636282329424e-8, 2.987247215169929e-7, 1.0239978545683006e-6, 2.613138190823197e-6, 5.556241088214156e-6, 1.0432643276571963e-5, 1.78950022535039e-5, 2.8650894294687452e-5, 4.344224930014449e-5, 6.302300986414049e-5, 8.813547340967848e-5, 0.00011948584691588386, 0.0001577196101397145, 0.00020339734132601643, 0.00025697170483493, 0.0003187663286050806, 0.0003889573069414963, 0.00046755804713102296, 0.0005544081339532743, 0.000649166812358728, 0.0007513115844924008, 0.0008601422832038644, 0.0009747908220375459, 0.0010942366347230059, 0.0012173276103022656, 0.001342806109869487, 0.0014693394256119416, 0.0015955538220534563, 0.001720071093982039, 0.00184154639708457, 0.0019587059676549395, 0.002070383258333135, 0.0021755519878296626, 0.002273354642077942, 0.002363125077400708, 0.002444404064438304, 0.0025169468717899397, 0.0025807223128549214, 0.0026359030556078744, 0.002682847405889932, 0.002722073199357977, 0.002754224851802948, 0.002780034997182501, 0.0028002824629568294, 0.0028157485709444565, 0.0028271738913143094, 0.0028352176089577723, 0.00284042161323752, 0.0028431818558712126]
    yv_HR = [-0.9997137267734413, -0.9984919506395958, -0.9962951347331251, -0.9931249370374434, -0.9889843952429918, -0.983877540706057, -0.9778093584869183, -0.9707857757637063, -0.9628136542558156, -0.9539007829254917, -0.944055870136256, -0.9332885350430795, -0.921609298145334, -0.9090295709825297, -0.895561644970727, -0.8812186793850184, -0.8660146884971647, -0.8499645278795913, -0.8330838798884008, -0.8153892383391763, -0.7968978923903145, -0.7776279096494956, -0.7575981185197073, -0.7368280898020207, -0.7153381175730565, -0.693149199355802, -0.6702830156031411, -0.6467619085141293, -0.6226088602037078, -0.5978474702471789, -0.5725019326213813, -0.5465970120650943, -0.5201580198817632, -0.493210789208191, -0.4657816497733582, -0.43789740217203155, -0.40958529167830166, -0.38087298162462996, -0.3517885263724217, -0.32236034390052926, -0.292617188038472, -0.26258812037150336, -0.23230248184497404, -0.20178986409573646, -0.1710800805386034, -0.1402031372361141, -0.10918920358006115, -0.07806858281343654, -0.046871682421591974, -0.015628984421543188, 0.015628984421543188, 0.046871682421591974, 0.07806858281343654, 0.10918920358006115, 0.1402031372361141, 0.1710800805386034, 0.20178986409573646, 0.23230248184497404, 0.26258812037150336, 0.292617188038472, 0.32236034390052926, 0.3517885263724217, 0.38087298162462996, 0.40958529167830166, 0.43789740217203155, 0.4657816497733582, 0.493210789208191, 0.5201580198817632, 0.5465970120650943, 0.5725019326213813, 0.5978474702471789, 0.6226088602037078, 0.6467619085141293, 0.6702830156031411, 0.693149199355802, 0.7153381175730565, 0.7368280898020207, 0.7575981185197073, 0.7776279096494956, 0.7968978923903145, 0.8153892383391763, 0.8330838798884008, 0.8499645278795913, 0.8660146884971647, 0.8812186793850184, 0.895561644970727, 0.9090295709825297, 0.921609298145334, 0.9332885350430795, 0.944055870136256, 0.9539007829254917, 0.9628136542558156, 0.9707857757637063, 0.9778093584869183, 0.983877540706057, 0.9889843952429918, 0.9931249370374434, 0.9962951347331251, 0.9984919506395958, 0.9997137267734413]
    angular_slice_HR = [1.1029713758640485e-10, 3.060375717181062e-9, 1.8466384884082954e-8, 6.356764884941657e-8, 1.6311624562055977e-7, 3.4921456720207833e-7, 6.611038469008745e-7, 1.1449042621872858e-6, 1.8533074265403314e-6, 2.8452224375253043e-6, 4.185376607539799e-6, 5.9438723516514955e-6, 8.195701999978998e-6, 1.1020222359937881e-5, 1.4500591332905648e-5, 1.8723168821184317e-5, 2.377688465883826e-5, 2.9752576775892047e-5, 3.674230255295517e-5, 4.4838626985422165e-5, 5.413389155403563e-5, 6.471946780478684e-5, 7.668500009365596e-5, 9.01176420244179e-5, 0.0001051012915151461, 0.00012171582959462118, 0.00014003636816879356, 0.0001601325123074633, 0.00018206764258708592, 0.00020589822320708892, 0.00023167314169410242, 0.00025943308594382925, 0.000289209964334087, 0.0003210263746109066, 0.0003548951270656064, 0.0003908188273014499, 0.0004287895237310467, 0.0004687884245203356, 0.0005107856883677279, 0.0005547402931129853, 0.0006005999856037676, 0.0006483013157537844, 0.0006977697570680006, 0.0007489199152998977, 0.000801655826150321, 0.0008558713421663539, 0.0009114506082537659, 0.0009682686242743237, 0.0010261918924505506, 0.0010850791463663386, 0.0011447821574432407, 0.0012051466139479188, 0.0012660130666126825, 0.001327217934162512, 0.0013885945611649437, 0.001449974319824061, 0.0015111877466277013, 0.0015720657040397704, 0.001632440556864822, 0.0016921473523694836, 0.001751024992877764, 0.0018089173892098798, 0.0018656745831678043, 0.001921153827259216, 0.0019752206098663087, 0.0020277496143689274, 0.0020786256010587225, 0.002127744201218513, 0.002175012613444975, 0.002220350193070618, 0.0022636889265635744, 0.0023049737838449335, 0.0023441629427293485, 0.0023812278810378907, 0.00241615333337996, 0.002448937111170742, 0.002479589786014324, 0.002508134238297516, 0.0025346050744819285, 0.00255904791828471, 0.0025815185826111205, 0.0026020821306916575, 0.0026208118364710337, 0.0026377880556821983, 0.0026530970204171663, 0.0026668295711594083, 0.0026790798412499274, 0.0026899439096267253, 0.002699518438236163, 0.0027078993109950766, 0.0027151802912986868, 0.002721451715051696, 0.0027267992359137345, 0.0027313026388636435, 0.0027350347375609037, 0.002738060369954071, 0.0027404355057567826, 0.002742206479582535, 0.002743409372967668, 0.002744069724259361]
    fLR = interpolate.make_interp_spline(yv_LR, angular_slice_LR, k=5, t=None, bc_type=None, axis=0, check_finite=True)
    fMR = interpolate.make_interp_spline(yv_MR, angular_slice_MR, k=5, t=None, bc_type=None, axis=0, check_finite=True)
    fHR = interpolate.make_interp_spline(yv_HR, angular_slice_HR, k=5, t=None, bc_type=None, axis=0, check_finite=True)
    print(fLR(0),fMR(0),fHR(0))
    x = -0.995
    print(np.log((fHR(x)-fMR(x))/(fMR(x)-fLR(x)))/np.log(2))
    x = 0.2
    print(np.log((fHR(x)-fMR(x))/(fMR(x)-fLR(x)))/np.log(2))
    x = 0.4
    print(np.log((fHR(x)-fMR(x))/(fMR(x)-fLR(x)))/np.log(2))
    #self_convergence(dataR1,dataR2,dataR3,dataI1,dataI2,dataI3)

convergence_with_interp()

print(times_LR)
#print(times - times_HR)
#print(times_HR - times_XHR)

#phases_XLR = omega_from_f(Ref_XLR,Imf_XLR,times_XLR)
#data_XLR,slope_XLR = find_period(phases_XLR,times_XLR)
#amp_XLR,wi_XLR = find_omegaI(Ref_XLR,Imf_XLR,times_XLR)

phases_HR = omega_from_f(Ref_HR,Imf_HR,times_HR)
data_HR,slope_HR = find_period(phases_HR,times_HR)
amp_HR,wi_HR = find_omegaI(Ref_HR,Imf_HR,times_HR)

phases_MR = omega_from_f(Ref_MR,Imf_MR,times_MR)
data_MR,slope_MR = find_period(phases_MR,times_MR)
amp_MR,wi_MR = find_omegaI(Ref_MR,Imf_MR,times_MR)

#phases_MR = omega_from_f(Ref_MR[300:],Imf_MR[300:],times[300:])
#data_MR,slope_MR = find_period(phases_MR,times[300:])
#amp_MR,wi_MR = find_omegaI(Ref_MR[300:],Imf_MR[300:],times[300:])

#frequency_evolution(times_LR,Ref_LR,Imf_LR)

self_convergence(Ref_LR,Ref_MR,Ref_HR,Imf_LR,Imf_MR,Imf_HR)

#phases_MR = omega_from_f(Ref_MR,Imf_MR,times_MR)
#data_MR,slope_MR = find_period(phases_MR,times_MR)
#amp_MR,wi_MR = find_omegaI(Ref_MR,Imf_MR,times_MR)

phases_LR = omega_from_f(Ref_LR,Imf_LR,times_LR)
data_LR,slope_LR = find_period(phases_LR,times_LR)
amp_LR,wi_LR = find_omegaI(Ref_LR,Imf_LR,times_LR)

#phases_HR = omega_from_f(Ref_HR,Imf_HR,times_HR)
#data_HR,slope_HR = find_period(phases_HR,times_HR)
#amp_HR = find_omegaI(Ref_HR,Imf_HR,times_HR)

# convergence factor for result
#wr =  -0.5321879751341276  with err  0.0006461247390380805

#print('convergence = ',np.log((slope_MR+ 0.5326002435510183)/(slope_HR+0.5326002435510183))/np.log(2))
#print('convergence = ',np.log((wi_MR+0.0807929627407481 )/(wi_HR+0.0807929627407481))/np.log(2))

#print((np.log(np.abs(slope_HR - slope_MR) /np.abs(slope_HR-slope_XHR)))/np.log(2))
#print((np.log(np.abs(wi_HR - wi_MR)/np.abs(wi_XHR-wi_HR)))/np.log(2))

#print((np.log(np.abs(slope_MR - slope_HR)/ np.abs(slope_XHR-slope_HR)))/np.log(2))
#print((np.log(np.abs(wi_HR - wi_MR)**-1 *np.abs(wi_XHR-wi_HR)))/np.log(2))

#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 12})


pp = PdfPages('./plots/logAmp.pdf')
plt.figure()
#plt.plot(rs_ID,cheb_ID,label='cheb reconstructed')
#plt.plot(rs_ID,ID,label='vr solved')
#plt.plot(times,modelPredictions)
#plt.plot(times,phases,'o',label='logf')
#plt.plot(times,data,'o')
#plt.plot(timesLRn,data2,'o',label='LR')
#plt.plot(timesLRn,ImfLRn,label='Imf')
#plt.plot(times,Imf)
plt.plot(times_LR,np.log(amp_LR),label='LR')
plt.plot(times_HR,np.log(amp_HR),label='HR')
#plt.plot(times_XLR,np.log(amp_XLR),label='XLR')
plt.plot(times_MR,np.log(amp_MR),label='MR')
#plt.plot(timesHR,np.log(ampHR),label='HR')
#plt.plot(times,-5-times*0.08,label='slope wi')
plt.legend()
plt.ylabel('Log Amp')
plt.xlabel('t')
pp.savefig()
pp.close()

pp = PdfPages('./plots/phases.pdf')
plt.figure()
plt.plot(times_LR,data_LR,label='LR')
plt.plot(times_MR,data_MR,label='MR')
plt.plot(times_HR,data_HR,label='HR')
#plt.plot(times_XLR,data_XLR,'o',label='XLR')
plt.legend()
plt.ylabel('Phase')
plt.xlabel('t')
pp.savefig()
pp.close()

pp = PdfPages('./plots/Repsi0.pdf')
plt.figure()
plt.plot(times_LR,Ref_LR,label='lr')
plt.plot(times_MR,Ref_MR,label='mr')
plt.plot(times_HR,Ref_HR,label='hr')
#plt.plot(times_XLR,Ref_XLR,label='xlr')
plt.legend()
plt.ylabel('Re(psi0) - 2Y22')
plt.xlabel('t')
pp.savefig()
pp.close()

#pp = PdfPages('./plots/test.pdf')
#plt.figure()
#plt.plot(times[:1000],BHm[:1000],label='mass')
#plt.plot(times[:1000],BHs[:1000],label='spin')
#plt.legend()
#plt.ylabel('BH param')
#plt.xlabel('t')
#pp.savefig()
#pp.close()


#pp = PdfPages('./plots/Mode_amps.pdf')
#plt.figure()
#plt.plot(times,np.log(amp_22),label='(2,2)')
#plt.plot(times,np.log(amp_23),label='(3,2)')
#plt.plot(times,np.log(amp_24),label='(4,2)')
#plt.plot(times,np.log(amp_33),label='(3,3)')
#plt.plot(times,np.log(amp_34),label='(4,3)')
#plt.plot(times,-5-times*0.08,label='slope wi')
#plt.legend()
#plt.ylabel('Log Amp(l,m)')
#plt.xlabel('t')
#pp.savefig()
#pp.close()


pp = PdfPages('./plots/Mode_amps_xrange.pdf')
plt.figure()
plt.plot(f_22,label='(2,2)')
plt.plot(f_23,label='(3,2)')
plt.plot(f_24,label='(4,2)')
#plt.plot(times,Ref_33,label='(3,3)')
#plt.plot(times,Ref_34,label='(4,3)')
plt.legend()
plt.ylabel('A(l,m)')
plt.xlabel('x')
pp.savefig()
pp.close()


plt.show()
