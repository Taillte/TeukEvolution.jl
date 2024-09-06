from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from scipy import stats
from scipy import interpolate
from matplotlib import pyplot as plt
#scipy.integrate.simpson
from scipy.integrate import simpson as simps
#times = np.loadtxt("test.csv",delimiter=",", dtype=float)[:,0]
#BHm = np.loadtxt("test.csv",delimiter=",", dtype=float)[:,1]
#BHs = np.loadtxt("test.csv",delimiter=",", dtype=float)[:,2]

#file1 = np.loadtxt("a=0.4_3evol_30/filtered_data_LR/except220_filtered_data.csv",delimiter=",", dtype=float)
#times_LR = file1[:,0]
#Ref_LR = file1[:,1]
#Imf_LR = file1[:,2]

#file1 = np.loadtxt("spher_3evolinit_30/filtered_changing_freq_model/except3overtones_filtered_data.csv",delimiter=",", dtype=float)
#times_LR = file1[:,0]
#Ref_LR = file1[:,1]
#Imf_LR = file1[:,2]

file1 = np.loadtxt("a=0.7_evol_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)
times_LR = file1[:,0]
Ref_LR = file1[:,1]
Imf_LR = np.loadtxt("a=0.7_evol_reltoextremal/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

#file1 = np.loadtxt("a=0.7_proj_LRresstudy/2_Harm_re_2_2.csv",delimiter=",", dtype=float)
#times_LR = file1[:,0]
#Ref_LR = file1[:,1]
#Imf_LR = np.loadtxt("a=0.7_proj_LRresstudy/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

#file1 = np.loadtxt("a=0.7_evolinit_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)
#times_evoinit = file1[:,0]
#Ref_evoinit = file1[:,1]
#Imf_evoinit = np.loadtxt("a=0.7_evolinit_reltoextremal/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

#file1 = np.loadtxt("a=0.7_proj_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)
#times_proj = file1[:,0]
#Ref_proj = file1[:,1]
#Imf_proj = np.loadtxt("a=0.7_proj_reltoextremal/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

#file1 = np.loadtxt("a=0.4_3evol_30_MR/filtered_data_LR/except220_filtered_data.csv",delimiter=",", dtype=float)
#times_MR = file1[:,0]
#Ref_MR = file1[:,1]
#Imf_MR = file1[:,2]
#print(np.mean(np.abs(Ref_LR)),np.mean(np.abs(Ref_MR)))

#file1 = np.loadtxt("a=0.4_3evol_30_HR/filtered_data_LR/except220_filtered_data.csv",delimiter=",", dtype=float)
#times_HR = file1[:,0]
#Ref_HR = file1[:,1]
#Imf_HR = file1[:,2]

#file1 = np.loadtxt("a=0.4_3evol_30_MR/2_Harm_re_LR_2_2.csv",delimiter=",", dtype=float)
#times_MR = file1[:,0]
#Ref_MR = file1[:,1]
#Imf_MR = np.loadtxt("a=0.4_3evol_30_MR/2_Harm_im_LR_2_2.csv",delimiter=",", dtype=float)[:,1]

times_2 = np.loadtxt("a=0.7_2evol_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,0]
Ref_2 = np.loadtxt("a=0.7_2evol_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
Imf_2 = np.loadtxt("a=0.7_2evol_reltoextremal/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

times_3 = np.loadtxt("a=0.7_3evol_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,0]
Ref_3 = np.loadtxt("a=0.7_3evol_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
Imf_3 = np.loadtxt("a=0.7_3evol_reltoextremal/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

#times_HR = np.loadtxt("a=0.4_3evol_30_HR/2_Harm_re_LR_2_2.csv",delimiter=",", dtype=float)[:,0]
#Ref_HR = np.loadtxt("a=0.4_3evol_30_HR/2_Harm_re_LR_2_2.csv",delimiter=",", dtype=float)[:,1]
#Imf_HR = np.loadtxt("a=0.4_3evol_30_HR/2_Harm_im_LR_2_2.csv",delimiter=",", dtype=float)[:,1]

#times_XLR = np.loadtxt("a=0.7_lrevol_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,0]
#Ref_XLR = np.loadtxt("a=0.7_lrevol_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
#Imf_XLR = np.loadtxt("a=0.7_lrevol_reltoextremal/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

#times_l3 = np.loadtxt("a=0.7_3evol_reltoextremal/2_Harm_re_2_3.csv",delimiter=",", dtype=float)[:,0]
#Ref_l3 = np.loadtxt("a=0.7_3evol_reltoextremal/2_Harm_re_2_3.csv",delimiter=",", dtype=float)[:,1]
#Imf_l3 = np.loadtxt("a=0.7_3evol_reltoextremal/2_Harm_im_2_3.csv",delimiter=",", dtype=float)[:,1]

times_fix = np.loadtxt("a=0.7_fixed_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,0]
Ref_fix = np.loadtxt("a=0.7_fixed_reltoextremal/2_Harm_re_2_2.csv",delimiter=",", dtype=float)[:,1]
Imf_fix = np.loadtxt("a=0.7_fixed_reltoextremal/2_Harm_im_2_2.csv",delimiter=",", dtype=float)[:,1]

#file1 = np.loadtxt("mode_comparison_full_xrange/2_Harm_2_2.csv",delimiter=",", dtype=float)
#f_22 = file1[:,1]
#f_23 = np.loadtxt("mode_comparison_full_xrange/2_Harm_2_3.csv",delimiter=",", dtype=float)[:,1]
#f_24 = np.loadtxt("mode_comparison_full_xrange/2_Harm_2_4.csv",delimiter=",", dtype=float)[:,1]


#Imf_34 = np.loadtxt("mode_comparison/2_Harm_im_3_4.csv",delimiter=",", dtype=float)[:,1]


#print(Ref_LR.shape)
#print(Ref_MR.shape)
#print(Ref_HR.shape)

def _freqs(spin):
    wr222 = 0.0
    wi222 = 0.0
    wr223 = 0.0
    wi223 = 0.0
    wr224 = 0.0
    wi224 = 0.0
    wr230 = 0.0
    wi230 = 0.0
    wr220 = -0.0
    wi220 = 0.0
    wr_220 = 0.0
    wi_220 = 0.0
    wr221 = 0.0
    wi221 = 0.0
    if np.abs(spin - 0.7) < 10**-8:
        #print('set freq to 0.7')
        wr220 = -0.532600243551018
        wi220 = -0.08079287315500867
        wr221 = -0.5211607652680398
        wi221 = -0.24423831581238872
        wr222 = -0.4999062578250007
        wi222 = -0.41226153615173294
        wr223 = -0.4713362988408515
        wi223 = -0.5843007159749214
        wr224 = -0.4403849906392548
        wi224 = -0.753799653405434
        wr230 = -0.7591747231826982
        wi230 = -0.084189645785704
        wr_220 = 0.3098081303582614
        wi_220 = -0.08871719112891699
    if np.abs(spin - 0.7221) < 10**-8:
        wr220 = -0.5211607652680398
        wi220 = -0.24423831581238872
        #+0.023625978195093277
        #-0.4659664830942165  +i -0.029563722060915
        # put second mode to fit as 222 and third to 223
        wr_220 = -0.4999062578250007
        wi_220 = -0.41226153615173294
        #wr_220 = -0.5211607652680398
        #wi_220 = -0.24423831581238872
        wr221 = -0.4713362988408515
        wi221 = -0.5843007159749214
        wr222 = -0.4403849906392548
        wi222 = -0.753799653405434
    if np.abs(spin - 0.7 +0.022) < 10**-8:
        wr220 = 0.3098081303582614
        wi220 = -0.08871719112891699
    if np.abs(spin - 0.732) < 10**-8:
        wr220 = -0.7591747231826982
        wi220 = -0.084189645785704
    if np.abs(spin - (0.7-0.032)) < 10**-8:
        wr220 = 0.5316104315944815
        wi220 = -0.09117094742912757
    if np.abs(spin - 0.742) < 10**-8:
        wr220 = -0.9733886174477426
        wi220 = -0.08654487556987038
    if np.abs(spin - (0.7-0.042)) < 10**-8:
        wr220 = 0.9480093304365507
        wi220 = -0.0921264414028547
    if np.abs(spin - 0.752) < 10**-8:
        wr220 = -1.1815804275114292
        wi220 = -0.08786581657360668
    if np.abs(spin - (0.7-0.052)) < 10**-8:
        wr220 = 0.9480093304365507
        wi220 = -0.0921264414028547
    if np.abs(spin - 0.762) < 10**-8:
        wr220 = -1.3867446086501234
        wi220 = -0.08866999998258994
    if np.abs(spin - (0.7-0.062)) < 10**-8:
        wr220 = 1.1519288621668144
        wi220 = -0.09221581653180737
    if np.abs(spin - 0.772) < 10**-8:
        wr220 = -1.5902076683684574
        wi220 = -0.08920093584642762
    if np.abs(spin - (0.7-0.072)) < 10**-8:
        wr220 = 1.3546112183923658
        wi220 = -0.09223946004143191
    if np.abs(spin - 0.782) < 10**-8:
        wr220 = -1.7926241790923074
        wi220 = -0.08957396299594288
    if np.abs(spin - (0.7-0.082)) < 10**-8:
        wr220 = 1.5565075472397643
        wi220 = -0.09223398246862032
    #------------------------------------
    if np.abs(spin - 0.6) < 10**-8:
        wr220 = -0.49404478178138417
        wi220 = -0.08376520216104322
        wr221 = -0.4798066652457194
        wi221 = -0.2538468645349165
        wr_220 = 0.3167839926826497
        wi_220 = -0.08889174477892696
    if np.abs(spin - 0.6221) < 10**-8:
        wr220 = -0.4798066652457194
        wi220 = -0.2538468645349165
        wr221 = -0.4219766062313804
        wi221 = -0.6131535697953431
        wr_220 = -0.45417908920998973
        wi_220 = -0.43031520148939145
    if np.abs(spin - 0.6 +0.022) < 10**-8:
        wr220 = 0.3167839926826497
        wi220 = - 0.08889174477892696
    if np.abs(spin - 0.632) < 10**-8:
        wr220 = -0.7228019690705132
        wi220 = -0.08727995938648962
    if np.abs(spin - (0.6-0.032)) < 10**-8:
        wr220 = 0.5387961177380912
        wi220 = -0.09164443162689725
    #------------------------------------
    if np.abs(spin - 0.5) < 10**-8:
        wr220 = -0.46412302597593846
        wi220 = -0.08563883498806525
        wr221 = -0.4474070374532017
        wi221 = -0.2602245536370363
        wr_220 = 0.32430731434882354
        wi_220 = -0.08903154514941881
    if np.abs(spin - 0.5221) < 10**-8:
        wr220 = -0.4474070374532017
        wi220 = -0.2602245536370363
        wr221 = -0.38230890045095567
        wi221 = -0.6355364866733135
        wr_220 = -0.41792504473185443
        wi_220 = -0.4432868576633334
    if np.abs(spin - 0.5 +0.022) < 10**-8:
        wr220 = 0.32430731434882354
        wi220 = -0.08903154514941881
    if np.abs(spin - 0.532) < 10**-8:
        wr220 = -0.6934403948529166
        wi220 = -0.08928035399690099
    if np.abs(spin - (0.5-0.032)) < 10**-8:
        wr220 = 0.546663331506853
        wi220 = -0.0920366456127094
    #------------------------------------
    if np.abs(spin - 0.4) < 10**-8:
        wr220 = -0.4398419217353868
        wi220 = -0.08688196202939805
        wr221 = -0.4208466772299831
        wi221 = -0.2647334490018907
        wr_220 = 0.33245789036210477
        wi_220 = -0.08889174477892696
    if np.abs(spin - 0.4221) < 10**-8:
        wr220 = -0.4208466772299831
        wi220 = -0.2647334490018907
        wr221 = -0.3488558646168155
        wi221 = -0.6538081140561683
        wr_220 = -0.38777925474657043
        wi_220 = -0.4532594990474683
    if np.abs(spin - 0.4 +0.022) < 10**-8:
        wr220 = 0.33245789036210477
        wi220 = -0.08913134749418042
    if np.abs(spin - 0.432) < 10**-8:
        wr220 = -0.6689185112878061
        wi220 = -0.09062029136261898
    if np.abs(spin - (0.4-0.032)) < 10**-8:
        wr220 = 0.5552848116516478
        wi220 = -0.09234909704062234
    #------------------------------------
    if np.abs(spin - 0.3) < 10**-8:
        wr220 = -0.41952668176385166
        wi220 = -0.0877292718943111
    if np.abs(spin - 0.3221) < 10**-8:
        wr220 = -0.3983903260441566
        wi220 = -0.2680485703684768
        wr221 = -0.31988611040672216
        wi221 = -0.6692429338209821
        wr_220 = -0.3619272335159867
        wi_220 = -0.46125968993480565
    if np.abs(spin - 0.3 +0.022) < 10**-8:
        wr220 = 0.34133292557061845
        wi220 = -0.08918455000634863
    if np.abs(spin - 0.332) < 10**-8:
        wr220 = -0.6479580761437355
        wi220 = -0.09152479482817695
    if np.abs(spin - (0.3-0.032)) < 10**-8:
        wr220 = 0.5647493639850567
        wi220 = -0.09257992273500766
    #------------------------------------
    if np.abs(spin - 0.2) < 10**-8:
        wr220 = -0.40214532424765576
        wi220 = -0.08831116620760175
    if np.abs(spin - 0.2221) < 10**-8:
        wr220 = -0.37897635738373187
        wi220 = -0.27054560468828515
        wr221 = -0.27054560468828515
        wi221 = -0.68262839264893
        wr_220 = -0.3392851395194377
        wi_220 = -0.46787212388216887
    if np.abs(spin - 0.2 +0.022) < 10**-8:
        wr220 = 0.35105255048081224
        wi220 = -0.08918254197837719
    if np.abs(spin - 0.232) < 10**-8:
        wr220 = -0.6297384964740593
        wi220 = -0.09212426247876081
    if np.abs(spin - (0.2-0.032)) < 10**-8:
        wr220 = 0.5751665897150002
        wi220 = -0.09272352287510985
    #------------------------------------
    if np.abs(spin - 0.1) < 10**-8:
        wr220 = -0.3870175383664506
        wi220 = -0.08870569902764013
    if np.abs(spin - 0.1221) < 10**-8:
        wr220 = -0.3619096769861095
        wi220 = -0.27245210670518466
        wr221 = -0.2717487190296748
        wi221 = -0.6944809003917451
        wr_220 = -0.31915346872592654
        wi_220 = -0.4734630798788008
    if np.abs(spin - 0.1 +0.022) < 10**-8:
        wr220 = 0.36176770601052416
        wi220 = -0.0891137672847152
    if np.abs(spin - 0.132) < 10**-8:
        wr220 = -0.6137000190552652
        wi220 = -0.09249952991844297
    if np.abs(spin - (0.1-0.032)) < 10**-8:
        wr220 = 0.5866733983564321
        wi220 = -0.09276983111902914
    #------------------------------------
    if np.abs(spin - 0.0) < 10**-8:
        wr220 = -0.37367168441804177
        wi220 = -0.08896231568893546
        wr221 = -0.34671099687916285
        wi221 = -0.27391487529123504
        wr_220 = 0.37367168441804177
        wi_220 = -0.08896231568893546
    if np.abs(spin - 0.0221) < 10**-8:
        wr220 = -0.34671099687916285
        wi220 = -0.27391487529123504
        wr_220 = -0.30105345461250427
        wi_220 = -0.47827698322307044
        wr221 = -0.25150496222643026
        wi221 = -0.7051482024421033
    if np.abs(spin + 0.022) < 10**-8:
        wr220 = 0.37367168441804177
        wi220 = -0.08896231568893546
    if np.abs(spin - 0.032) < 10**-8:
        wr220 = -0.5994432884374902
        wi220 = -0.09270304794494749
    if np.abs(spin - (0.0-0.032)) < 10**-8:
        wr220 = 0.5994432884374902
        wi220 = -0.09270304794494749
    #------------------------------------
    if np.abs(spin - 0.025) < 10**-8:
        wr220 = -0.37686144047976894
        wi220 = -0.0889090878448264
    if np.abs(spin - 0.025 -0.0221) < 10**-8:
        wr220 = -0.3503561592226021
        wi220 = -0.273584743582656
        wr_220 = -0.30540853116926436
        wi_220 = -0.47713607406030945
        wr221 = -0.2563592081742995
        wi221 = -0.7025780806287536
    if np.abs(spin -0.025 + 0.022) < 10**-8:
        wr220 = 0.3705716311740007
        wi220 = -0.08900901919757231
    if np.abs(spin -0.025 - 0.032) < 10**-8:
        wr220 = -0.6028579767667704
        wi220 = -0.09266612559612487
    if np.abs(spin - (0.025-0.032)) < 10**-8:
        wr220 = 0.5961211877434203
        wi220 = -0.09273147837147921
    #------------------------------------
    if np.abs(spin - 0.9) < 10**-8:
        wr220 = -0.6716142721321631
        wi220 = -0.06486923587579635
        wr221 = -0.6676575508183792
        wi221 = -0.19525206705561435
        wr_220 = 0.29724421359059877
        wi_220 = -0.08828065931500927
        wr230 = -0.8761828189259053
        wi230 = -0.06890473056665566
        wr222 = -0.6598266820168089
        wi222 = -0.3275183753546835
    if np.abs(spin - 0.9221) < 10**-8:
        wr220 = -0.6676575508183792
        wi220 = -0.19525206705561435
        wr221 = -0.6478690335580597
        wi221 = -0.4628642465426118
        wr_220 = -0.6598266820168089
        wi_220 = -0.3275183753546835
    if np.abs(spin + 0.922) < 10**-8:
        wr220 = 0.29724421359059877
        wi220 = -0.08828065931500927
    if np.abs(spin - 0.932) < 10**-8:
        wr220 = -0.8761828189259053
        wi220 = -0.06890473056665566
    if np.abs(spin - (0.9-0.032)) < 10**-8:
        wr220 = 0.5190523174803382
        wi220 = -0.08995773773896432
    #------------------------------------
    if np.abs(spin - 0.8) < 10**-8:
        wr220 = -0.58601697490887
        wi220 = -0.07562955235606075
        wr221 = -0.5779223971672035
        wi221 = -0.22814894004163797
        wr_220 = 0.30331341891096103
        wi_220 = -0.08851223689698554
    if np.abs(spin - 0.8221) < 10**-8:
        wr220 = -0.5779223971672035
        wi220 = -0.22814894004163797
        wr221 = -0.5389559793540262
        wi221 = -0.5428881235954349
        wr_220 = -0.5622398151870551
        wi_220 = -0.38389521060820636
    if np.abs(spin + 0.822) < 10**-8:
        wr220 = 0.30331341891096103
        wi220 = -0.08851223689698554
    if np.abs(spin - 0.832) < 10**-8:
        wr220 = -0.8067827531824597
        wi220 = -0.07904586187349943
    if np.abs(spin - (0.8-0.032)) < 10**-8:
        wr220 = 0.5250460288508134
        wi220 = -0.09061142093147509
    #------------------------------------
    if np.abs(spin - 0.75) < 10**-8:
        wr220 = -0.5568173912621359
        wi220 = - 0.07860254371215213
    if np.abs(spin - 0.7721) < 10**-8:
        wr220 = -0.5469635330892949
        wi220 = -0.23735737224993403
        wr221 = -0.5019252500002859
        wi221 = -0.5659103718577457
        wr_220 = -0.5282769515819851
        wi_220 = -0.3999569520437921
    if np.abs(spin + 0.772) < 10**-8:
        wr220 = 0.30650433130483795
        wi220 = -0.08861826725646041
    if np.abs(spin - 0.782) < 10**-8:
        wr220 = -0.7811619162415332
        wi220 = -0.08197651920346763
    if np.abs(spin - (0.75-0.032)) < 10**-8:
        wr220 = 0.5282539282493041
        wi220 = -0.09090236640986027

    #------------------------------------
    if np.abs(spin - 0.0) < 10**-8:
        wr220 = -0.37367168441804177
        wi220 = -0.08896231568893546
        wr221 = -0.34671099687916285
        wi221 = -0.27391487529123504
        wr_220 = 0.37367168441804177
        wi_220 = -0.08896231568893546
        wr222 = -wr221
        wi222 = wi221
    if np.abs(spin - 0.0221) < 10**-8:
        wr220 = -0.34671099687916285
        wi220 = -0.27391487529123504
        wr_220 = -0.30105345461250427
        wi_220 = -0.47827698322307044
        wr221 = -0.25150496222643026
        wi221 = -0.7051482024421033
    if np.abs(spin + 0.022) < 10**-8:
        wr220 = 0.37367168441804177
        wi220 = -0.08896231568893546
    if np.abs(spin - 0.032) < 10**-8:
        wr220 = -0.5994432884374902
        wi220 = -0.09270304794494749
    if np.abs(spin - (0.0-0.032)) < 10**-8:
        wr220 = 0.5994432884374902
        wi220 = -0.09270304794494749


    #print(-0.532600243551018-wr220)
    #wr = np.array([wr220,wr221,wr222,wr223,wr224])
    #wi = np.array([wi220,wi221,wr222,wi223,wi224])
    wr = np.array([wr220,wr_220,wr221,wr222,wr230,wr223,wr224])
    wi = np.array([wi220,wi_220,wi221,wi222,wi230,wi223,wi224])
    #wr = np.array([wr221])
    #wi = np.array([wi221])
    return wr,wi



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

def self_convergence(dataR1,dataR2,dataR3,dataI1,dataI2,dataI3,times1=np.array([0]),times2=np.array([0])):
    from scipy import stats
    nt = dataR3.size
    print(times_LR[1000],times_MR[1000],times_HR[1000])
    #int(dataR3.size/4)-3
    if times1.size!=1:
        from scipy.interpolate import CubicSpline
        spl = CubicSpline(times2, dataR2)
        #dataR2 = spl(times1)
        spl = CubicSpline(times2, dataI2)
        #dataI2 = spl(times1)
    #nt =1
    if dataR1.size != dataR2.size or dataR1.size != dataR3.size:
        print('Error: times not aligned')
    #Amp1 = np.sqrt(dataR1**2+dataI1**2)[1:]
    #Amp2 = np.sqrt(dataR2**2+dataI2**2)[1:]
    #Amp3 = np.sqrt(dataR3**2+dataI3**2)[1:]
    Amp1 = np.zeros(nt)
    Amp2 = np.zeros(nt)
    Amp3 = np.zeros(nt)
    phase1 = np.zeros(nt)
    phase2 = np.zeros(nt)
    phase3 = np.zeros(nt)
    diffR12 = np.zeros(nt)
    diffR23 = np.zeros(nt)
    diffI12 = np.zeros(nt)
    diffI23 = np.zeros(nt)
    diffA12 = np.zeros(nt)
    diffA23 = np.zeros(nt)
    err = np.zeros(nt)
    for i in range(nt):
        #diffA12[i] = (np.sqrt(dataR1[i]**2+dataI1[i]**2)-np.sqrt(dataR2[2*i+1]**2+dataI2[2*i+1]**2))
        #diffA23[i] = -(np.sqrt(dataR3[4*i+3]**2+dataI3[4*i+3]**2)+np.sqrt(dataR2[2*i+1]**2+dataI2[2*i+1]**2))
        diffA12[i] = (np.sqrt(dataR1[i]**2+dataI1[i]**2)-np.sqrt(dataR2[i]**2+dataI2[i]**2))
        diffA23[i] = (-np.sqrt(dataR3[i]**2+dataI3[i]**2)+np.sqrt(dataR2[i]**2+dataI2[i]**2))
        #diffA23[i] = (-np.sqrt(dataR3[i]**2+dataI3[i]**2)+np.sqrt(dataR1[i]**2+dataI1[i]**2))
        #Amp1[i] = np.sqrt(dataR1[i]**2+dataI1[i]**2)
        #Amp2[i] = np.sqrt(dataR2[2*i+1]**2+dataI2[2*i+1]**2)
        #Amp3[i] = np.sqrt(dataR3[4*i+3]**2+dataI3[4*i+3]**2)
        #phase1[i] = cmath.phase(dataR1[i]+1j*dataI1[i])
        #phase2[i] = cmath.phase(dataR2[2*i+1]+1j*dataI2[2*i+1])
        #phase3[i] = cmath.phase(dataR3[4*i+3]+1j*dataI3[4*i+3])
        diffR12[i] = (dataR1[i]-dataR2[i])
        diffR23[i] = (dataR2[i]-dataR3[i])
        #diffR12[i] = (dataR1[i]-dataR2[2*i+1])
        #diffR23[i] = (dataR2[2*i+1]-dataR3[4*i+3])
        #diffI12[i] = (dataI1[i]-dataI2[2*i+1])
        #diffI23[i] = (dataI3[4*i+3]-dataI2[2*i+1])
        err[i] = np.abs(np.sqrt(dataR2[i]**2+dataI2[i]**2)-np.sqrt(dataR1[i]**2+dataI1[i]**2))/np.sqrt(dataR2[i]**2+dataI2[i]**2)
            #np.abs((dataR1[i+1]-dataR2[i+1])**2+(dataI1[i+1]-dataI2[i+1])**2)**0.5 / np.sqrt(dataR2[i+1]**2+dataI2[i+1]**2)
        #convA = np.log(np.abs((Amp1-Amp2)/(Amp2-Amp3)))/np.log(2)
        #convA = np.log(diffA12 /diffA23)/np.log(2)
        #convR = np.log(diffR12 /diffR23)/np.log(2)
        #convI = np.log(diffI12 /diffI23)/np.log(2)
        #err = (1.0/120.0)**4* err /((1.0/120.0)**4-(1.0/(120.0*2))**4)
        #err =  err /(2.0**4)
        
        #spl = CubicSpline(times1[int(4*times1.size/5)+1:], err[int(4*times1.size/5):])
        #extrap_err = spl(np.linspace(times1[int(3*times1.size/4)],400))
        #spl = stats.linregress(times1[int(4*times1.size/5)+1:], np.log10(err[int(4*times1.size/5):]))
        #extrap_err = np.interp(np.linspace(times1[int(3*times1.size/4)],400),times1[int(3*times1.size/4)+1:],np.log10(err[int(3*times1.size/4):]))
        #wi220 = -0.07582831597057399
        #wi221 = -0.22887254833789544
        # Two factors -- one for relative amplitude, the other for winding back to t=0
        #overtone_prop = np.exp((wi221-wi220)*times_LR[1:nt])*np.exp(wi221*times_LR[1:nt])
        #print('self convergence = ',np.mean(convA[:])," ",np.mean(convA[:]))
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('./plots/convergence.pdf')
    plt.figure()
        #plt.plot(times_LR[:],convR,label='R')
        #plt.plot(times_LR[:],convI,label='I')
        #plt.plot(times_LR[:nt],convA,label='A')
        #plt.plot(times_LR[:nt],np.ones_like(times_LR[:nt])*4,'--')
    #plt.plot(times_LR[:nt],diffA12*(16/15)/Amp1,label='$\Delta A(t)/A(t)$')
        #plt.plot(times_LR[:nt],((-2**4+4**4)/(2**4-1))*diffA23[:],label='16*(A_{MR}-A_{HR})')
        #plt.plot(times_LR[:nt],diffA12[:],label='(A_{LR}-A_{MR})')
    plt.plot(times_LR[index_cutofft(times_LR,6):index_cutofft(times_LR,100)],diffR12,label='$\Delta = Re(\Psi_4^{LR}-\Psi_4^{MR})$')
    plt.plot(times_LR[index_cutofft(times_LR,6):index_cutofft(times_LR,100)],16*diffR23,'--',label='$\Delta = 16\\times Re(\Psi_4^{MR}-\Psi_4^{HR})$')
        #plt.plot(phase1-phase2,label='LR-MR')
        #plt.plot(np.ones_like(convA)*4)
        #plt.plot((phase2-phase3)*((1-0.5**4)/(0.5**4-0.25**4)),label='fac*(MR-HR)')
        #plt.plot(times_LR,np.log(Amp2)+times_LR*0.08079287315500867,label='MR')
        #plt.plot(times_LR,np.log(Amp3)+times_LR*0.08079287315500867,label='HR')
        #plt.plot(times_LR,np.log(Amp1)+times_LR*0.08079287315500867,label='LR')
        #plt.plot(times_MR[1:nt],convI,'.',label='im')
        #plt.plot(times_LR[1:nt],np.log10(err),'b',label='Richardson extrapolated')
        #plt.plot(np.linspace(times1[int(3*times1.size/4)],400),np.log10(extrap_err),'-.',label='Error extrapolated')
        #plt.plot(np.linspace(times1[int(3*times1.size/4)],400),spl.intercept+spl.slope*np.linspace(times1[int(3*times1.size/4)],400),'r',linestyle='dotted', linewidth=1.0,label='Error extrapolated')
        #plt.plot(np.ones((50))*(-2-spl.intercept)/spl.slope,np.linspace(-13,-1),'k--',linewidth=0.5)
        #plt.plot(np.linspace(0,400),np.ones((50))*-2,'k--',linewidth=0.5)
        #plt.plot(np.linspace(0,60),np.ones((50))*-2,'k--',linewidth=0.5)
        #plt.plot(times_LR[:int(nt/3)],np.log10(err/overtone_prop)[:int(nt/3)],'--',linewidth=0.5,label=r"221 $A(t_0)$ relative error")
    #plt.ylim(-10**-8,10**-8)
    plt.xlim(0,100)
    plt.legend(loc="best")
    plt.tight_layout()
        #plt.ylabel('$\log(A)-\omega_i^{220}t$')
        #plt.ylabel('$\log_{10}(A_1)/\log_{10}(A_2)$ Amplitude relative error')
    plt.ylabel('$\left[\Delta(t)\\right]/Re(\Psi_4(0))$')
    plt.xlabel('$t/M$')
    pp.savefig()
    pp.close()
        #np.savetxt('Amp_error.txt',err)
        #print('percent level accuracy at t=',(-2-spl.intercept)/spl.slope,'M')
        #return err/overtone_prop
        #return err

    def ddata_dt(data, times): 
        dt = times[1]-times[0]
        inv_dt = 1 / dt
        print('1/dt = ',inv_dt)
        nt = times.size
        ddata = np.zeros_like(data)
        for k in range(nt-4):
            i = k+2
            ddata[i] =(
                    -(1 / 12.0) * data[i+2] + (2 / 3.0) * data[i+1] -
                    (2 / 3.0) * data[i-1] + (1 / 12.0) * data[i-2]
                ) * inv_dt
            #print(ddata[i])
        
        ddata[0] =(
                -3 * data[4] + 16 * data[3] - 36 * data[2] + 48 * data[1] -
                25 * data[0]) *(1 / 12.0) *inv_dt
        ddata[1] =(
                +1 * data[4] - 6 * data[3] + 18 * data[2] - 10 * data[1] -
                3 * data[0]) *(1 / 12.0) *inv_dt
        ddata[nt-1] =-(
                -3 * data[nt-5] + 16 * data[nt-4] - 36 * data[nt-3] +
                48 * data[nt-2] - 25 * data[nt-1]) *(1 / 12.0) *inv_dt
        ddata[nt-2] =-(
                +1 * data[nt-5] - 6 * data[nt-4] + 18 * data[nt-3] -
                10 * data[nt-2] - 3 * data[nt-1]) *(1 / 12.0) *inv_dt
        print(ddata)
        return ddata


    def frequency_evolution(time,dataR,dataI):
        npoint = time.size
        #wr = []
        #wi_list = []
        #time_list = []
        #for i in range(npoint-500):
        #    phases = omega_from_f(dataR[i:i+500],dataI[i:i+500],time[i:i+500])
        #    data,slope = find_period(phases,time[i:i+500])
        #    amp,wi = find_omegaI(dataR[i:i+500],dataI[i:i+500],time[i:i+500])
        #    wr.append(slope)
        #    wi_list.append(wi)
        #    time_list.append(time[i+250])

        phases = omega_from_f(dataR,dataI,time)
        data,slope = find_period(phases,time)
        amp,wi = find_omegaI(dataR,dataI,time)

        wr = ddata_dt(phases, time)
        wi_list = ddata_dt(np.log(amp),time)

        print('wr = ',wr)
        pp = PdfPages('./plots/freq_evol.pdf')
        plt.figure()
        #plt.plot(times,convR,label='R')
        #plt.plot(times,convI,label='I')
        plt.plot(time,-wr,label='wr')
        #plt.plot(time,-wi_list,label='wi')
        #plt.plot(time,0.5325999583183444*np.ones(npoint),'--',label='final wr')
        #plt.plot(time,0.07584132636225223*np.ones(npoint),'--',label='final wi')
        #plt.plot(time,0.08079287315500867*np.ones(npoint),'--',label='initial wi')
        #plt.plot(times,dataR3,label='HR')
        plt.legend()
        #plt.ylim(0.3,0.45)
        #plt.ylim(0.05,0.12)
        #plt.xlim(0,80)
        plt.ylabel('freq')
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


def reconstruct_mode(times,wi,wr,phase,amp):
    ref_phase = phase[-200]
    ref_amp = amp[-200]
    ref_time = times[-200]
    print('ref time = ',ref_time,ref_amp,ref_phase)
    initial_amp = np.exp(-wi*ref_time)*ref_amp
    initial_phase = cmath.phase((np.cos(-wr*ref_time)+1j*np.sin(-wr*ref_time))*cmath.exp(1j*ref_phase))
    Re = np.cos(initial_phase + wr* times) * initial_amp * np.exp(wi*times)
    Im = np.sin(initial_phase + wr* times) * initial_amp * np.exp(wi*times)
    print(np.sqrt(Re[-200]**2+Im[-200]**2), cmath.phase(Re[-200]+1j*Im[-200]))
    return Re,Im

def Mismatch(times, Ref1, Imf1, Ref2, Imf2):
    # Calculate mismatch to return final error
    psi_fstar_R = Ref1*Ref2 + Imf1*Imf2
    psi_fstar_I = Imf1*Ref2 - Ref1*Imf2
    psi_psistar = Ref1*Ref1 + Imf1*Imf1
    f_fstar = Ref2*Ref2+Imf2*Imf2
    intR = simps(psi_fstar_R,x=times)
    #print(intR)
    intI = simps(psi_fstar_I,x=times)
    norm1 = simps(psi_psistar,x=times)
    norm2 = simps(f_fstar,x=times)
    result = intR/np.sqrt(norm1*norm2)
    print(norm1)
    #print(1-norm2/np.sqrt(norm1*norm2))
    return (1-result)

def vary_w(times,Ref,Imf,free_freq=False,tail=False,num_modes=1,spin=0.0,ftol=1e-08):
    # Maximize overlap?
    import scipy.optimize
    norm_fac = np.amax(np.sqrt(Ref**2+Imf**2))**-1
    Ref = Ref*norm_fac
    Imf = Imf*norm_fac
    wr,wi = _freqs(spin)
    print(wr[0],wi[0])
    DM = 3*0.00711759
    wrf = wr
    wif = wi
    wr/=(1-DM)
    wi/=(1-DM)
    def _fit_function(inp,final=False):
        #print(inp)
        psiR = np.zeros_like(times)
        psiI = np.zeros_like(times)
        if num_modes==1:
            t0 = inp[0]
            #t0 = 4.92716938
            Q = inp[1] + 1j*inp[4]
            #Q = 0
            #Q = (1+4.19*DM*cmath.exp(1j*5.07*DM))*(cmath.exp(1j*(-wrf[0]-1j*wif[0])*(-t0)))/(cmath.exp(1j*(-wr[0]-1j*wi[0])*(-t0)))
            #print(Q)
            Ar = inp[2]
            Ai = inp[3]
            for i in range(times.size):
                dm = DM*(1-np.exp(2*wi[0]*(times[i]+t0)))
                #psi = (Ar+1j*Ai)*(1+Q*dm/DM)*cmath.exp((1-DM)/(1-DM+dm)*(1j*wr[0]+wi[0])*(times[i]-times[0]))
                phase = -(DM-1)*(wr[0]+1j*wi[0])*( cmath.log(-wi[0]*(cmath.exp(-2*wi[0]*times[i])-DM)) - cmath.log(-wi[0]*cmath.exp(-2*wi[0]*t0)+wi[0]*DM))/(2* wi[0])
                #print(phase.real,phase.imag)
                psi = (Ar+1j*Ai)*(1+Q*dm/DM)*cmath.exp(-1j*phase.real-phase.imag)
                psiR[i] = psi.real
                psiI[i] = psi.imag
        elif num_modes==2:
            t0 = inp[0]
            #t0 = 4.92716938
            Q1 = inp[1] + 1j*inp[4]
            #Q = 0
            #Q = (1+4.19*DM*cmath.exp(1j*5.07*DM))*(cmath.exp(1j*(-wrf[0]-1j*wif[0])*(-t0)))/(cmath.exp(1j*(-wr[0]-1j*wi[0])*(-t0)))
            #print(Q)
            Ar1 = inp[2]
            Ai1 = inp[3]
            Q2 = inp[5] + 1j*inp[6]
            Ar2 = inp[7]
            Ai2 = inp[8]
            for i in range(times.size):
                dm = DM*(1-np.exp(2*wi[0]*(times[i]+t0)))
                #psi = (Ar+1j*Ai)*(1+Q*dm/DM)*cmath.exp((1-DM)/(1-DM+dm)*(1j*wr[0]+wi[0])*(times[i]-times[0]))
                phase1 = -(DM-1)*(wr[0]+1j*wi[0])*( cmath.log(-wi[0]*(cmath.exp(-2*wi[0]*times[i])-DM)) - cmath.log(-wi[0]*cmath.exp(-2*wi[0]*t0)+wi[0]*DM))/(2* wi[0])
                phase2 = -(DM-1)*(wr[1]+1j*wi[1])*( cmath.log(-wi[1]*(cmath.exp(-2*wi[1]*times[i])-DM)) - cmath.log(-wi[1]*cmath.exp(-2*wi[1]*t0)+wi[1]*DM))/(2* wi[1])
                #print(phase.real,phase.imag)
                psi = (Ar1+1j*Ai1)*(1+Q1*dm/DM)*cmath.exp(-1j*phase1.real-phase1.imag) + (Ar2+1j*Ai2)*(1+Q2*dm/DM)*cmath.exp(-1j*phase2.real-phase2.imag)
                psiR[i] = psi.real
                psiI[i] = psi.imag
        if final==True:
            # Calculate mismatch to return final error
            psi_fstar_R = psiR*Ref + psiI*Imf
            psi_fstar_I = psiI*Ref - psiR*Imf
            psi_psistar = psiR*psiR + psiI*psiI
            f_fstar = Ref*Ref+Imf*Imf
            intR = simps(psi_fstar_R,x=times)
            intI = simps(psi_fstar_I,x=times)
            norm1 = simps(psi_psistar,x=times)
            norm2 = simps(f_fstar,x=times)
            result = intR/np.sqrt(norm1*norm2)
            print(1-result)
            return (1-result)
        print(np.sqrt(psiR**2+psiI**2)/np.sqrt(Ref**2+Imf**2))
        return np.reshape(np.array([psiR-Ref,psiI-Imf],dtype=float),(2*psiR.size))

    zData = np.ones_like(times)
    # Initialising from rough fits
    if num_modes==1:
        initialParameters = [ 5,  -0.0, Ref[0],  Imf[0],0]
    elif num_modes==2:
        initialParameters = [ 5,  -0.0, Ref[0],  Imf[0],0,0,0,0.001*Ref[0],0.001*Imf[0]]
    #data = 0
    # here a non-linear surface fit is made with scipy's curve_fit()
    #fittedParameters, pcov = curve_fit(_fit_function, data, zData, p0=initialParameters)
    print('initial = ',initialParameters)
    print('tol =',ftol)
    fit = scipy.optimize.least_squares(_fit_function,initialParameters,method='lm',ftol=ftol, xtol=ftol, gtol=ftol,verbose=0)
    #fit = scipy.optimize.least_squares(_fit_function,initialParameters)
    fittedParameters = fit.x
    print('fitted parameters', fittedParameters)
    modelPredictions = _fit_function(fittedParameters,final=True)
    print('final err = ',modelPredictions)
    #absError = (modelPredictions - zData)

    #fitR = fittedParameters[2]*np.exp(fittedParameters[1]*(times-times[0]))*np.cos(fittedParameters[0]*(times-times[0])) - fittedParameters[3]*np.exp(fittedParameters[1]*(times-times[0]))*np.sin(fittedParameters[0]*(times-times[0]))
    #fitI = fittedParameters[3]*np.exp(fittedParameters[1]*(times-times[0]))*np.cos(fittedParameters[0]*(times-times[0])) + fittedParameters[2]*np.exp(fittedParameters[1]*(times-times[0]))*np.sin(fittedParameters[0]*(times-times[0]))
    #pp = PdfPages('./plots/Mode_fit.pdf')
    #plt.figure()
    #plt.plot(times,Imf/norm_fac,'--',label='data')
    #plt.plot(times,fitI/norm_fac,'--',label='fit')
    #plt.legend()
    #plt.ylim(0.05,0.12)
    #plt.ylabel('psi4')
    #plt.xlabel('t')
    #pp.savefig()
    #pp.close()

    return fittedParameters,times[0],norm_fac,_fit_function(fittedParameters,final=True)

def fit_Mismatch(times,Ref,Imf,num_modes=1,free_freq=False,tail=False,spin=0.7,ftol=1e-08):
    # Maximize overlap?
    import scipy.optimize
    norm_fac = np.amax(np.sqrt(Ref**2+Imf**2))**-1
    Ref = Ref*norm_fac
    Imf = Imf*norm_fac
    wr,wi = _freqs(spin)
    def _fit_function(inp,final=False):
        #print(inp)
        psiR = np.zeros_like(times)
        psiI = np.zeros_like(times)
        if num_modes==1:
            if free_freq==True:
                wr[0] = inp[0]
                wi[0] = inp[1]
            Ar = inp[2]
            Ai = inp[3]
            for i in range(times.size):
                psi = (Ar+1j*Ai)*cmath.exp((1j*wr[0]+wi[0])*(times[i]-times[0]))
                psiR[i] = psi.real
                psiI[i] = psi.imag

        if num_modes==2:
            Ar = inp[:2]
            Ai = inp[2:]
            #Ar = inp[0]
            #Ai = inp[2]
            for mode in range(2):
                for i in range(times.size):
                    #psi = (Ar+1j*Ai)*cmath.exp((1j*wr[0]+wi[0])*(times[i]-times[0]))
                    #psiR[i] = psi.real
                    #psiI[i] = psi.imag
                    psi = (Ar[mode]+1j*Ai[mode])*cmath.exp((1j*wr[mode]+wi[mode])*(times[i]-times[0]))
                    psiR[i] += psi.real
                    psiI[i] += psi.imag

        if num_modes >= 3:
            Ar = inp[:num_modes]
            Ai = inp[num_modes:]
            for mode in range(num_modes):
                for i in range(times.size):
                    psi = (Ar[mode]+1j*Ai[mode])*cmath.exp((1j*wr[mode]+wi[mode])*(times[i]-times[0]))
                    psiR[i] += psi.real
                    psiI[i] += psi.imag

        if tail==True:
            Ar = inp[:num_modes+1]
            Ai = inp[num_modes+1:-1]
            pT = inp[-1]
            for mode in range(num_modes):
                for i in range(times.size):
                    psi = (Ar[mode]+1j*Ai[mode])*cmath.exp((1j*wr[mode]+wi[mode])*(times[i]-times[0]))
                    psiR[i] += psi.real
                    psiI[i] += psi.imag
            for i in range(times.size):
                psi = (Ar[-1]+1j*Ai[-1])*(times[i]-times[0])**pT
                psiR[i] += psi.real
                psiI[i] += psi.imag

        if final==True:
            # Calculate mismatch to return final error
            psi_fstar_R = psiR*Ref + psiI*Imf
            psi_fstar_I = psiI*Ref - psiR*Imf
            psi_psistar = psiR*psiR + psiI*psiI
            f_fstar = Ref*Ref+Imf*Imf
            intR = simps(psi_fstar_R,x=times)
            intI = simps(psi_fstar_I,x=times)
            norm1 = simps(psi_psistar,x=times)
            norm2 = simps(f_fstar,x=times)
            result = intR/np.sqrt(norm1*norm2)
            return (1-result)
        #print(np.array([Ref,Imf],dtype=float))
        return np.reshape(np.array([psiR-Ref,psiI-Imf],dtype=float),(2*psiR.size))

    zData = np.ones_like(times)
    # Initialising from rough fits
    if num_modes==1:
        #initialParameters = [ -0.5,  -0.08, Ref[0],  Imf[0]]
        initialParameters = [ 0.5,  0.08, 1,  1]
    if num_modes>=3:
        initialParameters = np.ones(2*num_modes)
        if tail ==True:
            initialParameters = np.ones(2*num_modes+3)
    if num_modes==2:
        #initialParameters = np.ones(4)
        initialParameters = np.array([Ref[0],Ref[0]*0.01,Imf[0],Imf[0]*0.01],dtype=float)
    #data = 0
    # here a non-linear surface fit is made with scipy's curve_fit()
    #fittedParameters, pcov = curve_fit(_fit_function, data, zData, p0=initialParameters)
    #print('initial = ',initialParameters)
    #print('tol =',ftol)
    if num_modes>1.5:
        fit = scipy.optimize.least_squares(_fit_function,initialParameters,method='lm',ftol=ftol, xtol=ftol, gtol=ftol,verbose=0)
    else:
        print('intial params = ',initialParameters)
        fit = scipy.optimize.least_squares(_fit_function,initialParameters,method='lm',ftol=ftol, xtol=ftol, gtol=ftol,verbose=0,bounds=([-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf]))
    #fit = scipy.optimize.least_squares(_fit_function,initialParameters)
    fittedParameters = fit.x
    print('fitted parameters', fittedParameters.size)
    modelPredictions = _fit_function(fittedParameters,final=True)
    print('final err = ',modelPredictions)
    #absError = (modelPredictions - zData)

    #fitR = fittedParameters[2]*np.exp(fittedParameters[1]*(times-times[0]))*np.cos(fittedParameters[0]*(times-times[0])) - fittedParameters[3]*np.exp(fittedParameters[1]*(times-times[0]))*np.sin(fittedParameters[0]*(times-times[0]))
    #fitI = fittedParameters[3]*np.exp(fittedParameters[1]*(times-times[0]))*np.cos(fittedParameters[0]*(times-times[0])) + fittedParameters[2]*np.exp(fittedParameters[1]*(times-times[0]))*np.sin(fittedParameters[0]*(times-times[0]))
    #pp = PdfPages('./plots/Mode_fit.pdf')
    #plt.figure()
    #plt.plot(times,Imf/norm_fac,'--',label='data')
    #plt.plot(times,fitI/norm_fac,'--',label='fit')
    #plt.legend()
    #plt.ylim(0.05,0.12)
    #plt.ylabel('psi4')
    #plt.xlabel('t')
    #pp.savefig()
    #pp.close()

    return fittedParameters,times[0],norm_fac,_fit_function(fittedParameters,final=True)
    #[0]
#return fittedParameters,times[0],norm_fac,Ref/norm_fac
#return fittedParameters,times[0],norm_fac,modelPredictions

def vary_fit_time(times,Ref,Imf,num_modes=1,free_freq=False,tail=False,spin=0.7,ftol=1e-08,ref_time=0):
    step =300
    times_to_fit = times[:-1000].size
    #times_to_fit = times[:-10].size/5
    #times_to_fit = times[:1000].size
    #times_to_fit = times[:index_cutofft(times,70)].size
    AmpR = np.zeros((num_modes,int(times_to_fit/step)))
    AmpI = np.zeros((num_modes,int(times_to_fit/step)))
    final_mismatch = np.zeros(int(times_to_fit/step))
    if free_freq==True:
        wr_fit = np.zeros(int(times_to_fit/step))
        wi_fit = np.zeros(int(times_to_fit/step))
    if tail==True:
        AmpR = np.zeros((num_modes+1,int(times_to_fit/step)))
        AmpI = np.zeros((num_modes+1,int(times_to_fit/step)))
        pT = np.zeros(int(times_to_fit/step))
    #time0s = times[:-10]
    time0s = np.zeros(int(times_to_fit/step))
    wr,wi = _freqs(spin)
    for time0 in range(int(times_to_fit/step)):
        print(times[time0*step],Ref[time0*step],Imf[time0*step])
        fittedParameters,timefit0,norm_fac,final_residual = fit_Mismatch(times[time0*step:],Ref[time0*step:],Imf[time0*step:],num_modes=num_modes,free_freq=free_freq,tail=tail,spin=spin,ftol=ftol)
        AR = np.zeros((num_modes))
        AI = np.zeros((num_modes))
        if tail==True:
            AR = np.zeros((num_modes+1))
            AI = np.zeros((num_modes+1))
        if free_freq==True:
            wr[0] = fittedParameters[0]
            wi[0] = fittedParameters[1]
            wr_fit[time0] = fittedParameters[0]
            wi_fit[time0] = fittedParameters[1]
            print('fitted freq = ',fittedParameters[0],' +i',fittedParameters[1])
        if num_modes==1:
            AR[0] = fittedParameters[2]
            AI[0] = fittedParameters[3]
        else:
            if tail==False:
                AR[:] = fittedParameters[:num_modes]
                AI[:] = fittedParameters[num_modes:]
            else:
                AR[:] = fittedParameters[:num_modes+1]
                AI[:] = fittedParameters[num_modes+1:-1]
                pTtemp = fittedParameters[-1]
        for mode in range(num_modes):
            psi = (AR[mode]+1j*AI[mode])*cmath.exp((1j*wr[mode]+wi[mode])*(ref_time-timefit0))
            AmpR[mode,time0] = psi.real/norm_fac
            AmpI[mode,time0] = psi.imag/norm_fac
            #AmpR[mode,time0] = AR[mode]/norm_fac
            #AmpI[mode,time0] = AI[mode]/norm_fac
        if tail==True:
            psi = (AR[-1]+1j*AI[-1])*(timefit0)**-pTtemp
            AmpR[-1,time0] = psi.real/norm_fac
            AmpI[-1,time0] = psi.imag/norm_fac
            pT[time0] = pTtemp
            #AmpR[-1,time0] = AR[mode]/norm_fac
            #AmpI[-1,time0] = AI[mode]/norm_fac
        #AmpR[time0]=(fittedParameters[2]*np.exp(wi*(-timefit0))*np.cos(wr*(-timefit0)) - fittedParameters[3]*np.exp(wi*(-timefit0))*np.sin(wr*(-timefit0)))/norm_fac
        #AmpI[time0]=(fittedParameters[3]*np.exp(wi*(-timefit0))*np.cos(wr*(-timefit0)) + fittedParameters[2]*np.exp(wi*(-timefit0))*np.sin(wr*(-timefit0)))/norm_fac
        final_mismatch[time0]=final_residual
        time0s[time0] = times[time0*step]
    if free_freq==True:
        return time0s,final_mismatch,AmpR,AmpI,wr_fit,wi_fit
    if tail==True:
        return time0s,final_mismatch,AmpR,AmpI,pT
    return time0s,final_mismatch,AmpR,AmpI

def plot_vary_fit(time,ref,imf,time_fin,ref_fin,imf_fin,num_modes=1,free_freq=False,tail=False,err=np.zeros(1),spin = 0.7,ftol=1e-08):
    wr,wi = _freqs(spin)
    if free_freq==True:
        time0s,final_mismatch,AmpR,AmpI,wr_fit,wi_fit = vary_fit_time(time[:],ref[:],imf[:],num_modes=num_modes,free_freq=True,spin=spin,ftol=ftol)
    elif tail==True:
        time0s,final_mismatch,AmpR,AmpI,pT = vary_fit_time(time[:],ref[:],imf[:],num_modes=num_modes,tail=tail,spin=spin,ftol=ftol)
    else:
        time0s,final_mismatch,AmpR,AmpI = vary_fit_time(time[:],ref[:],imf[:],num_modes=num_modes,spin=spin,ftol=ftol)
        #time0s,final_mismatch,AmpR,AmpI = vary_fit_time(time[:index_cutofft(time,times_HR[-1])],ref[:index_cutofft(time,times_HR[-1])],imf[:index_cutofft(time,times_HR[-1])],num_modes=num_modes,spin=spin)
    AmpR22 = AmpR[0,:]
    AmpI22 = AmpI[0,:]
    if err.size >1:
        from scipy import interpolate
        err_fit = interpolate.interp1d(time[:err.size],err)
    if num_modes>1:
        AmpR_220 = AmpR[1,:]
        AmpI_220 = AmpI[1,:]
        if num_modes>2:
            AmpR221 = AmpR[2,:]
            AmpI221 = AmpI[2,:]
            if num_modes>3:
                AmpR222 = AmpR[3,:]
                AmpI222 = AmpI[3,:]
                if err.size >1:
                    errbarsu = np.sqrt(AmpR221**2+AmpI221**2)+err_fit(time0s)/(np.exp((wi221-wi220)*time0s)*np.exp(wi221*time0s))
                    errbarsl = np.sqrt(AmpR221[:]**2+AmpI221[:]**2)-err_fit(time0s)/(np.exp((wi221-wi220)*time0s)*np.exp(wi221*time0s))
                    for n,er in enumerate(errbarsl):
                        if er<0:
                            errbarsl[n]=10**-4
                if num_modes>4:
                    AmpR230 = AmpR[4,:]
                    AmpI230 = AmpI[4,:]
                    if err.size >1:
                        errbarsu2 = np.sqrt(AmpR222**2+AmpI222**2)+err_fit(time0s)/(np.exp((wi222-wi220)*time0s)*np.exp(wi222*time0s))
                        errbarsl2 = np.sqrt(AmpR222[:]**2+AmpI222[:]**2)-err_fit(time0s)/(np.exp((wi222-wi220)*time0s)*np.exp(wi222*time0s))
                        for n,er in enumerate(errbarsl):
                            if er<0:
                                errbarsl2[n]=10**-4
    if tail==True:
        AmpRtail = AmpR[-1,:]
        AmpItail = AmpI[-1,:]

    #print(times_MR[:50000].size,Ref_MR[:50000].size,Imf_MR[:50000].size)
    
    #ftol = 10**-14
    #if tail==False:
    #    time0shr,final_mismatchhr,AmpRhr,AmpIhr = vary_fit_time(times_LR[:index_cutofft(times_LR,times_HR[-1])],Ref_LR[:index_cutofft(times_LR,times_HR[-1])],Imf_LR[:index_cutofft(times_LR,times_HR[-1])],num_modes=num_modes,tail=tail,spin=spin,ftol=ftol)
        #time0shr,final_mismatchhr,AmpRhr,AmpIhr = vary_fit_time(times_LR[:index_cutofft(times_LR,60)],Ref_LR[:index_cutofft(times_LR,60)],Imf_LR[:index_cutofft(times_LR,60)],num_modes=num_modes,tail=tail,spin=spin,ftol=ftol)
    #else:
    #    time0shr,final_mismatchhr,AmpRhr,AmpIhr,pThr = vary_fit_time(times_HR[:],Ref_HR[:],Imf_HR[:],num_modes=num_modes,tail=tail,spin=spin,ftol=ftol)
        
    #print(time0shr)
    #AmpR220_hr = AmpRhr[0,:]
    #AmpI220_hr = AmpIhr[0,:]
    #if num_modes>=2:
    #    AmpR_220_hr = AmpRhr[1,:]
    #    AmpI_220_hr = AmpIhr[1,:]
    #if num_modes>=3:
    #    AmpR221_hr = AmpRhr[2,:]
    #    AmpI221_hr = AmpIhr[2,:]
    #if num_modes>=4:
    #    AmpR230_hr = AmpRhr[3,:]
    #    AmpI230_hr = AmpIhr[3,:]
    #if tail==True:
    #    AmpRtail_hr = AmpRhr[-1,:]
    #    AmpItail_hr = AmpIhr[-1,:]

    #time0slr,final_mismatchlr,AmpRlr,AmpIlr = vary_fit_time(times_XLR[:index_cutofft(times_XLR,60)],Ref_XLR[:index_cutofft(times_XLR,60)],Imf_XLR[:index_cutofft(times_XLR,60)],num_modes=num_modes,spin=spin)
    #AmpR220_lr = AmpRlr[0,:]
    #AmpI220_lr = AmpIlr[0,:]
    #if num_modes>=2:
    #    AmpR_220_lr = AmpRlr[1,:]
    #    AmpI_220_lr = AmpIlr[1,:]
    #if num_modes>=3:
    #    AmpR221_lr = AmpRlr[2,:]
    #    AmpI221_lr = AmpIlr[2,:]
    #if num_modes>=4:
    #    AmpR230_lr = AmpRlr[3,:]
    #    AmpI230_lr = AmpIlr[3,:]


    #time0s2,final_mismatch2,AmpR2,AmpI2 = vary_fit_time(times_2[:28724],Ref_2[:28724],Imf_2[:28724],num_modes=num_modes)
    #AmpR22_2 = AmpR2[0,:]
    #AmpI22_2 = AmpI2[0,:]
    #if num_modes>=2:
    #    AmpR_220_2 = AmpR2[1,:]
    #    AmpI_220_2 = AmpI2[1,:]
    #if num_modes>=3:
    #    AmpR221_2 = AmpR2[2,:]
    #    AmpI221_2 = AmpI2[2,:]

    #time0s3,final_mismatch3,AmpR3,AmpI3 = vary_fit_time(times_3[:28724],Ref_3[:28724],Imf_3[:28724],num_modes=num_modes)
    #AmpR22_3 = AmpR3[0,:]
    #AmpI22_3 = AmpI3[0,:]
    #if num_modes>=2:
    #    AmpR_220_3 = AmpR3[1,:]
    #    AmpI_220_3 = AmpI3[1,:]
    #if num_modes>=3:
    #    AmpR221_3 = AmpR3[2,:]
    #    AmpI221_3 = AmpI3[2,:]
    #if num_modes>=4:
    #    AmpR221_5 = AmpR2[3,:]
    #    AmpI221_5 = AmpI2[3,:]
    
    #time0s3,final_mismatch3,AmpR3,AmpI3 = vary_fit_time(times_3,Ref_3,Imf_3,num_modes=1)
    #AmpR22_3 = AmpR3[0,:]
    #AmpI22_3 = AmpI3[0,:]

    #time0sf,final_mismatchf,AmpRf,AmpIf = vary_fit_time(time_fin,ref_fin,imf_fin,num_modes=1,spin=spin)
    #AmpR22_f = AmpRf[0,:]
    #AmpI22_f = AmpIf[0,:]

    #fitA221 = np.argmin(np.abs(ddata_dt(np.sqrt(AmpR221**2+AmpI221**2), time0s)[index_cutofft(time0s,20):index_cutofft(time0s,40)]))
    #fitA220R = np.argmin(np.abs(ddata_dt(np.sqrt(AmpR_220**2+AmpI_220**2), time0s)[index_cutofft(time0s,40):]))
    #fitA220 = np.argmin(np.abs(ddata_dt(np.sqrt(AmpR22**2+AmpI22**2), time0s)[index_cutofft(time0s,20):]))

    #A221fit = (np.sqrt(AmpR221**2+AmpI221**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2))[index_cutofft(time0s,20)+fitA221]
    #A220Rfit = (np.sqrt(AmpR_220**2+AmpI_220**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2))[index_cutofft(time0s,40)+fitA220R]
    #d220fit = (np.abs(np.sqrt(AmpR22**2+AmpI22**2)-np.sqrt(AmpR22_f**2+AmpI22_f**2))/np.sqrt(AmpR22_f**2+AmpI22_f**2))[index_cutofft(time0s,20)+fitA220]

    #print('221 = ',(np.sqrt(AmpR221**2+AmpI221**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2))[index_cutofft(time0s,20)+fitA221])
    #print('220R = ',(np.sqrt(AmpR_220**2+AmpI_220**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2))[index_cutofft(time0s,40)+fitA220R])
    #print('d220 = ',(np.abs(np.sqrt(AmpR22**2+AmpI22**2)-np.sqrt(AmpR22_f**2+AmpI22_f**2))/np.sqrt(AmpR22_f**2+AmpI22_f**2))[index_cutofft(time0s,20)+fitA220])

    #for i in range(time0s.size):
    #    print('time = ',time0s[i])
    #    print('221 = ',(np.sqrt(AmpR221**2+AmpI221**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2))[i])
    #    print('220R = ',(np.sqrt(AmpR_220**2+AmpI_220**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2)[i])
    #    print('d220 = ',(np.abs(np.sqrt(AmpR22**2+AmpI22**2)-np.sqrt(AmpR22_f**2+AmpI22_f**2))/np.sqrt(AmpR22_f**2+AmpI22_f**2))[i])

    pp = PdfPages('./plots/Amp_fit_vary_to2.pdf')
    plt.figure()
    #plt.plot(time0sf,np.abs(AmpR22_f),'b--',label='Re(Amp) fixed')
    #plt.plot(time0sf,np.abs(AmpI22_f),'g--',label='Im(Amp) fixed')
    plt.plot(time0s,np.abs(AmpR22),'b',label='Re(Amp) changing')
    plt.plot(time0s,np.abs(AmpI22),'g',label='Im(Amp) changing')
    if num_modes>1:
        plt.plot(time0s,np.abs(AmpR_220),'b-.',label='Re(Amp -220) changing')
        plt.plot(time0s,np.abs(AmpI_220),'g-.',label='Im(Amp -220) changing')
    plt.legend()
    plt.ylabel('Amp fitted')
    plt.xlabel('$t_0$')
    plt.title('Fixed freq single Amp fit')
    pp.savefig()
    pp.close()
    pp = PdfPages('./plots/phases_vary_to2.pdf')
    plt.figure()
    #plt.plot(time0s,np.arctan(AmpI22/AmpR22)-np.arctan(AmpI22_f/AmpR22_f),label='changing $1.0\\times dM$')
    #plt.plot(time0s,1/3.0*(np.arctan(AmpI22_3/AmpR22_3)-np.arctan(AmpIfinal/AmpRfinal)),label='changing $3\\times dM$')
    #plt.plot(time0s,0.5*(np.arctan(AmpI22_2/AmpR22_2)-np.arctan(AmpI22_f/AmpR22_f)),label='changing $2\\times dM$')
    plt.ylabel('$\phi_{physical}-\phi_{fixed}$')
    plt.xlabel('$t_0$')
    plt.legend()
    plt.title('Fixed freq single Amp fit -- phase')
    pp.savefig()
    pp.close()
    pp = PdfPages('./plots/Amp_vary_to0.72.pdf')
    plt.figure()
    #plt.plot(time0s,np.ones_like(time0s),'--',label='fixed')
    #plt.plot(time0s,np.log10(np.abs(np.sqrt(AmpR22**2+AmpI22**2)-np.sqrt(AmpR22_f**2+AmpI22_f**2))/np.sqrt(AmpR22_f**2+AmpI22_f**2)),'b',label='$220$ mode')
    #plt.plot(time0s[index_cutofft(time0s,20)+fitA220]*np.ones(50),np.linspace(-2.5,0),'b:')
    #plt.plot(time0s,np.log10(np.abs(-np.sqrt(AmpR22**2+AmpI22**2)+np.sqrt(AmpR22_f[1]**2+AmpI22_f[1]**2)))/np.sqrt(AmpR22_f[1]**2+AmpI22_f[1]**2),label='$\Delta 220$ mode')
    plt.plot(time0s,np.log10(np.abs(np.sqrt(AmpR22**2+AmpI22**2))),label='$220$ mode')
    #plt.plot(time0shr,np.log10(np.sqrt(AmpR220_hr**2+AmpI220_hr**2)),'b:',label='lower tol')
    #plt.plot(time0slr,np.log10(np.sqrt(AmpR220_lr**2+AmpI220_lr**2)),'b--',label='Low res')
    #plt.plot(time0s,np.log10(np.sqrt(AmpR22**2+AmpI22**2)),'b',label='221 mode')
    if tail==True:
        plt.plot(time0s,np.log10(np.sqrt(AmpRtail**2+AmpItail**2)),'r',label='tail')
        plt.plot(time0s,np.log10(pT*np.ones_like(time0s)),'r--',label='pT')
        plt.plot(time0shr,np.log10(np.sqrt(AmpRtail_hr**2+AmpItail_hr**2)),'r:')
        plt.plot(time0shr,np.log10(pThr),'r:')
    if num_modes>1:
        #plt.plot(time0s,np.log10(np.sqrt(AmpR_220**2+AmpI_220**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2)),'g',label='$220_R$ mode')
        #plt.plot(time0s[fitA220R+index_cutofft(time0s,40)]*np.ones(50),np.linspace(-2.5,0),'g:')
        #plt.plot(time0s2,np.log10(0.5*np.sqrt(AmpR_220_2**2+AmpI_220_2**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2)),'g:',label='changing $2\\times dM$')
        #plt.plot(time0s3,np.log10((3.0**-1)*np.sqrt(AmpR_220_3**2+AmpI_220_3**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2)),'g--',label='changing $3\\times dM$')
        #plt.plot(time0shr,np.log10(np.sqrt(AmpR_220_hr**2+AmpI_220_hr**2)),'g:')
        #plt.plot(time0slr,np.log10(np.sqrt(AmpR_220_lr**2+AmpI_220_lr**2)),'g--')
        plt.plot(time0s,np.log10(np.sqrt(AmpR_220**2+AmpI_220**2)/np.sqrt(Ref_fix[1]**2+Imf_fix[1]**2)),'g',label='$220R$ mode')
        if num_modes>2:
            #plt.plot(time0s,np.log10(np.sqrt(AmpR221**2+AmpI221**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2)),'y',label='$221$ mode')
            #plt.plot(time0s[index_cutofft(time0s,20)+fitA221]*np.ones(50),np.linspace(-2.5,0),'y:')
            #plt.plot(time0s[index_cutofft(time0s,20)+fitA221]*np.ones(50),np.linspace(-2.5,0),'y:')
            #plt.plot(time0s2,np.log10(0.5*np.sqrt(AmpR221_2**2+AmpI221_2**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2)),'y:',label='changing $2\\times dM$')
            #plt.plot(time0s3,np.log10((3.0**-1)*np.sqrt(AmpR221_3**2+AmpI221_3**2)/np.sqrt(AmpR22_f**2+AmpI22_f**2)),'y--',label='changing $3\\times dM$')
            #plt.plot(time0shr,np.log10(np.sqrt(AmpR221_hr**2+AmpI221_hr**2)),'y:')
            plt.plot(time0s,np.log10(np.sqrt(AmpR221**2+AmpI221**2)/np.sqrt(Ref_fix[1]**2+Imf_fix[1]**2)),'y',label='$221$ mode')
            #plt.plot(time0slr,np.log10(np.sqrt(AmpR221_lr**2+AmpI221_lr**2)),'y--')
            if num_modes>3:
                plt.plot(time0s,np.log10(np.sqrt(AmpR222**2+AmpI222**2)),'r',label='222 mode')
                #plt.plot(time0shr,np.log10(np.sqrt(AmpR221_hr**2+AmpI221_hr**2)),'r:')
                #if err.size >1:
                    #plt.plot(time0s,np.log10(errbarsu/np.sqrt(AmpRfinal**2+AmpIfinal**2)),'r--',linewidth=0.5,label='error on 221')
                    #plt.plot(time0s,np.log10(errbarsl/np.sqrt(AmpRfinal**2+AmpIfinal**2)),'r--',linewidth=0.5)
                if num_modes>4:
                    plt.plot(time0s,np.log10(np.sqrt(AmpR222**2+AmpI222**2)/np.sqrt(AmpRfinal**2+AmpIfinal**2)),'k',label='230 mode')
                    if err.size >1:
                        plt.plot(time0s,np.log10(errbarsu2/np.sqrt(AmpRfinal**2+AmpIfinal**2)),'k--',linewidth=0.5,label='error on 221')
                        plt.plot(time0s,np.log10(errbarsl2/np.sqrt(AmpRfinal**2+AmpIfinal**2)),'k--',linewidth=0.5)
    #plt.plot(time0s,0.75**-1*(np.sqrt(AmpR22_75**2+AmpI22_75**2)-np.sqrt(AmpR22_f**2+AmpI22_f**2))/np.sqrt(AmpR22_f**2+AmpI22_f**2),label='changing $0.75\\times dM$')
    #plt.plot(time0s,0.5*(np.sqrt(AmpR22_2**2+AmpI22_2**2)-np.sqrt(AmpR22_f**2+AmpI22_f**2))/np.sqrt(AmpR22_f**2+AmpI22_f**2),label='changing $2\\times dM$')
    #plt.plot(time0s,3.0**-1*(np.sqrt(AmpR22_3**2+AmpI22_3**2)-np.sqrt(AmpR22_f**2+AmpI22_f**2))/np.sqrt(AmpR22_f**2+AmpI22_f**2),label='changing $3\\times dM$')
    #plt.plot(time0s,np.ones_like(time0s),'--')
    #plt.ylim(-2,2)
    plt.xlim(0,20)
    plt.ylabel('$log (Amp_{phys}(t=t_0) / Amp^{220}_{fixed}(t=0))$')
    #plt.ylabel('$\\frac{1}{dM Amp_{fixed}}( Amp_{physical} - Amp_{fixed})$')
    plt.xlabel('$t_0$')
    plt.legend()
    plt.title('Fixed freq fit')
    pp.savefig()
    pp.close()

    pp = PdfPages('./plots/Mismatch_vary_to0.9.pdf')
    plt.figure()
    plt.plot(time0s,np.log10(final_mismatch))
    plt.ylabel('log Mismatch')
    plt.xlabel('$t_0$')
    plt.title('Fixed freq single Amp fit')
    pp.savefig()
    pp.close()

    if free_freq == True:
        tinit = 9

        print(wr_fit[index_cutofft(time0s,19)]," ",wi_fit[index_cutofft(time0s,19)])
        pp = PdfPages('./plots/w_vary_to.pdf')
        plt.figure()
        plt.scatter(x=-wr_fit[1:index_cutofft(time0s,59)], y=wi_fit[1:index_cutofft(time0s,59)], c=time0s[1:index_cutofft(time0s,59)]-tinit, cmap="plasma")
        plt.colorbar(label="$t_0/M$", orientation="horizontal")
        #plt.plot([wr220,wr221,wr230],[wi220,wi221,wi230],'o',label='QNMs')
        plt.scatter(-wr[:1],wi[:1], s=80, facecolors='none', edgecolors='k',label='220')
        #plt.scatter(-wr[1:2],wi[1:2], s=80, facecolors='none', edgecolors='g',label='220R')
        #plt.scatter(-wr[2:4],wi[2:4], s=80, facecolors='none', edgecolors='r',label='22n')
        plt.scatter(-wr[2:3],wi[2:3], s=80, facecolors='none', edgecolors='r',label='221')
        #plt.scatter(-wr[4:5],wi[4:5], s=80, facecolors='none', edgecolors='c',label='320')
        plt.ylim(-0.252,-0.065)
        plt.xlim(0.505,0.535)
        #plt.axhline(y=3*wi[:1], color='k', linestyle=':',label='$3I\omega_{200} M$')
        plt.ylabel('$\omega_i M$')
        plt.xlabel('$\omega_r M$')
        plt.legend()
        #plt.title('Free freq fit')
        pp.savefig()
        pp.close()

        pp = PdfPages('./plots/w_vary_dist2.pdf')
        plt.figure()
        plt.scatter(x=time0s[1:]-tinit, y=np.log10(np.sqrt((wr_fit[1:]-wr[2])**2+(wi_fit[1:]-wi[2])**2)), c=time0s[1:]-tinit, cmap="plasma")
        #plt.scatter(x=time0s[1:]-tinit, y=np.log10(np.sqrt((wr_fit[1:]-0.5316104315944815)**2+(wi_fit[1:]+0.09117094742912757)**2)), c=time0s[1:]-tinit, cmap="plasma")
        plt.colorbar(label="$t_0/M$", orientation="horizontal")
        #plt.plot(time0s,np.log10(np.sqrt((wr_fit-0.5316104315944815)**2+(wi_fit+0.09117094742912757)**2)),'.')
        plt.ylabel('$log_{10}\left|\omega - \omega_{201}\\right|M$')
        plt.xlabel('$t_0/M$')
        #plt.title('Free freq residual fit')
        pp.savefig()
        pp.close()

        #np.savetxt('free_freq_fit.txt',np.array([time0s,wr_fit,wi_fit,AmpR22,AmpI22]).flatten)
    return time0s,np.log10(np.sqrt(AmpR22**2+AmpI22**2))

def fit_amps_22(foldername,tinit,spin,perturbing=1,filter_fac = np.ones((5))):
    # First fundamental mode
    file1 = np.loadtxt("%s/except220_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    index_init = index_cutofft(times,tinit)
    index_final = index_cutofft(times,100)
    Ref = file1[:,1]
    Imf = file1[:,2]
    filtered_amp = np.sqrt(Ref[index_init]**2+Imf[index_init]**2)
    time0s_220,final_mismatch_220,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin,ftol=1e-08,ref_time=tinit)
    AmpR_220 = AmpR[0,:]/perturbing
    AmpI_220 = AmpI[0,:]/perturbing

    # Fitting overtones
    file1 = np.loadtxt("%s/except3overtones_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    
    time0s_overtones,final_mismatch_overtones,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=3,free_freq=False,tail=False,spin=spin+0.0221,ftol=1e-08,ref_time=tinit)
    AmpR_221 = AmpR[0,:]/perturbing
    AmpI_221 = AmpI[0,:]/perturbing
    AmpR_222 = AmpR[1,:]/perturbing
    AmpI_222 = AmpI[1,:]/perturbing
    AmpR_223 = AmpR[2,:]/perturbing
    AmpI_223 = AmpI[2,:]/perturbing
    
    # Fitting 220R
    file1 = np.loadtxt("%s/except220R_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    
    time0s_220R,final_mismatch_220R,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.022,ftol=1e-08,ref_time=tinit)
    AmpR_220R = AmpR[0,:]/perturbing
    AmpI_220R = AmpI[0,:]/perturbing
   
    if spin > 10**-8:
        # Fit 320
        file1 = np.loadtxt("%s/except320_filtered_data.csv"%foldername,delimiter=",", dtype=float)
        times = file1[:,0]
        Ref = file1[:,1]
        Imf = file1[:,2]

        time0s_320,final_mismatch_320,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin+0.032,ftol=1e-08,ref_time=tinit)
        AmpR_320 = AmpR[0,:]/perturbing
        AmpI_320 = AmpI[0,:]/perturbing

        # Fit 320R
        file1 = np.loadtxt("%s/except320R_filtered_data.csv"%foldername,delimiter=",", dtype=float)
        times = file1[:,0]
        Ref = file1[:,1]
        Imf = file1[:,2]

        time0s_320R,final_mismatch_320R,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.032,ftol=1e-08,ref_time=tinit)
        AmpR_320R = AmpR[0,:]/perturbing
        AmpI_320R = AmpI[0,:]/perturbing

    else:
        # Fit 320
        file1 = np.loadtxt("a=0.0_3evol_30/filtered_data_l3/l3_filtered_data.csv",delimiter=",", dtype=float)
        times = file1[:,0]
        Ref = file1[:,1]
        Imf = file1[:,2]

        time0s_320,final_mismatch_320,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin+0.032,ftol=1e-08,ref_time=tinit)
        AmpR_320 = AmpR[0,:]/perturbing
        AmpI_320 = AmpI[0,:]/perturbing

        # Fit 320R
        file1 = np.loadtxt("a=0.0_3evol_30/filtered_data_l3/l3R_filtered_data.csv",delimiter=",", dtype=float)
        times = file1[:,0]
        Ref = file1[:,1]
        Imf = file1[:,2]

        time0s_320R,final_mismatch_320R,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.032,ftol=1e-08,ref_time=tinit)
        AmpR_320R = AmpR[0,:]/perturbing
        AmpI_320R = AmpI[0,:]/perturbing

    
    to_save = np.zeros((time0s_220.size,6))
    to_save[:,0] = time0s_220
    to_save[:,1] = np.sqrt(AmpR_220**2+AmpI_220**2)/filter_fac[0]
    to_save[:,2] = np.sqrt(AmpR_221**2+AmpI_221**2)/filter_fac[1]
    to_save[:,3] = np.sqrt(AmpR_220R**2+AmpI_220R**2)/filter_fac[2]
    to_save[:,4] = np.sqrt(AmpR_320**2+AmpI_320**2)/filter_fac[3]
    to_save[:,5] = np.sqrt(AmpR_320R**2+AmpI_320R**2)/filter_fac[4]
    
    np.savetxt('%s/amp_fits.csv'%foldername, to_save, delimiter=',')
    #file1 = np.loadtxt("%s/amp_fits_newmode_w221.csv"%foldername,delimiter=",", dtype=float)
    #timesnm = file1[:,0]
    #Anm = file1[:,2]

    
    print('220 =', (np.sqrt(AmpR_320**2+AmpI_320**2)[10]))
    print('221 =', (np.sqrt(AmpR_320R**2+AmpI_320R**2)[10]))

    pp = PdfPages('./plots/fitted_amps_%f.pdf'%spin)
    plt.figure()
    plt.plot(time0s_220-tinit, np.log10((1/filter_fac[0])*np.sqrt(AmpR_220**2+AmpI_220**2)),label = '220')
    plt.plot(time0s_overtones-tinit, np.log10((1/filter_fac[1])*np.sqrt(AmpR_221**2+AmpI_221**2)*1.4925),label = '221')
    #plt.plot(time0s_overtones-tinit, np.log10((1/filter_fac[1])*np.sqrt(AmpR_222**2+AmpI_222**2)*1.206),label = '221')
    plt.plot(time0s_220R-tinit, np.log10((1/filter_fac[2])*np.sqrt(AmpR_220R**2+AmpI_220R**2)*1.4925),label = '220R')
    plt.plot(time0s_320-tinit, np.log10(1/(0.10065371160738153*filter_fac[3])*np.sqrt(AmpR_320**2+AmpI_320**2)*1.4925),label = '320')
    plt.plot(time0s_320R-tinit, np.log10(1/(0.06975824765089322*filter_fac[4])*np.sqrt(AmpR_320R**2+AmpI_320R**2)*1.4925),label = '320R')
    #plt.plot(timesnm-tinit, np.log10(Anm*1.4925),'--',label = 'new mode')
    #plt.plot(time0s_220[:index_cutofft(time0s_220,30)]-tinit,-2.5+0.038*time0s_220[:index_cutofft(time0s_220,30)],':',color='grey',label='$\exp(2 \omega_{220})$')
    #plt.axvspan(25, 30, alpha=0.5, color='silver')
    plt.ylim(-6,0)
    plt.xlim(0,80)
    plt.ylabel('$\log_{10}(A_{\ell m n})$')
    plt.xlabel('$t_0/M$')
    plt.legend(loc='upper right')
    #plt.title('Amplitudes fit from $t_0$, reference time $t=0$')
    pp.savefig()
    pp.close()


def fit_amps_20(foldername,tinit,spin,perturbing=1):
    # First fundamental mode
    file1 = np.loadtxt("%s/except200_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    index_init = index_cutofft(times,tinit)
    index_final = index_cutofft(times,100)
    Ref = file1[:,1]
    Ref200 = Ref
    Imf = file1[:,2]
    time0s_220,final_mismatch_220,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin,ftol=1e-08)
    AmpR_220 = AmpR[0,:]/perturbing
    AmpI_220 = AmpI[0,:]/perturbing

    # Fitting overtones
    file1 = np.loadtxt("%s/except3overtones_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    Ref201 = Ref
    time0s_overtones,final_mismatch_overtones,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=3,free_freq=False,tail=False,spin=spin+0.0221,ftol=1e-08)
    AmpR_221 = AmpR[0,:]/perturbing
    AmpI_221 = AmpI[0,:]/perturbing
    AmpR_222 = AmpR[1,:]/perturbing
    AmpI_222 = AmpI[1,:]/perturbing
    AmpR_223 = AmpR[2,:]/perturbing
    AmpI_223 = AmpI[2,:]/perturbing

    # Fitting 220R
    file1 = np.loadtxt("%s/except20R_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_220R,final_mismatch_220R,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.022,ftol=1e-08)
    AmpR_220R = AmpR[0,:]/perturbing
    AmpI_220R = AmpI[0,:]/perturbing

    # Fit extra 22nR?
    # and 320/320R?

    to_save = np.zeros((time0s_220.size,4))
    to_save[:,0] = time0s_220
    to_save[:,1] = np.sqrt(AmpR_220**2+AmpI_220**2)/0.2494661103
    to_save[:,2] = np.sqrt(AmpR_221**2+AmpI_221**2)/0.37163293949473075
    to_save[:,3] = np.sqrt(AmpR_220R**2+AmpI_220R**2)/0.3536365145847537

    np.savetxt('%s/amp_fits.csv'%foldername, to_save, delimiter=',')

    # Include factors for filtering

    pp = PdfPages('./plots/fitted_amps%f.pdf'%spin)
    plt.figure()
    plt.plot(time0s_220-tinit, np.log10(np.sqrt(AmpR_220**2+AmpI_220**2)/0.2494661103),label = '200')
    plt.plot(time0s_overtones-tinit, np.log10(np.sqrt(AmpR_221**2+AmpI_221**2)/0.37163293949473075),label = '201')
    #plt.plot(time0s_overtones[:index_cutofft(time0s_overtones,20)], np.log10(np.sqrt(AmpR_222**2+AmpI_222**2)[:index_cutofft(time0s_overtones,20)]),label = '202')
    plt.plot(time0s_220R-tinit, np.log10(np.sqrt(AmpR_220R**2+AmpI_220R**2)/0.3536365145847537),label = '200R')
    #plt.plot(times[index_init:index_final]-tinit,np.log10(np.abs(Ref200[index_init:index_final])))
    #plt.plot(times[index_init:index_final]-tinit,np.log10(np.abs(Ref201[index_init:index_final])))
    plt.ylabel('$\log A_{\ell m n}$')
    plt.xlabel('$t_0/M$')
    plt.xlim(0,50)
    plt.ylim(-3,0.0)
    plt.legend()
    plt.title('Amplitudes fit from $t_0$, reference time $t=0$')
    pp.savefig()
    pp.close()

def fit_amps_l(foldername,tinit,spin,perturbing=1,filter_fac = np.ones((5))):
    # l = 3
    file1 = np.loadtxt("%s/l3_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    index_init = index_cutofft(times,tinit)
    index_final = index_cutofft(times,100)
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_320,final_mismatch_320,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin+0.032,ftol=1e-08)
    AmpR_320 = AmpR[0,:]/perturbing
    AmpI_320 = AmpI[0,:]/perturbing

    file1 = np.loadtxt("%s/l3R_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_320R,final_mismatch_overtones,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.032,ftol=1e-08)    
    AmpR_320R = AmpR[0,:]/perturbing
    AmpI_320R = AmpI[0,:]/perturbing
    
    print('-------------------------------------------------l4')

    # l = 4
    file1 = np.loadtxt("%s/l4_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_420,final_mismatch_420,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin+0.042,ftol=1e-08)
    AmpR_420 = AmpR[0,:]/perturbing
    AmpI_420 = AmpI[0,:]/perturbing

    file1 = np.loadtxt("%s/l4R_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_420R,final_mismatch_420R,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.042,ftol=1e-08) 
    AmpR_420R = AmpR[0,:]/perturbing
    AmpI_420R = AmpI[0,:]/perturbing

    print('-------------------------------------------------l5')

    # l = 5
    file1 = np.loadtxt("%s/l5_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_520,final_mismatch_520,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin+0.052,ftol=1e-08)
    AmpR_520 = AmpR[0,:]/perturbing
    AmpI_520 = AmpI[0,:]/perturbing

    file1 = np.loadtxt("%s/l5R_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_520R,final_mismatch_520R,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.052,ftol=1e-08) 
    AmpR_520R = AmpR[0,:]/perturbing
    AmpI_520R = AmpI[0,:]/perturbing

    print('-------------------------------------------------l6')

    # l = 6
    file1 = np.loadtxt("%s/l6_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_620,final_mismatch_620,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin+0.062,ftol=1e-08)
    AmpR_620 = AmpR[0,:]/perturbing
    AmpI_620 = AmpI[0,:]/perturbing

    file1 = np.loadtxt("%s/l6R_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_620R,final_mismatch_overtones,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.062,ftol=1e-08) 
    AmpR_620R = AmpR[0,:]/perturbing
    AmpI_620R = AmpI[0,:]/perturbing

    print('-------------------------------------------------l7')

    # l = 7
    file1 = np.loadtxt("%s/l7_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_720,final_mismatch_620,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin+0.072,ftol=1e-08)
    AmpR_720 = AmpR[0,:]/perturbing
    AmpI_720 = AmpI[0,:]/perturbing

    file1 = np.loadtxt("%s/l7R_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_720R,final_mismatch_overtones,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.072,ftol=1e-08)
    AmpR_720R = AmpR[0,:]/perturbing
    AmpI_720R = AmpI[0,:]/perturbing

    print('-------------------------------------------------l8')
    """
    # l = 8
    file1 = np.loadtxt("%s/l8_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_820,final_mismatch_620,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin+0.082,ftol=1e-08)
    AmpR_820 = AmpR[0,:]/perturbing
    AmpI_820 = AmpI[0,:]/perturbing

    file1 = np.loadtxt("%s/l8R_filtered_data.csv"%foldername,delimiter=",", dtype=float)
    times = file1[:,0]
    Ref = file1[:,1]
    Imf = file1[:,2]
    time0s_820R,final_mismatch_overtones,AmpR,AmpI = vary_fit_time(times[index_init:index_final],Ref[index_init:index_final],Imf[index_init:index_final],num_modes=1,free_freq=False,tail=False,spin=spin-0.082,ftol=1e-08)
    AmpR_820R = AmpR[0,:]/perturbing
    AmpI_820R = AmpI[0,:]/perturbing
    """
    print(np.mean((1/filter_fac[0])*np.sqrt(AmpR_320**2+AmpI_320**2)),np.mean((1/filter_fac[1])*np.sqrt(AmpR_320R**2+AmpI_320R**2)),np.mean((1/filter_fac[2])*np.sqrt(AmpR_420**2+AmpI_420**2)),np.mean((1/filter_fac[3])*np.sqrt(AmpR_420R**2+AmpI_420R**2)),np.mean((1/filter_fac[4])*np.sqrt(AmpR_520**2+AmpI_520**2)),np.mean((1/filter_fac[5])*np.sqrt(AmpR_520R**2+AmpI_520R**2)),np.mean((1/filter_fac[6])*np.sqrt(AmpR_620**2+AmpI_620**2)),np.mean((1/filter_fac[7])*np.sqrt(AmpR_620R**2+AmpI_620R**2)),np.mean((1/filter_fac[8])*np.sqrt(AmpR_720**2+AmpI_720**2)),np.mean((1/filter_fac[9])*np.sqrt(AmpR_720R**2+AmpI_720R**2)))
    #,np.mean((1/filter_fac[10])*np.sqrt(AmpR_820**2+AmpI_820**2)),np.mean((1/filter_fac[11])*np.sqrt(AmpR_820R**2+AmpI_820R**2)))

    pp = PdfPages('./plots/fitted_amps.pdf')
    plt.figure()
    plt.plot(time0s_320-tinit, np.log10((1/filter_fac[0])*np.sqrt(AmpR_320**2+AmpI_320**2)),label = '320')
    plt.plot(time0s_320R-tinit, np.log10((1/filter_fac[1])*np.sqrt(AmpR_320R**2+AmpI_320R**2)),label = '320R')
    plt.plot(time0s_420-tinit, np.log10((1/filter_fac[2])*np.sqrt(AmpR_420**2+AmpI_420**2)),label = '420')
    plt.plot(time0s_420R-tinit, np.log10((1/filter_fac[3])*np.sqrt(AmpR_420R**2+AmpI_420R**2)),label = '420R')
    plt.plot(time0s_520-tinit, np.log10((1/filter_fac[4])*np.sqrt(AmpR_520**2+AmpI_520**2)),label = '520')
    plt.plot(time0s_520R-tinit, np.log10((1/filter_fac[5])*np.sqrt(AmpR_520R**2+AmpI_520R**2)),label = '520R')
    plt.plot(time0s_620-tinit, np.log10((1/filter_fac[6])*np.sqrt(AmpR_620**2+AmpI_620**2)),label = '620')
    plt.plot(time0s_620R-tinit, np.log10((1/filter_fac[7])*np.sqrt(AmpR_620R**2+AmpI_620R**2)),label = '620R')
    plt.plot(time0s_720-tinit, np.log10((1/filter_fac[8])*np.sqrt(AmpR_720**2+AmpI_720**2)),label = '720')
    plt.plot(time0s_720R-tinit, np.log10((1/filter_fac[9])*np.sqrt(AmpR_720R**2+AmpI_720R**2)),label = '720R')
    #plt.plot(time0s_820-tinit, np.log10((1/filter_fac[10])*np.sqrt(AmpR_820**2+AmpI_820**2)),label = '820')
    #plt.plot(time0s_820R-tinit, np.log10((1/filter_fac[11])*np.sqrt(AmpR_820R**2+AmpI_820R**2)),label = '820R')
    plt.ylabel('log Amplitudes')
    plt.xlabel('$t_0$')
    plt.legend()
    plt.title('Amplitudes fit from $t_0$, reference time $t=0$')
    pp.savefig()
    pp.close()

def reconstruct(fittedParameters,time0,norm_fac,times,num_modes=1,free_freq=False,changing_w=False,spin=0.7):
    #print('norm = ',norm_fac)
    fitR = np.zeros_like(times)
    fitI = np.zeros_like(times)
    wr,wi = _freqs(spin)
    if num_modes==1:
        #fitR = (fittedParameters[2]*np.exp(fittedParameters[1]*(times-time0))*np.cos(fittedParameters[0]*(times-time0)) - fittedParameters[3]*np.exp(fittedParameters[1]*(times-time0))*np.sin(fittedParameters[0]*(times-time0)))
        #fitI = (fittedParameters[3]*np.exp(fittedParameters[1]*(times-time0))*np.cos(fittedParameters[0]*(times-time0)) + fittedParameters[2]*np.exp(fittedParameters[1]*(times-time0))*np.sin(fittedParameters[0]*(times-time0)))
        if free_freq==True:
            wr[0] = fittedParameters[0]
            wi[0] = fittedParameters[1]
        Ar = fittedParameters[2]
        Ai = fittedParameters[3]
        if changing_w:
            DM = 0.00711759
            wr/=(1-DM)
            wi/=(1-DM)
            t0 = fittedParameters[0]
            #Q = fittedParameters[1]
            Q = fittedParameters[1] + 1j*fittedParameters[4]
            if num_modes==2:
                Q2 = inp[5] + 1j*inp[6]
                Ar2 = inp[7]
                Ai2 = inp[8]
        #print('mode amp =',np.sqrt(Ar**2+Ai**2)*np.exp((wi[0])*(-time0)))
        for i in range(times.size):
            if changing_w:
                dm = DM*(1-np.exp(2*wi[0]*(times[i]+t0)))
                phase = -(DM-1)*(wr[0]+1j*wi[0])*( cmath.log(-wi[0]*(cmath.exp(-2*wi[0]*times[i])-DM)) - cmath.log(-wi[0]*cmath.exp(-2*wi[0]*t0)+wi[0]*DM))/(2* wi[0])
                psi = (Ar+1j*Ai)*(1+Q*dm/DM)*cmath.exp(-1j*phase.real-phase.imag)
            else:
                psi = (Ar+1j*Ai)*cmath.exp((1j*wr[0]+wi[0])*(times[i]-time0))
            fitR[i] = psi.real
            fitI[i] = psi.imag
    if num_modes==2:
        for mode in range(2):
            print('mode amp =',np.sqrt(fittedParameters[mode]**2+fittedParameters[2+mode]**2)*np.exp((wi[mode])*(-time0))/norm_fac)
            for i in range(times.size):
                psi = (fittedParameters[mode]+1j*fittedParameters[2+mode])*cmath.exp((1j*wr[mode]+wi[mode])*(times[i]-time0))
                fitR[i] += psi.real
                fitI[i] += psi.imag
        if changing_w:
            Ar = fittedParameters[2]
            Ai = fittedParameters[3]
            DM = 0.00711759
            wr/=(1-DM)
            wi/=(1-DM)
            t0 = fittedParameters[0]
            #Q = fittedParameters[1]
            Q =0 
            #fittedParameters[1] + 1j*fittedParameters[4]
            Q2 =0
            #fittedParameters[5] + 1j*fittedParameters[6]
            Ar2 = fittedParameters[7]
            Ai2 = fittedParameters[8]
            #print('mode amp =',np.sqrt(Ar**2+Ai**2)*np.exp((wi[0])*(-time0)))
            for i in range(times.size):
                dm = DM*(1-np.exp(2*wi[0]*(times[i]+t0)))
                phase = -(DM-1)*(wr[0]+1j*wi[0])*( cmath.log(-wi[0]*(cmath.exp(-2*wi[0]*times[i])-DM)) - cmath.log(-wi[0]*cmath.exp(-2*wi[0]*t0)+wi[0]*DM))/(2* wi[0])
                phase2 = -(DM-1)*(wr[1]+1j*wi[1])*( cmath.log(-wi[1]*(cmath.exp(-2*wi[1]*times[i])-DM)) - cmath.log(-wi[1]*cmath.exp(-2*wi[1]*t0)+wi[1]*DM))/(2* wi[1])
                #print(phase.real,phase.imag)
                psi = (Ar+1j*Ai)*(1+Q*dm/DM)*cmath.exp(-1j*phase.real-phase.imag) + (Ar2+1j*Ai2)*(1+Q2*dm/DM)*cmath.exp(-1j*phase2.real-phase2.imag)
                fitR[i] = psi.real
                fitI[i] = psi.imag
    if num_modes>=3:
        for mode in range(num_modes):
            #fittedParameters[mode] = 1
            #fittedParameters[5+mode]=1
            #print('exp fac = ',wi[mode]*(times[0]-time0))
            #print(wr)
            for i in range(times.size):
                psi = (fittedParameters[mode]+1j*fittedParameters[num_modes+mode])*cmath.exp((1j*wr[mode]+wi[mode])*(times[i]-time0))
                fitR[i] += psi.real
                fitI[i] += psi.imag
            #fitR += (fittedParameters[mode]*np.exp(wi[mode]*(times-time0))*np.cos(wr[mode]*(times-time0)) - fittedParameters[5+mode]*np.exp(wi[mode]*(times-time0))*np.sin(wr[mode]*(times-time0)))/norm_fac
            #fitI += (fittedParameters[5+mode]*np.exp(wi[mode]*(times-time0))*np.cos(wr[mode]*(times-time0)) + fittedParameters[mode]*np.exp(wi[mode]*(times-time0))*np.sin(wr[mode]*(times-time0)))/norm_fac
    return fitR/norm_fac, fitI/norm_fac

def index_cutofft(times,cutoff):
    return np.argmin(np.abs(times-cutoff))

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'figure.autolayout': True})

#0.9
#[0.09412139785015725, 0.030220899253522136, 0.16493332476891212, 0.17031551027454994, 0.28253324088021115]
#0.8
#[0.09001133754301312, 0.02706393507139698,0.14719804607978063, 0.16300840918341233, 0.25586655959850024]
#0.75
#[0.08929612971383158,0.026579581938142034, 0.14180674849850866, 0.1612480353472615, 0.24729289686907766]
#0.7
# [0.08899300831198992,0.02634306705698163,0.13754482528767836,0.1601205490648968,0.2403656379655776]
# 0.6
#[0.0891054089743969, 0.026228816065367627, 0.13113925810246438,0.159075634626519,0.2297221661546744]
# 0.5
#[0.0897776484621301,0.02637663715699773,0.12653456324386342,0.15908062171938941,0.2219150025293779]
#0.4
#[0.09079907145411496,0.02670661299915439, 0.12312557805923781,0.1598008199123173,0.2160587751390988]
#0.3
#[0.09208882261502922,0.027204551085778877,0.12061684756542355,0.16107727181893314,0.21170586495639004]
#0.2
#[0.09362688677707293,0.02788533729077224,0.1188645759425573,0.16284089832809595,0.20863236939676788]
#0.1
#[0.0954347914889578, 0.028791443241960705, 0.11783790565007288, 0.16508153739091813, 0.20679338371554468]
#0.025
#[0.698455548774322, 0.02964772738799512, 0.11760661582815395, 0.16705306292680072,0.20635219072517091]
#0.0
#[0.09752122070759008, 0.0299526066279761, 0.11763921252685976, 0.12437110980172982*10, 0.1800715856587312*10]

#higher l
#[0.12437110980172982,0.1800715856587312,0.17825656069332882,0.23355923985323437,0.29547901724194564,0.3682357818336704,0.345641262622642,0.40687300378039126,0.3857681344780759,0.43547966655089226,0.41770199136022873,0.45726815098786816]

#fit_amps_20('spher_proj_30/filtered_data',4.39445,0.0,perturbing=1)
#fit_amps_20('spher_reltoa0_proj/filtered_data',4.39445,0.0,perturbing=1)
#fit_amps_20('spher_3evolinit_30/filtered_changing_freq_overtoneoptions',4.39445,0.0,perturbing=1)

#print(np.sqrt(Imf_LR[0]**2+Ref_LR[0]**2))
#fit_amps_22('a=0.8_3evol_30_MR/filtered_data_LR_allts',10.9638,0.8,perturbing=1,filter_fac =[0.09001133754301312, 0.02706393507139698,0.14719804607978063, 0.16300840918341233, 0.25586655959850024])
#fit_amps_22('a=0.7_hrevol_reltoextremal/filtered_data',8.912,0.7,perturbing=1,filter_fac =[0.08899300831198992,0.02634306705698163,0.13754482528767836,0.1601205490648968,0.2403656379655776])
#print(np.sqrt(Imf_LR[0]**2+Ref_LR[0]**2))

#tref = 8.91205
#wr,wi = _freqs(0.7)
#pert = (np.sqrt(Imf_LR[0]**2+Ref_LR[0]**2)*np.abs(cmath.exp(wi[0]*(tref))))

#fit_amps_l('a=0.7_hrevol_reltoextremal/filtered_data_ls',8.91205,0.7,perturbing=1,filter_fac =[0.12437110980172982,0.1800715856587312,0.17825656069332882,0.23355923985323437,0.29547901724194564,0.3682357818336704,0.345641262622642,0.40687300378039126,0.3857681344780759,0.43547966655089226,0.41770199136022873,0.45726815098786816])

#file1 = np.loadtxt("a=0.4_3evol_30/filtered_data_LR_allts/amp_fits.csv",delimiter=",", dtype=float)
#times_LR = file1[:,0]
#A221_LR = file1[:,5]
#file1 = np.loadtxt("a=0.4_3evol_30_HR/filtered_data/filtered_data/amp_fits.csv",delimiter=",", dtype=float)
#times_HR = file1[:,0]
#A221_HR = file1[:,5]

#file1 = np.loadtxt("a=0.4_3evol_30_MR/filtered_data_LR_allts/amp_fits.csv",delimiter=",", dtype=float)
#times_MR = file1[:,0]
#A221_MR = file1[:,5]
#A221 = file1[:,2]
#A220R = file1[:,3]
#A320 = file1[:,4]
#A320R = file1[:,5]
tref = 11+5
#print(A220[index_cutofft(times,tref)]," ",A221[index_cutofft(times,tref)]," ",A220R[index_cutofft(times,tref)]," ",A320[index_cutofft(times,tref)]," ",A320R[index_cutofft(times,tref)])
#print(A220[index_cutofft(times,tref)]," ",A221[index_cutofft(times,tref)]," ",A220R[index_cutofft(times,tref)])
tref = 9
#wr,wi = _freqs(0.7)
#print(np.sqrt(Imf_LR[0]**2+Ref_LR[0]**2)*np.abs(cmath.exp(wi[0]*(tref-times_LR[0]))))

#print((np.log(A221[index_cutofft(times,19)])-np.log(A221[index_cutofft(times,29)]))/(times[index_cutofft(times,19)]-times[index_cutofft(times,29)]))


#convergence_with_interp()

#print('Mismathc = ',Mismatch(times_LR, Ref_LR, Imf_LR, Ref_fix, Imf_fix)

#fit,time0,norm,fitR = vary_w(times_LR[index_cutofft(times_LR,2):index_cutofft(times_LR,100)],Ref_LR[index_cutofft(times_LR,2):index_cutofft(times_LR,100)],Imf_LR[index_cutofft(times_LR,2):index_cutofft(times_LR,100)],num_modes=1,free_freq=False,tail=False,spin=0.0,ftol=1e-08)

#fit = np.array([4.0,0,0,1.0,1.0])
#fit2,time02,norm2,fitR2 = fit_Mismatch(times_LR[index_cutofft(times_LR,100):],Ref_LR[index_cutofft(times_LR,100):],Imf_LR[index_cutofft(times_LR,100):],num_modes=1,spin=0.7)

#fit2,time02,norm2,fitR2 = fit_Mismatch(times_LR[index_cutofft(times_LR,2):index_cutofft(times_LR,100)],Ref_LR[index_cutofft(times_LR,2):index_cutofft(times_LR,100)],Imf_LR[index_cutofft(times_LR,2):index_cutofft(times_LR,100)],num_modes=2,spin=0.0)

#print(times_LR[10])
#print(times_MR[10])
#print(times_HR[10])

#phases_LR = omega_from_f(Ref_LR,Imf_LR,times_LR)
#data_1,slope_XLR = find_period(phases_LR,times_LR)
#amp_XLR,wi_XLR = find_omegaI(Ref_XLR,Imf_XLR,times_XLR)

#phases_fix = omega_from_f(Ref_fix,Imf_fix,times_fix)
#data_2,slope_HR = find_period(phases_fix,times_fix)
#amp_HR,wi_HR = find_omegaI(Ref_HR,Imf_HR,times_HR)

#phases_MR = omega_from_f(Ref_3,Imf_3,times_3)
#data_3,slope_MR = find_period(phases_MR,times_3)
#amp_MR,wi_MR = find_omegaI(Ref_MR[1000:],Imf_MR[1000:],times_MR[1000:])

#from scipy import interpolate
#Ref_f = interpolate.interp1d(times_fix, Ref_fix)
#Imf_f = interpolate.interp1d(times_fix, Imf_fix)
#Ref_fix2 = Ref_f(times_LR[:index_cutofft(times_LR,60)])*np.sqrt(Ref_LR[0]**2+Imf_LR[0]**2)/np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2)
#Imf_fix2 = Imf_f(times_LR[:index_cutofft(times_LR,60)])*np.sqrt(Ref_LR[0]**2+Imf_LR[0]**2)/np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2)


#phases_MR = omega_from_f(Ref_fix2,Imf_fix2,times_LR[:index_cutofft(times_LR,60)])
#data_fix,slope_MR = find_period(phases_MR,times_LR[:index_cutofft(times_LR,60)])
#amp_MR,wi_MR = find_omegaI(Ref_MR[300:],Imf_MR[300:],times[300:])

#frequency_evolution(times_LR,Ref_LR,Imf_LR)

#print('num time steps = ',times_LR.size, ' real ',Ref_LR.size,)

#overtone_errs = self_convergence(Ref_LR[index_cutofft(times_LR,6):index_cutofft(times_LR,100)],Ref_MR[index_cutofft(times_MR,6):index_cutofft(times_MR,100)],Ref_HR[index_cutofft(times_HR,6):index_cutofft(times_HR,100)],Imf_LR[index_cutofft(times_LR,6):index_cutofft(times_LR,100)],Imf_MR[index_cutofft(times_MR,6):index_cutofft(times_MR,100)],Imf_HR[index_cutofft(times_HR,6):index_cutofft(times_HR,100)])
#print(times_LR[:10])
#print(times_MR[:10])
#print(times_HR[:10])
#,times1=times_LR[:97252],times2=times_MR[:97252])

#phases_MR = omega_from_f(Ref_MR,Imf_MR,times_MR)
#data_MR,slope_MR = find_period(phases_MR,times_MR)
#amp_MR,wi_MR = find_omegaI(Ref_MR,Imf_MR,times_MR)

phases_LR = omega_from_f(Ref_LR,Imf_LR,times_LR)
data_LR,slope_LR = find_period(phases_LR,times_LR)
#amp_LR,wi_LR = find_omegaI(Ref_LR,Imf_LR,times_LR)

#vary_fit_time(times_LR,Ref_LR,Imf_LR,num_modes=1)

#fit,time0,norm,fitR = fit_Mismatch(times_l3[10000:],Ref_l3[10000:],Imf_3[10000:],num_modes=1,free_freq=False)
#fit,time0,norm,fitR = fit_Mismatch(times_LR[9000:],Ref_3[9000:],Imf_3[9000:],num_modes=3,free_freq=False)
#fit,time0,norm,fitR = fit_Mismatch(times_LR[index_cutofft(times_LR,60):],Ref_LR[index_cutofft(times_LR,60):],Imf_LR[index_cutofft(times_LR,60):],num_modes=2,free_freq=False,spin=0.0)
#fit_Mismatch(times_LR[35000:],Ref_LR[30000:],Imf_LR[35000:],num_modes=1)

#data,slope_LR = find_period(phases_LR[63000:],times_LR[63000:])
#amp,wi_LR = find_omegaI(Ref_LR[63000:],Imf_LR[63000:],times_LR[63000:])

#Re22_LR, Im22_LR = reconstruct(fit,time0,norm,times_LR[:],num_modes=1,changing_w=True,spin=0.0)
#Re22_LRl, Im22_LRl = reconstruct(fit,time0,norm,times_LR[:],num_modes=1,changing_w=True,spin=0.0)
#to_save1 = np.zeros((times_LR.size,2))
#to_save2 = np.zeros((times_LR.size,2))
#to_save1[:,0] = times_LR
#to_save1[:,1] = Re22_LR
#to_save2[:,0] = times_LR
#to_save2[:,1] = Im22_LR
#np.savetxt("spher_3evolinit_30/Re_201changing_w.csv",to_save1,delimiter=",")
#np.savetxt("spher_3evolinit_30/Im_201changing_w.csv",to_save2,delimiter=",")

#Re22_LR2, Im22_LR2 = reconstruct(fit2,time02,norm2,times_LR[:index_cutofft(times_LR,101)],num_modes=2,spin=0.0)
#Re22_LR2, Im22_LR2 = reconstruct(fit2,time02,norm2,times_LR[:index_cutofft(times_LR,101)],num_modes=1,spin=0.7)

#nt = times_LR.size
#print(times_HR[-1]," ",times_LR[-1]," ",times_fix[-1])

#t01, amp1 = plot_vary_fit(times_LR[:index_cutofft(times_LR,100)],Ref_LR[:index_cutofft(times_LR,100)],Imf_LR[:index_cutofft(times_LR,100)],times_fix[:index_cutofft(times_LR,times_LR[-1])],Ref_fix[:index_cutofft(times_LR,times_LR[-1])],Imf_fix[:index_cutofft(times_LR,times_LR[-1])],num_modes=4,free_freq=False,tail=False,spin=0.0)

#fit,time0,norm,fitR = fit_Mismatch(times_LR[index_cutofft(times_LR,0):],Ref_LR[index_cutofft(times_LR,0):],Imf_LR[index_cutofft(times_LR,0):],num_modes=3,free_freq=False,spin=0.7)
#print(fit,time0,norm,fitR)
#fit,time0,norm,fitR = fit_Mismatch(times_MR[index_cutofft(times_MR,0)+1:],Ref_MR[index_cutofft(times_MR,0)+1:],Imf_MR[index_cutofft(times_MR,0)+1:],num_modes=3,free_freq=False,spin=0.7)
#print(fit,time0,norm,fitR)
#fit,time0,norm,fitR = fit_Mismatch(times_HR[index_cutofft(times_HR,0)+3:],Ref_HR[index_cutofft(times_HR,0)+3:],Imf_HR[index_cutofft(times_HR,0)+3:],num_modes=3,free_freq=False,spin=0.7)
#print(fit,time0,norm,fitR)

#print(times_LR[index_cutofft(times_LR,0)],times_MR[index_cutofft(times_MR,0)+1],times_HR[index_cutofft(times_HR,0)+3])
#Re22_LR, Im22_LR = reconstruct(fit,time0,norm,times_LR[:],num_modes=3,free_freq=False,spin=0.7)

#t02, amp2 = plot_vary_fit(times_LR[index_cutofft(times_LR,9):index_cutofft(times_LR,100)],Ref_LR[index_cutofft(times_LR,9):index_cutofft(times_LR,100)],Imf_LR[index_cutofft(times_LR,9):index_cutofft(times_LR,100)],times_LR[:index_cutofft(times_LR,times_LR[-1])],Ref_LR[:index_cutofft(times_LR,times_LR[-1])],Imf_LR[:index_cutofft(times_LR,times_LR[-1])],num_modes=1,free_freq=True,tail=False,spin=0.7)


#t02, amp2 = plot_vary_fit(times_LR[index_cutofft(times_LR,0):index_cutofft(times_LR,80)],Ref_LR[index_cutofft(times_LR,0):index_cutofft(times_LR,80)],Imf_LR[index_cutofft(times_LR,0):index_cutofft(times_LR,80)],times_LR[:index_cutofft(times_LR,times_LR[-1])],Ref_LR[:index_cutofft(times_LR,times_LR[-1])],Imf_LR[:index_cutofft(times_LR,times_LR[-1])],num_modes=1,free_freq=True,tail=False,spin=0.7)

#plot_vary_fit(times_LR[index_cutofft(times_LR,5):index_cutofft(times_LR,100)],Ref_LR[index_cutofft(times_LR,5):index_cutofft(times_LR,100)],Imf_LR[index_cutofft(times_LR,5):index_cutofft(times_LR,100)],times_fix[:index_cutofft(times_fix,10)],Ref_fix[:index_cutofft(times_fix,90)],Imf_fix[:index_cutofft(times_fix,90)],num_modes=3,free_freq=False,tail=False,spin=0.0)

#fit,time0,norm,fitR = fit_Mismatch(times_LR[index_cutofft(times_LR,47):],Ref_LR[index_cutofft(times_LR,47):]-Re22_LR,Imf_LR[index_cutofft(times_LR,47):]-Im22_LR,num_modes=1,free_freq=False,spin=0.9221)

#wr,wi = _freqs(0.9221)
#Ar = fit[2]
#Ai = fit[3]
#print('mode amp =',np.sqrt(Ar**2+Ai**2)*np.exp((wi[0])*(-time0))/norm)

#plot_vary_fit(times_LR[:index_cutofft(times_LR,100)],Ref_LR[:index_cutofft(times_LR,100)]-Re22_LR,Imf_LR[:index_cutofft(times_LR,100)]-Im22_LR,times_fix[:index_cutofft(times_fix,100)],Ref_fix[:index_cutofft(times_fix,100)],Imf_fix[:index_cutofft(times_fix,100)],num_modes=1,free_freq=True,tail=False,spin=0.0)

#print(times_HR[-1],times_LR[index_cutofft(times_LR,times_HR[-1])],times_fix[index_cutofft(times_LR,times_HR[-1])])

#plot_vary_fit(times_LR[:30000],Ref_LR[:30000],Imf_LR[:30000],times_HR[:30000],Ref_HR[:30000],Imf_HR[:30000],num_modes=4,free_freq=False)
#plot_vary_fit(times_MR[:],Ref_MR[:],Imf_MR[:],times_HR[:times_MR.size],Ref_HR[:times_MR.size],Imf_HR[:times_MR.size],num_modes=5,free_freq=False,err = overtone_errs)

#plot_vary_fit(times_LR[:index_cutofft(times_LR,60)],Ref_LR[:index_cutofft(times_LR,60)]-Re22_LR,Imf_LR[:index_cutofft(times_LR,60)]-Im22_LR,times_HR[:index_cutofft(times_LR,60)],Ref_HR[:index_cutofft(times_LR,60)],Imf_HR[:index_cutofft(times_LR,60)],num_modes=1,free_freq=True,spin=0.9)

#plot_vary_fit(times_LR[:],Ref_LR[:]-Re22_LR,Imf_LR[:]-Im22_LR,times_HR[:index_cutofft(times_LR,60)],Ref_HR[:index_cutofft(times_LR,60)],Imf_HR[:index_cutofft(times_LR,60)],num_modes=1,free_freq=True,spin=0.7)

#plot_vary_fit(times_LR[:],Ref_LR[:],Imf_LR[:],times_HR[:times_MR.size],Ref_HR[:times_MR.size],Imf_HR[:times_MR.size],num_modes=5,free_freq=False,err = overtone_errs)

#plot_vary_fit(times_l3[:28724],Ref_l3[:28724],Imf_l3[:28724],times_fix[:28724],Ref_fix[:28724],Imf_fix[:28724],num_modes=1,free_freq=True)

#fit,time0,norm,fitR = fit_Mismatch(times_LR[100000:],Ref_LR[100000:],Imf_LR[100000:],num_modes=1,free_freq=True)
#Re22_LR, Im22_LR = reconstruct(fit,time0,norm,times_LR[:],num_modes=3,free_freq=False)
#fit,time0,norm = fit_Mismatch(times_LR[5000:],(Ref_LR-Re22_LR)[5000:],(Imf_LR-Im22_LR)[5000:],num_modes=1)
#reconstruct_mode(times_LR,wi_LR,slope_LR,phases_LR,amp_LR)
#phases22_LR = omega_from_f(Re22_LR,Im22_LR,times_LR)
#data22_LR,slope22_LR = find_period(phases22_LR,times_LR)
#amp22_LR,wi22_LR = find_omegaI(Re22_LR,Im22_LR,times_LR)

#phases_sub = omega_from_f((Ref_LR-Re22_LR)[:],(Imf_LR-Im22_LR)[:],times_LR[:])
#data_sub,slope_sub = find_period(phases_sub,times_LR[:])
#amp_sub,wi_sub = find_omegaI((Ref_LR-Re22_LR)[:],(Imf_LR-Im22_LR)[:],times_LR[:])

#frequency_evolution(times_LR[:5000],(Ref_LR-Re22_LR)[:5000],(Imf_LR-Re22_LR)[:5000])


#phases_HR = omega_from_f(Ref_HR,Imf_HR,times_HR)
#data_HR,slope_HR = find_period(phases_HR,times_HR)
#amp_HR = find_omegaI(Ref_HR,Imf_HR,times_HR)

# convergence factor for result
#wr =  -0.532187plt.plot(times_LR,Ref_LR,label='physical')9751341276  with err  0.0006461247390380805

#print('convergence = ',np.log((slope_MR+ 0.5326002435510183)/(slope_HR+0.5326002435510183))/np.log(2))
#print('convergence = ',np.log((wi_MR+0.0807929627407481 )/(wi_HR+0.0807929627407481))/np.log(2))

#print((np.log(np.abs(slope_HR - slope_MR) /np.abs(slope_HR-slope_XHR)))/np.log(2))
#print((np.log(np.abs(wi_HR - wi_MR)/np.abs(wi_XHR-wi_HR)))/np.log(2))

#print((np.log(np.abs(slope_MR - slope_HR)/ np.abs(slope_XHR-slope_HR)))/np.log(2))
#print((np.log(np.abs(wi_HR - wi_MR)**-1 *np.abs(wi_XHR-wi_HR)))/np.log(2))

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})


#pp = PdfPages('./plots/logAmp.pdf')
plt.figure()
#plt.plot(rs_ID,cheb_ID,label='cheb reconstructed')
#plt.plot(rs_ID,ID,label='vr solved')
#plt.plot(times,modelPredictions)
#plt.plot(t01,amp1,label='221 Amp')
#plt.plot(t02,amp2,':',label='high res')
#plt.plot(times,phases,'o',label='logf')
#plt.plot(times,data,'o')
#plt.plot(timesLRn,data2,'o',label='LR')
#plt.plot(timesLRn,ImfLRn,label='Imf')
#plt.plot(times,Imf)
#plt.plot(times_LR,np.log(amp_LR),label='full')
#plt.plot(times_LR,np.log(amp22_LR),label='dominant')
#plt.plot(times_LR[4500:],np.log(amp_sub)[4500:],label='sub-dom')
#plt.plot(times_MR,np.log(amp_MR),label='horizon')
#plt.plot(timesHR,np.log(ampHR),label='HR')
#plt.plot(times,-5-times*0.08,label='slope wi')
plt.legend()
plt.ylabel('Log Amp')
plt.xlabel('t')
#pp.savefig()
#pp.close()

#print(data_2)
#print(data_2.size)
#print(index_cutofft(times_2,50),index_cutofft(times_2,75))
#print(np.mean(data_1[index_cutofft(times_LR,50):index_cutofft(times_LR,80)]-data_fix[index_cutofft(times_LR,50):index_cutofft(times_LR,80)]))
#print(np.mean(data_2[index_cutofft(times_2,50):index_cutofft(times_2,75)]-data_fix[index_cutofft(times_2,50):index_cutofft(times_2,75)]))
#print(np.mean(data_LR[index_cutofft(times_LR,50):index_cutofft(times_LR,60)]-data_fix[index_cutofft(times_LR,50):index_cutofft(times_LR,60)]))


pp = PdfPages('./plots/phases.pdf')
plt.figure()
#plt.plot(times_LR[:index_cutofft(times_LR,60)],data_LR[:index_cutofft(times_LR,60)]-data_fix[:index_cutofft(times_LR,60)],label='$1\\times\Delta M/M$')
#plt.plot(times_2[:index_cutofft(times_2,80)],data_2[:index_cutofft(times_2,80)]-data_fix[:index_cutofft(times_2,80)],'--',label='$2\\times\Delta M/M$')
#plt.plot(times_3[:index_cutofft(times_LR,80)],data_3[:index_cutofft(times_LR,80)]-data_fix[:index_cutofft(times_LR,80)],'-.',label='$3\\times\Delta M/M$')
#plt.plot(times_XLR,data_XLR,'o',label='XLR')
#plt.plot([6.75,6.75],[-0.015,0.1],linestyle=':',color='k')
plt.legend()
plt.xlim(0,60)
plt.ylabel('$\Delta \phi$')
plt.xlabel('t/M')
pp.savefig()
pp.close()

from scipy import interpolate
Ref_f = interpolate.interp1d(times_fix, Ref_fix)
Imf_f = interpolate.interp1d(times_fix, Imf_fix)
Ref_fix2 = Ref_f(times_LR[:index_cutofft(times_LR,80)])
#*np.sqrt(Ref_LR[0]**2+Imf_LR[0]**2)/np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2)
Imf_fix2 = Imf_f(times_LR[:index_cutofft(times_LR,80)])
#*np.sqrt(Ref_LR[0]**2+Imf_LR[0]**2)/np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2)
Ref_LR*=(np.sqrt(Ref_LR[0]**2+Imf_LR[0]**2)/np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2))**-1
Imf_LR*=(np.sqrt(Ref_LR[0]**2+Imf_LR[0]**2)/np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2))**-1


Ref_f = interpolate.interp1d(times_2, Ref_2)
Imf_f = interpolate.interp1d(times_2, Imf_2)
Ref_2n = Ref_f(times_LR[:index_cutofft(times_LR,80)])*np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2)/np.sqrt(Ref_2[0]**2+Imf_2[0]**2)
Imf_2n = Imf_f(times_LR[:index_cutofft(times_LR,80)])*np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2)/np.sqrt(Ref_2[0]**2+Imf_2[0]**2)

Ref_f = interpolate.interp1d(times_3, Ref_3)
Imf_f = interpolate.interp1d(times_3, Imf_3)
Ref_3n = Ref_f(times_LR[:index_cutofft(times_LR,80)])*np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2)/np.sqrt(Ref_3[0]**2+Imf_3[0]**2)
Imf_3n = Imf_f(times_LR[:index_cutofft(times_LR,80)])*np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2)/np.sqrt(Ref_3[0]**2+Imf_3[0]**2)

#phases_LR = omega_from_f(Ref_fix2,Imf_fix2,times_LR[:index_cutofft(times_LR,80)])
#data_fix,slope_LR = find_period(phases_LR,times_LR[:index_cutofft(times_LR,80)])

#dR1 = ((Ref_fix2[:index_cutofft(times_LR,90)])-Ref_LR[:index_cutofft(times_LR,90)])*(1/0.67)/np.sqrt(Ref_LR[index_cutofft(times_LR,4)]**2+Imf_LR[index_cutofft(times_LR,4)]**2)
#dI1 = ((Imf_fix2[:index_cutofft(times_LR,90)])-Imf_LR[:index_cutofft(times_LR,90)])*(1/0.67)/np.sqrt(Ref_LR[index_cutofft(times_LR,4)]**2+Imf_LR[index_cutofft(times_LR,4)]**2)
#h12 = (dR1**2+ dI1**2)
#result = np.trapz(h12[index_cutofft(times_LR,4):], x=times_LR[index_cutofft(times_LR,4):index_cutofft(times_LR,90)])
#print('SNR 1 =',result**0.5)
pp = PdfPages('./plots/Repsi02.pdf')
plt.figure()
# integrating twice for h
from scipy import integrate
#plt.plot(times_LR[:index_cutofft(times_LR,80)],Ref_fix2[:index_cutofft(times_LR,80)])
analytic_int = np.zeros_like(times_LR[:index_cutofft(times_LR,80)])
for i in range(index_cutofft(times_LR,80)):
    h_t = -(Ref_fix2[i])/(-0.532600243551018 -1j*0.08079287315500867)**2
    #(Ref_fix2[i]+1j*Imf_fix2[i])/(-0.532600243551018 -1j*0.08079287315500867)**2
    analytic_int[i] = h_t.real
plt.plot(times_LR[:index_cutofft(times_LR,80)],analytic_int,label='analytic twice int')
Ref_fix2 = integrate.cumtrapz(Ref_fix2, times_LR[:index_cutofft(times_LR,80)], initial=0)
Imf_fix2 = integrate.cumtrapz(Imf_fix2, times_LR[:index_cutofft(times_LR,80)], initial=0)
Ref_fix2-=Ref_fix2[-1]
Imf_fix2-=Imf_fix2[-1]
#plt.plot(times_LR[:index_cutofft(times_LR,80)],Ref_fix2[:index_cutofft(times_LR,80)],label='once int')
Ref_fix2 = integrate.cumtrapz(Ref_fix2, times_LR[:index_cutofft(times_LR,80)], initial=0)
Imf_fix2 = integrate.cumtrapz(Imf_fix2, times_LR[:index_cutofft(times_LR,80)], initial=0)
Ref_fix2-=Ref_fix2[-1]
Imf_fix2-=Imf_fix2[-1]
plt.plot(times_LR[:index_cutofft(times_LR,80)],Ref_fix2[:index_cutofft(times_LR,80)],label='twice int')

Ref_LR = integrate.cumtrapz(Ref_LR[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Imf_LR = integrate.cumtrapz(Imf_LR[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Ref_LR-=Ref_LR[-1]
Imf_LR-=Imf_LR[-1]
#plt.plot(times_LR[:index_cutofft(times_LR,80)],Ref_LR[:index_cutofft(times_LR,80)],label='once int')

Ref_LR = integrate.cumtrapz(Ref_LR[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Imf_LR = integrate.cumtrapz(Imf_LR[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Ref_LR-=Ref_LR[-1]
Imf_LR-=Imf_LR[-1]
plt.plot(times_LR[:index_cutofft(times_LR,80)],Ref_LR[:index_cutofft(times_LR,80)],label = 'dM twice int')


Ref_2n = integrate.cumtrapz(Ref_2n[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Imf_2n = integrate.cumtrapz(Imf_2n[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Ref_2n-=Ref_2n[-1]
Imf_2n-=Imf_2n[-1]
Ref_2n = integrate.cumtrapz(Ref_2n[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Imf_2n = integrate.cumtrapz(Imf_2n[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Ref_2n-=Ref_2n[-1]
Imf_2n-=Imf_2n[-1]

Ref_3n = integrate.cumtrapz(Ref_3n[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Imf_3n = integrate.cumtrapz(Imf_3n[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Ref_3n-=Ref_3n[-1]
Imf_3n-=Imf_3n[-1]
Ref_3n = integrate.cumtrapz(Ref_3n[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Imf_3n = integrate.cumtrapz(Imf_3n[:index_cutofft(times_LR,80)], times_LR[:index_cutofft(times_LR,80)], initial=0)
Ref_3n-=Ref_3n[-1]
Imf_3n-=Imf_3n[-1]

print(times_LR[-1],times_LR[-2600])

fit2,time02,norm2,fitR2 = fit_Mismatch(times_LR[index_cutofft(times_LR,60):index_cutofft(times_LR,80)],Ref_LR[index_cutofft(times_LR,60):index_cutofft(times_LR,80)],Imf_LR[index_cutofft(times_LR,60):index_cutofft(times_LR,80)],num_modes=1,spin=0.7)
Re22_LR2, Im22_LR2 = reconstruct(fit2,time02,norm2,times_LR[:index_cutofft(times_LR,101)],num_modes=1,spin=0.7)

dR1 = ((Re22_LR2[:index_cutofft(times_LR,80)])-Ref_LR[:index_cutofft(times_LR,80)])*(1/0.67)/np.sqrt(Ref_LR[index_cutofft(times_LR,4)]**2+Imf_LR[index_cutofft(times_LR,4)]**2)
dI1 = ((Im22_LR2[:index_cutofft(times_LR,80)])-Imf_LR[:index_cutofft(times_LR,80)])*(1/0.67)/np.sqrt(Ref_LR[index_cutofft(times_LR,4)]**2+Imf_LR[index_cutofft(times_LR,4)]**2)
#dR1 = ((Ref_fix2[:index_cutofft(times_LR,90)])-Ref_LR[:index_cutofft(times_LR,90)])*(1/0.67)/np.sqrt(Ref_LR[index_cutofft(times_LR,9)]**2+Imf_LR[index_cutofft(times_LR,9)]**2)
#dI1 = ((Imf_fix2[:index_cutofft(times_LR,90)])-Imf_LR[:index_cutofft(times_LR,90)])*(1/0.67)/np.sqrt(Ref_LR[index_cutofft(times_LR,9)]**2+Imf_LR[index_cutofft(times_LR,9)]**2)

h12 = (dR1**2+ dI1**2)

result0 = Mismatch(times_LR[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Ref_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Imf_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Ref_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Imf_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)])
result1 = Mismatch(times_LR[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Ref_LR[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Imf_LR[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Ref_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Imf_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)])
result2 = Mismatch(times_LR[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Ref_2n[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Imf_2n[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Ref_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Imf_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)])
result3 = Mismatch(times_LR[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Ref_3n[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Imf_3n[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Ref_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)], Imf_fix2[index_cutofft(times_LR,4):index_cutofft(times_LR,60)])

result = np.trapz(h12[index_cutofft(times_LR,4):index_cutofft(times_LR,80)], x=times_LR[index_cutofft(times_LR,4):index_cutofft(times_LR,80)])
print('SNR integral = ',result)

print('Mismatch =',result0, " ",result1, " ",result2, " ",result3)
#print(result1)
#print(index_cutofft(times_LR,6)-index_cutofft(times_LR,100))
#print(index_cutofft(times_MR,6)-index_cutofft(times_MR,100))
#Amr = interpolate.interp1d(times_MR, A221_MR)
#Ahr = interpolate.interp1d(times_HR, A221_HR)

#pp = PdfPages('./plots/Repsi02.pdf')
#plt.figure()
#plt.plot(times_LR[:index_cutofft(times_LR,79)],Ref_LR[:index_cutofft(times_LR,79)]-Ref_fix2[:index_cutofft(times_LR,79)],label='$dm=1$')
#plt.plot(times_LR[:index_cutofft(times_LR,79)],Ref_2n[:index_cutofft(times_LR,79)]-Ref_fix2[:index_cutofft(times_LR,79)],label='$dm=2$')
#plt.plot(times_LR[:index_cutofft(times_LR,79)],Ref_3n[:index_cutofft(times_LR,79)]-Ref_fix2[:index_cutofft(times_LR,79)],label='$dm=3$')
#plt.plot(times_LR[:index_cutofft(times_LR,79)],0*times_LR[:index_cutofft(times_LR,79)])
#plt.plot(times_LR[:index_cutofft(times_LR,79)],data_LR[:index_cutofft(times_LR,79)]-data_fix[:index_cutofft(times_LR,79)],label='$dm=1$')
#plt.plot(times_LR[:index_cutofft(times_LR,79)],Ref_fix2[:index_cutofft(times_LR,79)],label='fix')
#plt.plot(times_LR[1:-10],A221_LR[1:-10]-Amr(times_LR[1:-10]),label='LR')
#plt.plot(times_LR[1:-10],-16*(Amr(times_LR[1:-10])-Ahr(times_LR[1:-10])),label='MR')
#plt.plot(times_LR[1:-10],np.log(np.abs((A221_LR[1:-10]-Amr(times_LR[1:-10]))/(Amr(times_LR[1:-10])-Ahr(times_LR[1:-10])))))
#plt.plot(times_HR,np.log10(A221_HR),label='HR')
#plt.plot(times_LR[index_cutofft(times_LR,6):index_cutofft(times_LR,100)],Ref_LR[index_cutofft(times_LR,6):index_cutofft(times_LR,100)],label='LR')
#plt.plot(times_LR[index_cutofft(times_MR,6):index_cutofft(times_MR,100)],Ref_MR[index_cutofft(times_MR,6):index_cutofft(times_MR,100)],label='MR')
#plt.plot(times_LR[index_cutofft(times_HR,6):index_cutofft(times_HR,100)],Ref_HR[index_cutofft(times_HR,6):index_cutofft(times_HR,100)],label='HR')
#plt.plot(times_3[:index_cutofft(times_2,80)],(np.sqrt((Ref_3[:index_cutofft(times_2,80)]-Ref_fix2[:index_cutofft(times_2,80)])**2+(Imf_3[:index_cutofft(times_2,80)]-Imf_fix2[:index_cutofft(times_2,80)])**2))/(np.sqrt(Ref_fix2[:index_cutofft(times_2,80)]**2+Imf_fix2[:index_cutofft(times_2,80)]**2)),label='residual')
#plt.plot(times_2[:index_cutofft(times_2,80)], np.log(np.abs(Ref_fix2[:index_cutofft(times_2,80)])))
#plt.plot(times_2[:index_cutofft(times_2,80)], np.log(np.abs(Ref_2[:index_cutofft(times_2,80)])))
#plt.plot(times_LR[:index_cutofft(times_LR,100)],-((phases_LR[:index_cutofft(times_LR,100)]))+np.pi,label='changing bgd.')
#plt.plot(times_fix[:index_cutofft(times_fix,100)],np.sqrt(Ref_LR[0]**2+Imf_LR[0]**2)/np.sqrt(Ref_fix[0]**2+Imf_fix[0]**2)*((Ref_fix[:index_cutofft(times_fix,100)])),'--',label='fixed remnant bgd.')
#plt.plot(times_LR[:index_cutofft(times_LR,100)],np.sqrt(Ref_LR[0]**2+Imf_LR[0]**2)/np.sqrt(Re22_LR2[0]**2+Im22_LR2[0]**2)*((Re22_LR2[:index_cutofft(times_LR,100)])),'--',label='fixed remnant bgd. 2')
#plt.plot(times_LR[:index_cutofft(times_LR,90)],((Ref_fix2[:index_cutofft(times_LR,90)])-Ref_LR[:index_cutofft(times_LR,90)])*(1/0.67)/np.sqrt(Ref_LR[index_cutofft(times_LR,9)]**2+Imf_LR[index_cutofft(times_LR,9)]**2),':',label='difference')
#plt.plot(times_LR[:index_cutofft(times_LR,90)],((Re22_LR2[:index_cutofft(times_LR,90)])-Ref_LR[:index_cutofft(times_LR,90)])*(1/0.67)/np.sqrt(Ref_LR[index_cutofft(times_LR,9)]**2+Imf_LR[index_cutofft(times_LR,9)]**2),':',label='difference 2')
#plt.plot(times_fix[:index_cutofft(times_fix,40)],-((phases_fix[:index_cutofft(times_fix,40)]))+np.pi,'--',label='fixed bgd.')
#plt.plot(times_fix[:index_cutofft(times_fix,20)],(np.abs(np.sqrt(Ref_fix[:index_cutofft(times_fix,20)]**2+Imf_fix[:index_cutofft(times_fix,20)]**2)-np.sqrt(Ref_LR[:index_cutofft(times_LR,20)]**2+Imf_LR[:index_cutofft(times_LR,20)]**2)))/(np.sqrt(Ref_fix[:index_cutofft(times_fix,20)]**2+Ref_fix[:index_cutofft(times_fix,20)]**2)),label='changing bgd.')

#plt.plot(times_fix[:index_cutofft(times_fix,100)],(np.sqrt(np.abs(Ref_fix[:index_cutofft(times_fix,100)]-Ref_LR[:index_cutofft(times_fix,100)])**2 + np.abs(Imf_fix[:index_cutofft(times_fix,100)]-Imf_LR[:index_cutofft(times_fix,100)])**2)/np.sqrt(Ref_fix[:index_cutofft(times_fix,100)]**2+Imf_fix[:index_cutofft(times_fix,100)]**2)),label='changing bgd.')

#plt.plot(times_LR[:],np.log(np.sqrt(Ref_LR**2+Imf_LR**2)),label='final qnm, smooth')
#plt.plot(times_proj[:10000],Ref_proj[:10000],label='initial qnm, proj')
#plt.plot(times_evoinit[:10000],Ref_evoinit[:10000],label='initial qnm, smooth')
#plt.plot(times_2[:20000],Ref_2[:20000],label='changing background 2')
#plt.plot(times_3[:20000],Ref_3[:20000],label='changing background 3')
#plt.plot(times_HR[:20000],((Ref_HR[:20000])),label='fixed background')

#plt.plot(times_LR[:index_cutofft(times_LR,100)]-2,np.log10(np.abs(Re22_LR[:index_cutofft(times_LR,100)]-Ref_LR[:index_cutofft(times_LR,100)])),'--',label='changing freq')
#plt.plot(times_LR[:index_cutofft(times_LR,100)]-2,np.log10(np.abs(Re22_LR2[:index_cutofft(times_LR,100)]-Ref_LR[:index_cutofft(times_LR,100)])),label='200+201')

#plt.plot(times_LR[:],np.log(np.sqrt(Ref_LR**2+Imf_LR**2)),label='signal')
#plt.axvline(x=5,ls=':',label='$t/M = 5$')
#plt.plot(times_LR[:],np.log(np.abs(Ref_LR[:])),label='signal')
#plt.plot(times_LR[160000:],fitR,'--',label='fit')

#plt.plot(times_LR[:70000],Ref_LR[:70000]/np.sqrt(Ref_fix[:70000]**2+Imf_fix[:70000]**2),label='$h_{\\times}$')
#plt.plot(times_LR[:70000],(Ref_LR[:70000]-Ref_fix[:70000])/np.sqrt(Ref_fix[:70000]**2+Imf_fix[:70000]**2),label='$\Delta h_{\\times}$')

#plt.plot(times_proj,np.log(np.abs(Ref_proj-Re22_LR)),label='2fit residual')
#plt.plot(times_LR,np.log(np.abs(Ref_LR-Re22_LRff)),label='residual free w')
#plt.plot(times_MR,(Ref_LR-Ref_MR)/np.sqrt(Ref_HR[0]**2+Imf_HR[0]**2),label='$Re(\Psi_4^{LR}-\Psi_4^{MR})$')
#plt.plot(times_MR,16*(Ref_MR-Ref_HR)/np.sqrt(Ref_HR[0]**2+Imf_HR[0]**2),'--',label='$16\\times Re(\Psi_4^{MR}-\Psi_4^{HR})$')
#plt.plot(times_XLR,(Ref_XLR - Ref_HR)/np.sqrt(Ref_HR**2+Imf_HR**2),label='$\Delta M 3/4$')
#plt.plot(times_MR,Ref_MR-Ref_LR,label='physical - 0.5$\Delta M$')
#plt.plot(times_LR,np.sqrt(Ref_LR**2+Imf_LR**2)/np.sqrt(Ref_XLR**2+Imf_XLR**2),label='Amp $\ell = 2$')
#plt.plot(times_LR,np.sqrt(Ref_MR**2+Imf_MR**2)/np.sqrt(Ref_XLR3**2+Imf_XLR3**2),label='Amp $\ell = 3$')
#plt.plot(times_LR,np.sqrt(Ref_HR**2+Imf_HR**2)/np.sqrt(Ref_XLR4**2+Imf_XLR4**2),label='Amp $\ell = 4$')
#plt.plot(times_LR,np.sqrt(Ref_XHR**2+Imf_XHR**2)/np.sqrt(Ref_XLR5**2+Imf_XLR5**2),label='Amp $\ell = 5$')
#plt.plot(times_LR,np.log(np.sqrt(Ref_HR**2+Imf_HR**2)),label='Amp $\ell = 3$')
#plt.plot(times_LR,np.log(np.sqrt(Ref_HR4**2+Imf_HR4**2)),label='Amp $\ell = 4$')
#plt.plot(times_LR,np.log(np.sqrt(Ref_HR5**2+Imf_HR5**2)),label='Amp $\ell = 5$')
plt.legend(loc="upper left")
plt.tight_layout()
plt.xlim(0,65)
#plt.ylim(-10**-3,10**-3)
#plt.ylim(2,19)
#plt.ylabel('$\\frac{\sqrt{\Delta h_+^2+\Delta h_{\\times}^2 }}{\sqrt{h_+^2+h_{\\times}^2}}$')
#plt.ylabel('$h_{\\times}/A(t)$')
#plt.ylabel('$\log_{10}|$fit$-\Psi_4|$')
plt.ylabel('$\Delta\phi\left(\Psi_4 \\times rM\\right)$')
#plt.ylabel('$\Delta Re(\Psi_4(t) \\times rM)/Re(\Psi_4(0) \\times rM)$')
plt.xlabel('$t / M$')
#plt.title('Change in signal $a=0.7$')
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


#pp = PdfPages('./plots/Mode_amps_xrange.pdf')
#plt.figure()
#plt.plot(f_22,label='(2,2)')
#plt.plot(f_23,label='(3,2)')
#plt.plot(f_24,label='(4,2)')
#plt.plot(times,Ref_33,label='(3,3)')
#plt.plot(times,Ref_34,label='(4,3)')
#plt.legend()
#plt.ylabel('A(l,m)')
#plt.xlabel('x')
#pp.savefig()
#pp.close()


plt.show()
