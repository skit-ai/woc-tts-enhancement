import numpy as np
import scipy.interpolate as itp

# Weighted spectral average
# Made using: https://github.com/vipchengrui/traditional-speech-enhancement/tree/master/spectral_subtraction_noise_estimation
class Est_Weight(object):
	
	def __init__(self,ns_ps,para):
		self.ns_ps = ns_ps
		self.para = para
	
	def est(self):
		
		# input para
		ass = self.para['ass']
		beta = self.para['beta']
		noise_ps = self.para['noise_ps']
		P = self.para['P']
		P = ass * P + (1 - ass) * self.ns_ps
		
		# noise estiamtion
		# [9.30] in the power-spectrum domain
		for i in range(len(noise_ps)):
			if P[i] < beta * noise_ps[i]:
				noise_ps[i] = ass * noise_ps[i] + (1 - ass) * P[i]

		# output para 
		self.para['P'] = P
		self.para['noise_ps'] = noise_ps
		return self.para

# Continuous minimal tracking
class Est_ConMinTrack(object):

	def __init__(self,ns_ps,para):
		self.ns_ps = ns_ps
		self.para = para

	def est(self):

		# input para
		n = self.para['n']
		leng = self.para['leng']
		alpha = self.para['alpha']
		beta = self.para['beta']
		gamma = self.para['gamma']
		noise_ps = self.para['noise_ps']
		pxk_old = self.para['pxk_old']
		pxk = self.para['pxk']
		pnk_old = self.para['pnk_old']
		pnk = self.para['pnk']

		# noise estimation
		# [9.24]
		pxk = alpha * pxk_old + (1 - alpha) * self.ns_ps
		# [9.25]
		for t in range(leng):
		    if pnk_old[t] <= pxk[t]:
		        pnk[t] = (gamma * pnk_old[t]) + (((1 - gamma) / (1 - beta)) * (pxk[t] - beta * pxk_old[t]))
		    else:
		        pnk[t] = pxk[t]
		pxk_old = pxk
		pnk_old = pnk
		noise_ps = pnk

		# output
		self.para['n'] = n + 1
		self.para['noise_ps'] = noise_ps
		self.para['pnk'] = pnk
		self.para['pnk_old'] = pnk_old
		self.para['pxk'] = pxk
		self.para['pxk_old'] = pxk_old
		return self.para

# MCRA algorithm
class Est_MCRA(object):

	def __init__(self,ns_ps,para):
		self.ns_ps = ns_ps
		self.para = para

	def est(self):

		# input para
		ass = self.para['ass']
		ad = self.para['ad']
		ap = self.para['ap']
		pk = self.para['pk']
		delta = self.para['delta']
		L = self.para['L']
		n = self.para['n']
		leng = self.para['leng']
		noise_ps = self.para['noise_ps']
		P = self.para['P']
		Pmin = self.para['Pmin']
		Ptmp = self.para['Ptmp']

		# noise estimation
		# [9.55]
		P = ass * P + (1 - ass) * self.ns_ps  	
		# [9.23]	
		if n % L == 0:
		    Pmin = np.minimum(Ptmp , P)  			
		    Ptmp = P            					
		else:
		    Pmin = np.minimum(Pmin , P) 			
		    Ptmp = np.minimum(Ptmp , P) 	
		# [9.58]		
		Srk = P / Pmin 
		Ikl = np.zeros(leng)
		for i in range(len(Ikl)):
			if Srk[i] > delta:
				Ikl[i] = 1
		# [9.59]
		pk = ap * pk + (1 - ap) * Ikl  
		# [9.54]						
		adk = ad + (1 - ad) * pk  	
		# [9.53]						
		noise_ps = adk * noise_ps + (1 - adk) * self.ns_ps  

		# output para
		self.para['pk'] = pk
		self.para['n'] = n + 1
		self.para['noise_ps'] = noise_ps
		self.para['P'] = P
		self.para['Pmin'] = Pmin
		self.para['Ptmp'] = Ptmp
		return self.para

# MCRA2 algorithm
class Est_MCRA2(object):

	def __init__(self,ns_ps,para):
		self.ns_ps = ns_ps
		self.para = para

	def est(self):

		# input para
		n = self.para['n']
		leng = self.para['leng']
		ad = self.para['ad']
		ass = self.para['ass']
		ap = self.para['ap']
		beta = self.para['beta']
		gamma = self.para['gamma']
		alpha = self.para['alpha']
		pk = self.para['pk']
		delta = self.para['delta']
		noise_ps = self.para['noise_ps']
		pxk = self.para['pxk']
		pnk = self.para['pnk']
		pxk_old = self.para['pxk_old']
		pnk_old = self.para['pnk_old']

		# noise estimation
		# [9.61]
		pxk = alpha * pxk_old + (1 - alpha) * self.ns_ps
		# [9.25]
		for i in range(len(pnk)):
			if pnk_old[i] < pxk[i]:
				pnk[i] = (gamma * pnk_old[i]) + (((1 - gamma) / (1 - beta)) * (pxk[i] - beta * pxk_old[i]))
		pxk_old = pxk
		pnk_old = pnk
		# [9.57]
		Srk = np.zeros(leng)
		Srk = pxk / pnk
		# [9.58]
		Ikl = np.zeros(leng)
		for i in range(len(Ikl)):
			if Srk[i] > delta[i]:
				Ikl[i] = 1
		# [9.59]
		pk = ap * pk + (1 - ap) * Ikl  		
		# [9.54]			
		adk = ad + (1 - ad) * pk  						
		# [9.53]
		noise_ps = adk * noise_ps + (1 - adk) * pxk 	
		
		# output para
		self.para['n'] = n + 1
		self.para['pk'] = pk
		self.para['noise_ps'] = noise_ps
		self.para['pnk'] = pnk
		self.para['pnk_old'] = pnk_old
		self.para['pxk'] = pxk
		self.para['pxk_old'] = pxk_old
		return self.para

class Init_Weight(object):

	def __init__(self,ns_ps,fs):
		self.ns_ps = ns_ps
		self.fs = fs
	
	def info(self):
		parameters = {'ass':0.85 , 'beta':1.5 , 'noise_ps':self.ns_ps , 'P':self.ns_ps}
		return parameters

# Continuous minimal tracking
class Init_ConMinTrack(object):
	
	def __init__(self,ns_ps,fs):
		self.ns_ps = ns_ps
		self.fs = fs
	
	def info(self):
		len_val = len(self.ns_ps)
		parameters = {'n':2,'leng':len_val,'alpha':0.7,'beta':0.96,'gamma':0.998,\
		'noise_ps':self.ns_ps,'pxk_old':self.ns_ps,'pxk':self.ns_ps,'pnk_old':self.ns_ps,'pnk':self.ns_ps}
		return parameters

class Init_MCRA(object):

	def __init__(self,ns_ps,fs):
		self.ns_ps = ns_ps
		self.fs = fs

	def info(self):
		len_val = len(self.ns_ps)
		parameters = {'n':2,'ad':0.95,'ass':0.8,'L':1000*2//20,'delta':5,'ap':0.2,'leng':len_val,\
		'P':self.ns_ps,'Pmin':self.ns_ps,'Ptmp':self.ns_ps,'pk':np.zeros(len_val),'noise_ps':self.ns_ps}
		return parameters

# MCRA2 algorithm
class Init_MCRA2(object):

	def __init__(self,ns_ps,fs):
		self.ns_ps = ns_ps
		self.fs = fs

	def info(self):
		len_val = len(self.ns_ps)
		freq_res = self.fs / len_val
		k_1khz = int(1000 // freq_res)
		k_3khz = int(3000 // freq_res)
		
		# [9.60] delta
		low_1 = 2*np.ones(k_1khz,dtype=np.int),
		low_2 = 2*np.ones(k_3khz-k_1khz,dtype=np.int),
		high = 5*np.ones(len_val//2-k_3khz,dtype=np.int),
		delta_val = np.append(np.append(np.append(np.append(np.append(low_1,low_2),high),high),low_2),low_1)

		parameters = {'n':2,'leng':len_val,'ad':0.95,'ass':0.8,'ap':0.2,'beta':0.8,'beta1':0.98,'gamma':0.998,'alpha':0.7,\
		'delta':delta_val,'pk':np.zeros(len_val),'noise_ps':self.ns_ps,'pxk_old':self.ns_ps,'pxk':self.ns_ps,'pnk_old':self.ns_ps,'pnk':self.ns_ps}

		return parameters