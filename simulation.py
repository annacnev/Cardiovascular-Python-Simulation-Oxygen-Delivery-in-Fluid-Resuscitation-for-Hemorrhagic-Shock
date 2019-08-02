
import numpy
from numpy import multiply, exp, dot, divide
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, subplot
import numpy as np
from math import ceil
import copy


def arange(start, stop, step=1, **kwargs):
	expand_value = 1 if step > 0 else -1
	return np.arange(start, stop + expand_value, step, **kwargs).reshape(1, -1)

def mat_max(a, d=None):
	if d is not None:
		try:
			m = np.amax(a)
		except TypeError:
			m = a
		if m < d:
			return d
		else:
			return m
	else:
		return max(a)

def concat(args):
	t = [np.asarray(a) for a in args]
	try:
		return np.concatenate(t)
	except ValueError:
		if isinstance(args[0], np.ndarray):
			first = args[0]
		else:
			first = np.asarray(args[0])
		del args[0]
		for a in args:
			if isinstance(a, np.ndarray):
				first = np.append (first, a)
			else:
				a = np.asarray(a)
				first = np.append(first, a)
		return first

class VPQ:
	def __init__(self, V, P, Q):
		self.V = V
		self.P = P
		self.Q = Q


class c_obj():
	def __init__(self, p, N):
		self.Emax = numpy.zeros((4, 1), dtype=np.float64, order='F')
		self.D = numpy.zeros((4, 3), dtype=np.float64, order='F')
		self.En = numpy.zeros((3, 1), dtype=np.float64, order='F')
		self.Rv = numpy.zeros((4, 1), dtype=np.float64, order='F')
		self.V0 = numpy.zeros((4, 1), dtype=np.float64, order='F')
		self.Em = numpy.zeros((3, 1), dtype=np.float64, order='F')
		self.Kv = numpy.zeros((4, 1), dtype=np.float64, order='F')
		self.delay = numpy.zeros((4, 1), dtype=np.float64, order='F')
		self.rn = numpy.zeros((3, 1), dtype=np.float64, order='F')
		self.bv = 5600
		self.HR = 80
		self.a0 = numpy.zeros((9, 1), dtype=np.float64, order='F')
		self.a0[0:9, 0] = [- 118.3, - 127.5, - 155.6, - 203.3, - 277.3, - 404.8, - 578, - 915.3, - 2211]
		self.a1 = numpy.zeros((9, 1), dtype=np.float64, order='F')
		self.a1[0:9, 0] = [44.27, 25.77, 25.48, 42.3, 73.76, 138.5, 480.8, 465.7, 1492]
		self.a2 = numpy.zeros((9, 1), dtype=np.float64, order='F')
		self.a2[0:9, 0] = [- 3.848, - 1.245, - 1.068, - 2.425, - 5.758, - 14.64, - 127.2, - 76.97, - 332.4]
		self.a3 = numpy.zeros((9, 1), dtype=np.float64, order='F')
		self.a3[0:9, 0] = [0.144, 0.025, 0.018, 0.057, 0.18, 0.602, 12.7, 4.64, 25.72]
		self.Rba = numpy.zeros((7, 1), dtype=np.float64, order='F')
		self.Rba[0:7, 0] = [15.57, 6.2, 4.14, 2.32, 3.0, 11.85, 2.91]
		self.Rbv = numpy.zeros((7, 1), dtype=np.float64, order='F')
		self.Rbv[0:7, 0] = [7.27, 2.91, 1.94, 1.12, 1.45, 5.82, 1.45]
		self.Eb = numpy.zeros((7, 1), dtype=np.float64, order='F')
		self.Eb[0:7, 0] = [1.75, 0.7, 0.467, 0.27, 0.35, 1.4, 0.35]
		self.V0b = numpy.zeros((7, 1), dtype=np.float64, order='F')
		self.V0b[0:7, 0] = [20, 50, 75, 130, 100, 25, 100]
		self.Rsv = numpy.zeros((1, 1), dtype=np.float64, order='F')
		self.Rsv[0] = 0.004
		self.Esv = numpy.zeros((1, 1), dtype=np.float64, order='F')
		self.Esv[0] = 0.0057
		self.V0sv = numpy.zeros((1, 1), dtype=np.float64, order='F')
		self.V0sv[0] = 1750
		self.Rp = numpy.zeros((3, 1), dtype=np.float64, order='F')
		self.Rp[0:3, 0] = [0.267, 0.11, 0.004]
		self.Ep = numpy.zeros((3, 1), dtype=np.float64, order='F')
		self.Ep[0:3, 0] = [0.2, 0.18, 0.08]
		self.EV0p = numpy.zeros((3, 1), dtype=np.float64, order='F')
		self.EV0p[0:3, 0] = [40, 70, 122]
		self.Rr = numpy.zeros((4, 1), dtype=np.float64, order='F')
		self.Rr[0:4, 0] = [0.0022, 0.044, 0, 0]
		self.Lr = numpy.zeros((4, 1), dtype=np.float64, order='F')
		self.Lr[0:4, 0] = [0.0004, 0, 0, 0]
		self.dt = (60 / (multiply(p.HR, self.HR))) / N
		self.tmax = round((400 - multiply(multiply(1.8, self.HR), p.HR)) / (multiply(1000, self.dt)))
		self.Emax[0] = 2.0
		self.V0[0] = 15.0
		self.Emax[1] = 1.0
		self.V0[1] = 25.0
		self.Emax[2] = 1.0
		self.V0[2] = 5.0
		self.Emax[3] = 0.6
		self.V0[3] = 5.0
		self.En[0] = 0.158
		self.En[1] = 2.685
		self.En[2] = - 1.841
		self.Em[0] = 0.158
		self.Em[1] = 2.685
		self.Em[2] = - 1.841
		self.rn[0] = - 1.934
		self.rn[1] = 6.568
		self.rn[2] = - 3.734
		self.Rv[0] = 0.0004
		self.Rv[1] = 0.0004
		self.Rv[2] = 0.004
		self.Rv[3] = 0.004
		self.Kv[0] = 0.0003
		self.Kv[1] = 0.0003
		self.Kv[2] = 0
		self.Kv[3] = 0
		self.D[0, 0] = 7
		self.D[0, 1] = 0.15
		self.D[0, 2] = 0.025
		self.D[1, 0] = 7.0
		self.D[1, 1] = 0.06
		self.D[1, 2] = 0.025
		self.D[2, 0] = 7.0
		self.D[2, 1] = 0.15
		self.D[2, 2] = 0.09
		self.D[3, 0] = 7.0
		self.D[3, 1] = 0.15
		self.D[3, 2] = 0.08
		self.delay[0] = round(0.16 / (self.dt))
		self.delay[1] = round(0.16 / (self.dt))
		self.delay[2] = 0
		self.delay[3] = -0


class n_obj:
	def __init__(self):
		self.la = 3
		self.lv = 1
		self.ra = 4
		self.rv = 2


class v_obj:
	def __init__(self, N):
		self.h = numpy.zeros((4, N), dtype=object, order='F')
		self.h[0, 0:N] = VPQ(120, 10, 0)
		self.h[1, 0:N] = VPQ(120, 8, 0)
		self.h[2, 0:N] = VPQ(20, 10, 0)
		self.h[3, 0:N] = VPQ(20, 8, 0)


class p_obj:
	def __init__(p):
		p.HR = 1


class Sim:
	def __init__(self):
		self.test = 1
		self.test2 = -14000
		self.N = 1000
		self.n = n_obj()
		self.v = v_obj(self.N)
		self.p = p_obj()
		self.c = c_obj(self.p, self.N)
		self.dt = self.c.dt
		self.Qvein = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Vvein = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Vvein[0:self.N] = 3500
		self.Pvein = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Pvein[0:self.N] = multiply(self.c.Esv, (self.Vvein[0:self.N] - self.c.V0sv))
		self.Qseg = numpy.zeros((9, self.N), dtype=np.float64, order='F')
		self.Qbnch_inp = numpy.zeros((7, self.N), dtype=np.float64, order='F')
		self.Qbnch_out = numpy.zeros((7, self.N), dtype=np.float64, order='F')

		self.Vseg = numpy.zeros((9, self.N), dtype=np.float64, order='F')
		Vseg_values = [11.91, 20.379, 23.571, 17.254, 12.629, 9.243, 3.653, 5.789, 4.237]
		for i in range(0, len(Vseg_values)):
			self.Vseg[i, 0:self.N] = Vseg_values[i]

		self.Vbnch = numpy.zeros((7, self.N), dtype=np.float64, order='F')
		Vbnch_values = [40, 100, 150, 260, 200, 50, 200]
		for i in range(0, len(Vbnch_values)):
			self.Vbnch[i, 0:self.N] = Vseg_values[i]

		self.Vpulm = numpy.zeros((3, self.N), dtype=np.float64, order='F')
		self.Ppulm = numpy.zeros((3, self.N), dtype=np.float64, order='F')
		Vpulm_Ppulm = [(120, 32), (142, 12), (200, 8)]
		for i in range(0, len(Vpulm_Ppulm)):
			self.Vpulm[i, 0:self.N] = Vpulm_Ppulm[i][0]
			self.Ppulm[i, 0:self.N] = Vpulm_Ppulm[i][1]

		self.Qpulm = numpy.zeros((3, self.N), dtype=np.float64, order='F')
		for i in range(0, 2):
			self.Qpulm[i, 0:self.N] = mat_max((self.Ppulm[i, 0:self.N] - self.Ppulm[i + 1, 0:self.N]) / self.c.Rp[i], 0)

		self.Pseg = numpy.zeros((9, self.N), dtype=np.float64, order='F')
		self.Pbnch = numpy.zeros((7, self.N), dtype=np.float64, order='F')
		for i in arange(0, self.N - 1).reshape(-1):
			for j in arange(0, 8).reshape(-1):
				self.Pseg[j, i] = multiply(
					(multiply((multiply(self.c.a3[j], self.Vseg[j, i]) + self.c.a2[j]), self.Vseg[j, i]) + self.c.a1[j]),
					self.Vseg[j, i]) + self.c.a0[j]
			for j in arange(0, 6).reshape(-1):
				self.Pbnch[j, i] = multiply(self.c.Eb[j], (self.Vbnch[j, i] - self.c.V0b[j]))
			self.Qpulm[2, i] = mat_max((self.Ppulm[2, i] - self.v.h[2, i].P) / self.c.Rp[2], 0)

		self.x = plt.cm.hsv(np.linspace(0, 1, 1024))
		self.N_seg = 9
		self.N_pulm = 3
		self.N_branch = 7
		self.viscos_fac = 0.31855
		self.N_cycles = 5000
		self.monit_cycles = 100
		self.Pt = 0
		self.tissue_layers = 10
		self.c_elem = 50
		self.t_elem = self.tissue_layers * self.c_elem
		self.Y = numpy.zeros((self.t_elem, self.t_elem), dtype=np.float64, order='F')
		self.Y_Cdt = numpy.zeros((self.t_elem, self.t_elem), dtype=np.float64, order='F')
		self.unity_mx = np.ones((self.N_branch, 1), dtype=np.float64, order='F')
		self.seg_Len = numpy.zeros((9, 1), dtype=np.float64, order='F').flatten()
		self.seg_Len[0:9] = [2, 4, 6, 6, 6, 6, 3, 6, 6]
		self.seg_Len = multiply(1, self.seg_Len)
		self.seg_R0 = numpy.zeros((9, 1), dtype=np.float64, order='F').flatten()
		self.seg_R0[0:9] = [0.024, 0.012, 0.006, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024]
		self.seg_L0 = numpy.zeros((9, 1), dtype=np.float64, order='F').flatten()
		self.seg_L0[0:9] = [8e-05, 8e-05, 8e-05, 0.00016, 0.00016, 0.00016, 0.00016, 0.00016, 0.00016]
		self.seg_L0 = multiply(0.8, self.seg_L0)
		ra = multiply(multiply((multiply(multiply(np.pi ** 2, self.seg_R0.flatten()), (self.seg_Len.flatten()) ** 3)) / (self.Vseg[:, 0].T ** 2),self.viscos_fac), np.exp(multiply(2.6, 0.44)))
		self.Ra = numpy.zeros((self.N, 9), dtype=np.float64, order='F')
		self.La = numpy.zeros((self.N, 9), dtype=np.float64, order='F')
		self.Ra[0, 0:len(ra)] = ra
		la = (multiply(multiply(np.pi, (self.seg_Len.flatten()) ** 2), self.seg_L0.flatten())) / self.Vseg[:, 0].T
		self.La[0, 0:len(la)] = la
		self.var1 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var2  = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var3 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.Qb_tm_avg = numpy.zeros((1, self.monit_cycles), dtype=np.float64, order='F').flatten()
		self.Qb_tm_avg[0:self.monit_cycles] = 7.7e-09
		self.var18 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var19 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var30 = numpy.zeros((self.N_seg, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F')
		self.var40 = numpy.zeros((self.N_branch, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F')
		self.var8 = numpy.zeros((self.N_branch, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F')
		self.var5 = numpy.zeros((self.N_branch, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F')
		self.Pbnch_inp = numpy.zeros((self.N_branch, self.N), dtype=np.float64, order='F')
		self.Ppulm_out = numpy.zeros((self.N_pulm, self.N), dtype=np.float64, order='F')
		self.Qpulm_inp = numpy.zeros((self.N_pulm, self.N), dtype=np.float64, order='F')
		self.Qseg_bnch = numpy.zeros((self.N_seg, self.N), dtype=np.float64, order='F')
		self.Vtot_inp = 6456
		Q_old = self.La[0, 1:self.N_seg].T / (self.La[0, 1:self.N_seg].T + multiply(self.Ra[0, 1:self.N_seg].T, self.dt))
		self.Q_old_fac = numpy.zeros((self.N_seg, self.N), dtype=np.float64, order='F')
		self.Q_old_fac[1:self.N_seg, 0] = Q_old
		self.P_factor = numpy.zeros((self.N_seg, self.N), dtype=np.float64, order='F')
		self.P_factor[1:self.N_seg, 0] = (self.La[0, 1:self.N_seg].T + multiply(self.Ra[0, 1:self.N_seg].T, self.dt))
		self.Res_Prot = 0
		self.Cpl_res = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Qpl_inf = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Vtot = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Vtot[0:self.N] = self.Vtot_inp
		self.Vrbc_norm = multiply(0.44, 5600)
		self.Hct_norm = self.Vrbc_norm / self.Vtot_inp
		self.Vpl_norm = multiply((1 - self.Hct_norm), self.Vtot[0])
		self.Vpl = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Vpl[0:self.N] = self.Vpl_norm
		self.Hct = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Hct[0:self.N] = self.Hct_norm
		self.Vrbc = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Vrbc[0:self.N] = self.Vrbc_norm
		self.Vext_norm = 15136
		self.V_Inorm = self.Vext_norm - self.Vpl_norm
		self.V_I = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.V_I[0:self.N] = self.V_Inorm
		self.Ja_I = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Jv_I = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.J_L = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Cpl = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Cpl[0:self.N] = 7.3
		self.C_I = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.C_I[0:self.N] = 2.0
		self.Pprot_pl = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Pprot_pl[0:self.N] = multiply(0.2274, (self.Cpl[0:self.N] ** 2)) + multiply(2.1755, self.Cpl[0:self.N])
		self.Pprot_I = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Pprot_I[0:self.N] = multiply(0.2274, (self.C_I[0:self.N] ** 2)) + multiply(2.1755, self.C_I[0:self.N])
		self.P_I = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.P_I[0:self.N] = multiply(0.0025, self.V_I[0:self.N]) - 37
		self.R_pl_I = 50
		self.PO2_avg = numpy.zeros((10 * self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.PO2_cons_avg = numpy.zeros((10 * self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.PO2_cell_sum = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.PO2_cell_cons_sum = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.Hct_in = 0.44
		self.P_EC = numpy.zeros((10 * self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.P_EC_Hct = numpy.zeros((10, 1), dtype=np.float64, order='F')
		self.C_Hct = numpy.zeros((self.c_elem, 1), dtype=np.float64, order='F').flatten()
		self.cap_length = 0.03
		self.r_capillary = 0.000325
		self.r_tissue = 0.00325
		self.r = numpy.zeros((self.tissue_layers, self.c_elem), dtype=np.float64, order='F')
		self.PO2_t_solubility = 3e-05
		self.PO2_t_D = 1.5e-05
		self.PO2_c_D = self.PO2_t_D
		self.Qb = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Qb[0:self.N] = 7.7e-09
		self.r_centr = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.t_seg_vol = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.t_seg_cap = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.y_dn = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.y_up = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.y_R = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.R_L_R = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.R_d_u = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.Hct_st = 0.44
		self.PO2_consmp = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.PO2 = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F').flatten()
		self.PO2[0:self.t_elem] = 40
		self.tissue_volume = multiply(multiply(np.pi, (self.r_tissue ** 2 - self.r_capillary ** 2)), self.cap_length)
		self.axial_res = self.cap_length / self.c_elem
		self.radial_res = (self.r_tissue - self.r_capillary) / self.tissue_layers
		self.r_pos = np.asarray(arange(self.r_capillary, self.r_tissue, self.radial_res)).flatten()
		self.c_seg_vol = multiply(multiply(np.pi, self.r_capillary ** 2), self.axial_res)
		self.g_cap_tot = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.g_cap_tot[0:self.monit_cycles] = multiply((1.21 - multiply(4.3, self.Hct_st) + multiply(23.6, (self.Hct_st) ** 2)), 1e-06)
		self.g_cap = multiply(multiply(multiply(multiply(self.g_cap_tot[0], 2), np.pi), self.r_capillary), self.axial_res)
		self.R_up_capil = numpy.zeros((self.c_elem, 1), dtype=np.float64, order='F')
		self.R_up_capil[0:self.c_elem] = 1 / (self.g_cap)

		for j in arange(1, self.tissue_layers).reshape(-1):
			for i in arange(1 + multiply(self.c_elem, (j - 1)), multiply(j, self.c_elem)).reshape(-1):
				self.r_centr[i - 1] = (self.r_pos[j - 1] + self.r_pos[j]) / 2

		for j in arange(1, self.tissue_layers).reshape(-1):
			for i in arange(1 + multiply(self.c_elem, (j - 1)), multiply(j, self.c_elem)).reshape(-1):
				self.t_seg_vol[i - 1] = multiply(multiply(np.pi, (self.r_pos[j] ** 2 - self.r_pos[j - 1] ** 2)), self.axial_res)

		self.t_vol_factor = multiply(multiply(self.PO2_t_D, self.PO2_t_solubility), self.t_seg_vol)

		for j in arange(1, self.tissue_layers).reshape(-1):
			for i in arange(1 + multiply(self.c_elem, (j - 1)), multiply(j, self.c_elem)).reshape(-1):
				self.y_R[i - 1] = multiply(self.t_vol_factor[i - 1], (1 / (self.axial_res / 2) ** 2))

		self.R_R = 1.0 / self.y_R
		self.R_L = self.R_R
		for j in arange(1, self.tissue_layers).reshape(-1):
			for i in arange(1 + multiply(self.c_elem, (j - 1)), multiply(j, self.c_elem)).reshape(-1):
				self.y_dn[i - 1] = multiply(self.t_vol_factor[i - 1], (1 / (self.radial_res / 2) ** 2))
				self.y_up[i - 1] = multiply(self.t_vol_factor[i - 1], ((1 / (self.radial_res / 2) ** 2) + (1 / (multiply(self.r_centr[i - 1], (self.radial_res / 2))))))
		self.R_dn = 1.0 / self.y_dn
		self.R_up = 1.0 / self.y_up

		self.t_cap = numpy.zeros((self.t_elem, 1), dtype=np.float64, order='F')

		for j in arange(1, self.tissue_layers).reshape(-1):
			for i in arange(1 + multiply(self.c_elem, (j - 1)), multiply(j, self.c_elem)).reshape(-1):
				self.t_cap[i - 1] = multiply(self.PO2_t_solubility, self.t_seg_vol[i - 1])
		for i in arange(1, self.c_elem).reshape(-1):
			self.R_d_u[i - 1] = self.R_up_capil[i - 1]

		for j in arange(1, self.tissue_layers - 1).reshape(-1):
			for i in arange(1 + multiply(self.c_elem, (j - 1)), multiply(j, self.c_elem)).reshape(-1):
				self.R_d_u[i + self.c_elem - 1] = self.R_dn[i + self.c_elem - 1] + self.R_up[i - 1]

		self.g_d_u = 1.0 / (multiply(self.R_d_u, 1))
		for j in arange(1, self.tissue_layers).reshape(-1):
			for i in arange(1 + multiply(self.c_elem, (j - 1)), multiply(j, self.c_elem) - 1).reshape(-1):
				self.R_L_R[i - 1] = self.R_R[i - 1] + self.R_L[i + 1 - 1]

		for i in arange(self.c_elem, multiply(self.tissue_layers, self.c_elem), self.c_elem).reshape(-1):
			self.R_L_R[i - 1] = self.R_R[i - 1] + self.R_L[i - self.c_elem + 1 - 1]

		self.g_L_R = 1.0 / (multiply(self.R_L_R, 1))
		self.M_total = multiply(1, 8e-10)
		self.Km = 0.5
		self.Mmax = self.M_total / (multiply(multiply(self.cap_length, np.pi), (self.r_tissue ** 2 - self.r_capillary ** 2)))
		self.M_elem = multiply(self.Mmax, self.t_seg_vol)
		self.Zero_vec = numpy.zeros((1, self.t_elem - self.c_elem), dtype=np.float128, order='F')
		self.P_capillary = numpy.zeros((1, self.c_elem), dtype=np.float128, order='F')
		self.cap_press = np.append(self.P_capillary[0:self.c_elem], self.Zero_vec).flatten()
		self.Admittance_Matrix()
		self.Cin = 0.4889
		self.C = numpy.zeros((int(self.c_elem), int(multiply(10, self.monit_cycles) + 1)), dtype=np.float128, order='F')
		self.C[0:self.c_elem, 0:int(multiply(10, self.monit_cycles) + 1)] = 0.4889
		self.cycle_one = 1
		self.mres = 1
		self.col_fac = 100
		self.Hct_O2 = 0.5
		self.R_bleed = 24
		self.fac_rep = 2
		self.frame_disp = 10
		self.Vtot_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.V_I_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Vpl_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Hct_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Hct_avg[0:self.monit_cycles] = 0.44
		self.SV = numpy.zeros((1, self.N), dtype=np.float64, order='F').flatten()
		self.Ox_deliv = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.CO = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Bleed_vol_avg = numpy.zeros((1, self.monit_cycles), dtype=np.float64, order='F').flatten()
		self.Res_vol_avg = numpy.zeros((1, self.monit_cycles), dtype=np.float64, order='F').flatten()
		self.VlV = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.PlV = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.QlV = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.VrV = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.PrV = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.QrV = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.C_avg = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.C_avg[0:self.N] = 0.22
		self.PO2_cell_sum = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Bleed_vol = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Res_vol = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Pbnch_tot = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.colid_infus_vol = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Pbnch_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Pvein_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Ja_I_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Jv_I_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Net_filt = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Qbnch_bl_avg = numpy.zeros((1, self.monit_cycles), dtype=np.float64, order='F').flatten()
		self.Hct_prev = 0.44
		self.Qb_tm_avg[0:self.monit_cycles] = 7.7e-09
		self.Ox_deliv_bleed = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F')
		self.J_bleed_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F')
		self.J_res_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F')
		self.Qpl_inf_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F')
		self.EDV = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F')
		self.Pbnch_tot_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F')
		self.C_tm_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F')
		self.PO2_tm_cell_avg = numpy.zeros((self.monit_cycles, self.t_elem), dtype=np.float64, order='F')
		self.PO2_tm_space_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F')
		self.PO2_tm_capdir_avg = numpy.zeros((self.monit_cycles, self.tissue_layers), dtype=np.float64, order='F')
		self.P_EC_avg = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F')
		self.PO2_capdir_avg = numpy.zeros((self.N, self.tissue_layers), dtype=np.float64, order='F')
		self.Hct_avg_mod = numpy.zeros((self.monit_cycles, 1), dtype=np.float64, order='F').flatten()
		self.Hct_avg_mod[0:self.monit_cycles] = 0.44
		self.C_avg = numpy.zeros((self.N, 1), dtype=np.float64, order='F')
		self.C_avg[0:self.N] = 0.22
		self.flg = 0
		self.cycle_m = 1
		self.start_bleed = 15000
		self.end_bleed = 16500
		self.start_res = 17500
		self.end_res = 23500
		self.start_save = 24000
		self.stable_cycle = 250
		self.sec_cycle = 60 / 80
		self.cycle_tic = self.dt
		self.J_bleed = numpy.zeros((multiply(self.monit_cycles, self.N), 1), dtype=np.float64, order='F')
		self.Rbx = self.c.Rba
		self.Rby = self.c.Rbv
		self.Rbz = self.c.Rp
		self.Pt = 1
		self.refx = 0
		self.ref1 = 0
		self.start_plot = self.start_bleed - multiply(10, self.monit_cycles)
		self.cycle_global = ceil((self.start_save) / self.monit_cycles) + 1
		self.PO2_avg_monit = numpy.zeros((1, int(self.cycle_global)), dtype=np.float64, order='F').flatten()
		self.Hct_avg_monit = numpy.zeros((1, int(self.cycle_global)), dtype=np.float64, order='F').flatten()
		self.Qb_avg_monit = numpy.zeros((1, int(self.cycle_global)), dtype=np.float64, order='F').flatten()
		self.refpx = self.start_plot
		self.x_lim = numpy.zeros((3, 1), dtype=np.float64, order='F').flatten()
		self.y_lim = numpy.zeros((3, 1), dtype=np.float64, order='F').flatten()
		self.x_back = numpy.zeros((1, 1), dtype=np.float64, order='F').flatten()
		self.y_back = numpy.zeros((1, 1), dtype=np.float64, order='F').flatten()
		self.var12 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var13 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var14 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var15 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var16 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var17 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var60 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var61 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var62 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var63 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var6 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var7 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.var4 = numpy.zeros((1, multiply(multiply(3, self.monit_cycles), self.N)), dtype=np.float64, order='F').flatten()
		self.J_res = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.Qpl_bld = numpy.zeros((self.N, 1), dtype=np.float64, order='F').flatten()
		self.MAP = numpy.zeros((self.N + 1, 1), dtype=np.float64, order='F').flatten()
		self.colid_infus_vol_avg = numpy.zeros((self.N + 1, 1), dtype=np.float64, order='F').flatten()
		self.Px = numpy.zeros((self.tissue_layers, self.c_elem), dtype=np.float64, order='F')
		self.Py = numpy.zeros((self.tissue_layers, self.c_elem), dtype=np.float64, order='F')


	def Admittance_Matrix(self):

		self.Y[0, 0] = self.g_d_u[0] + self.g_d_u[self.c_elem] + self.g_L_R[0] + self.g_L_R[self.c_elem - 1]
		self.Y[0, 1] = - (self.g_L_R[0])
		self.Y[0, self.c_elem] = - (self.g_d_u[self.c_elem])
		self.Y[0, self.c_elem - 1] = - (self.g_L_R[self.c_elem - 1])

		self.Y[self.c_elem - 1, self.c_elem - 1] = self.g_d_u[self.c_elem - 1] + self.g_d_u[multiply(2, self.c_elem) - 1] + self.g_L_R[self.c_elem - 2] + self.g_L_R[self.c_elem - 1]
		self.Y[self.c_elem - 1, self.c_elem - 2] = - (self.g_L_R[self.c_elem - 2])
		self.Y[self.c_elem - 1, multiply(2, self.c_elem) - 1] = - (self.g_d_u[multiply(2, self.c_elem) - 1])
		self.Y[self.c_elem - 1, 0] = - (self.g_L_R[self.c_elem - 1])
		self.Y[self.t_elem - self.c_elem, self.t_elem - self.c_elem] = self.g_L_R[self.t_elem - self.c_elem] + self.g_d_u[self.t_elem - multiply(2, self.c_elem)] + self.g_L_R[self.t_elem - 1]

		self.Y[self.t_elem - self.c_elem, self.t_elem - multiply(2, self.c_elem)] = - (self.g_d_u[self.t_elem - multiply(2, self.c_elem)])
		self.Y[self.t_elem - self.c_elem, self.t_elem - self.c_elem + 1] = - (self.g_L_R[self.t_elem - self.c_elem])
		self.Y[self.t_elem - self.c_elem, self.t_elem - 1] = - (self.g_L_R[self.t_elem - 1])
		self.Y[self.t_elem - 1, self.t_elem - 1] = (self.g_d_u[self.t_elem - 1] + self.g_L_R[self.t_elem - 2] + self.g_L_R[self.t_elem - 1])

		self.Y[self.t_elem - 1, self.t_elem - 2] = - (self.g_L_R[self.t_elem - 2])
		self.Y[self.t_elem - 1, self.t_elem - self.c_elem - 1] = - (self.g_L_R[self.t_elem - 1])
		self.Y[self.t_elem - 1, self.t_elem - self.c_elem] = - (self.g_d_u[self.t_elem - 1])

		for i in arange(2, self.c_elem - 1, 1).reshape(-1):
			self.Y[i - 1, i - 1] = self.g_d_u[i - 1] + self.g_d_u[self.c_elem + i - 1] + self.g_L_R[i - 2] + self.g_L_R[i - 1]
			self.Y[i - 1, i - 2] = - (self.g_L_R[i - 2])
			self.Y[i - 1, i] = - (self.g_L_R[i - 1])
			self.Y[i - 1, i + self.c_elem - 1] = - (self.g_d_u[self.c_elem + i - 1])

		for i in arange((self.t_elem - self.c_elem + 2), self.t_elem - 1, 1).reshape(-1):
			self.Y[i - 1, i - 1] = (self.g_d_u[i - 1] + self.g_L_R[i - 2] + self.g_L_R[i - 1])
			self.Y[i - 1, i - 2] = - (self.g_L_R[i - 2])
			self.Y[i - 1, i] = - (self.g_L_R[i - 1])
			self.Y[i - 1, i - self.c_elem - 1] = - (self.g_d_u[i - 1])

		for i in arange(self.c_elem + 1, self.t_elem - multiply(2, self.c_elem) + 1, self.c_elem).reshape(-1):
			self.Y[i - 1, i - 1] = self.g_d_u[i - 1] + self.g_d_u[self.c_elem + i - 1] + self.g_L_R[i - 1] + self.g_L_R[i + self.c_elem - 2]
			self.Y[i - 1, i] = - (self.g_L_R[i - 1])
			self.Y[i - 1, i + self.c_elem - 1] = - (self.g_d_u[self.c_elem + i - 1])
			self.Y[i - 1, i - self.c_elem - 1] = - (self.g_d_u[i - 1])
			self.Y[i - 1, i + self.c_elem - 2] = - (self.g_L_R[i + self.c_elem - 2])

		for i in arange(multiply(2, self.c_elem), self.t_elem - self.c_elem, self.c_elem).reshape(-1):
			self.Y[i - 1, i - 1] = self.g_L_R[i - 1] + self.g_d_u[self.c_elem + i - 1] + self.g_L_R[i - 2] + self.g_L_R[i - 1]
			self.Y[i - 1, i - 2] = - (self.g_L_R[i - 2])
			self.Y[i - 1, i + self.c_elem - 1] = - (self.g_d_u[self.c_elem + i - 1])
			self.Y[i - 1, i - self.c_elem] = - (self.g_L_R[i - 1])
			self.Y[i - 1, i - self.c_elem - 1] = - (self.g_L_R[i - 1])

		for j in arange(1, self.tissue_layers - 2, 1).reshape(-1):
			for i in arange(1 + (1 + multiply(j, self.c_elem)), multiply((j + 1), self.c_elem) - 1, 1).reshape(-1):
				self.Y[i - 1, i - 1] = self.g_d_u[i - 1] + self.g_d_u[self.c_elem + i - 1] + self.g_L_R[i - 2] + self.g_L_R[i - 1]
				self.Y[i - 1, i - 2] = - (self.g_L_R[i - 2])
				self.Y[i - 1, i] = - (self.g_L_R[i - 1])
				self.Y[i - 1, i - self.c_elem - 1] = - (self.g_d_u[i - 1])
				self.Y[i - 1, i + self.c_elem - 1] = - (self.g_d_u[self.c_elem + i - 1])

		for j in arange(1, self.t_elem).reshape(-1):
			for i in arange(1, self.t_elem).reshape(-1):
				self.Y_Cdt[i - 1, j - 1] = multiply((multiply((1 / self.t_cap[i - 1]), self.dt)), self.Y[i - 1, j - 1])
		return self

	def monitor_variables1(self, t):
		self.var1[self.Pt - 1] = self.v.h[0, t - 1].P
		self.var2[self.Pt - 1] = self.v.h[0, t - 1].V
		self.var18[self.Pt - 1] = self.v.h[0, t - 1].Q
		self.var3[self.Pt - 1] = self.Pseg[0, t - 1]
		self.var30[:, self.Pt - 1] = self.Pseg[:, t - 1]
		self.var4[self.Pt - 1] = self.v.h[1, t - 1].P
		self.var5[:, self.Pt - 1] = self.Vbnch[:, t - 1]
		self.var6[self.Pt - 1] = self.Ppulm[0, t - 1]
		self.var7[self.Pt - 1] = self.v.h[1, t - 1].V
		self.var19[self.Pt - 1] = self.v.h[1, t - 1].Q
		self.var8[:, self.Pt - 1] = self.Qbnch_out[:, t - 1]
		self.var40[:, self.Pt - 1] = self.Pbnch[:, t - 1]
		self.var12[self.Pt - 1] = self.Vtot[t - 1]
		self.var13[self.Pt - 1] = self.Hct[t - 1]
		self.var14[self.Pt - 1] = self.Vpl[t - 1]
		self.var15[self.Pt - 1] = self.V_I[t - 1]
		self.var16[self.Pt - 1] = self.Ja_I[t - 1]
		self.var17[self.Pt - 1] = self.Jv_I[t - 1]
		self.var60[self.Pt - 1] = self.PO2_avg[t - 1]
		self.var61[self.Pt - 1] = self.PO2_cons_avg[t - 1]
		self.var62[self.Pt - 1] = self.P_EC[t - 1]
		self.var63[self.Pt - 1] = self.Qb[t - 1]

		if (t == self.N):
			self.var1[self.Pt] = self.var1[self.Pt - 1]
			self.var2[self.Pt] = self.var2[self.Pt - 1]
			self.var3[self.Pt] = self.var3[self.Pt - 1]
			self.var4[self.Pt] = self.var4[self.Pt - 1]
			self.var5[:, self.Pt] = self.var5[:, self.Pt - 1]
			self.var6[self.Pt] = self.var6[self.Pt - 1]
			self.var7[self.Pt] = self.var7[self.Pt - 1]
			self.var8[:, self.Pt] = self.var8[:, self.Pt - 1]
			self.var12[self.Pt] = self.var12[self.Pt - 1]
			self.var13[self.Pt] = self.var13[self.Pt - 1]
			self.var14[self.Pt] = self.var14[self.Pt - 1]
			self.var15[self.Pt] = self.var15[self.Pt - 1]
			self.var16[self.Pt] = self.var16[self.Pt - 1]
			self.var17[self.Pt] = self.var17[self.Pt - 1]
			self.var30[:, self.Pt] = self.var30[:, self.Pt - 1]
			self.var40[:, self.Pt] = self.var40[:, self.Pt - 1]
			self.var18[self.Pt] = self.var18[self.Pt - 1]
			self.var19[self.Pt] = self.var19[self.Pt - 1]
			self.var60[self.Pt] = self.var60[self.Pt - 1]
			self.var61[self.Pt] = self.var61[self.Pt - 1]
			self.var62[self.Pt] = self.var62[self.Pt - 1]
			self.var63[self.Pt] = self.var63[self.Pt - 1]
			self.Pt = self.Pt + 1
		return self

	def cvsim_heart2(self, t, j, h, Qi, Po):
		self.p.Emax = 1
		tn = (t - self.c.delay[j - 1]) / self.c.tmax

		if tn < 0:
			Enor = 0
		else:
			Enor = mat_max(multiply((multiply((multiply(self.c.En[2], tn) + self.c.En[1]), tn) + self.c.En[0]), tn), 0)
		V = h.V + multiply((Qi - h.Q), self.dt)

		P0 = multiply(multiply(multiply(self.p.Emax, self.c.Emax[j - 1]), Enor), (V - self.c.V0[j - 1]))
		Pdia = (multiply(self.c.D[j - 1, 0], np.log(multiply(self.c.D[j - 1, 2], V))) + multiply(self.c.D[j - 1, 1],np.exp(multiply(self.c.D[j - 1, 2], V))))

		if Pdia > P0:
			P0 = Pdia
		else:
			pass
		R = multiply(self.c.Kv[j - 1], P0) + self.c.Rv[j - 1] + self.c.Rr[j - 1]
		Q = mat_max((P0 - Po) / R, 0)
		Q = mat_max(Q, 0)
		P = multiply(P0, (1 - multiply(Q, (self.c.Kv[j - 1]))))

		return (V, P, Q)

	def mesh(self, input, figure_num, X_label, Y_label, Z_label, title):
		# mesh(Z) draws a wireframe mesh using X = 1:n and Y = 1:m, where[m, n] = size(Z)
		fig = plt.figure(figure_num)
		ax = fig.add_subplot(111, projection='3d')
		#for input in inputs:
		shape_ = input.shape
		x = list(range(0, shape_[0]))
		y = list(range(0, shape_[1]))
		X, Y = np.meshgrid(x, y)
		Z = input.reshape(X.shape)
		#ax.plot_surface(X, Y, Z)
		ax.plot_wireframe(X, Y, Z)
		ax.set_xlabel(X_label)
		ax.set_ylabel(Y_label)
		ax.set_zlabel(Z_label)
		plt.title(title)
		return self

	def cvsim_heart(self, t, j, h, Qi, Po):
		self.p.Emax = 1
		h = h
		tn = (t - self.c.delay[j - 1]) / self.c.tmax

		if tn < 0:
			Enor = 0
		else:
			Enor = mat_max(multiply((multiply((multiply(self.c.En[2], tn) + self.c.En[1]), tn) + self.c.En[0]), tn), 0)
		V = h.V + multiply((Qi - h.Q), self.dt)
		P0 = multiply(multiply(multiply(self.p.Emax, self.c.Emax[j - 1]), Enor), (V - self.c.V0[j - 1]))
		Pdia = (multiply(multiply(0.25*(10 ** - 7.5), self.c.D[0, 1]), exp(multiply(multiply(self.c.D[0, 2], 6.1), V)))) + (multiply(7, np.log(multiply(0.025, V))))

		if Pdia > P0:
			P0 = Pdia
		else:
			pass
		R = multiply(self.c.Kv[j - 1], P0) + self.c.Rv[j - 1] + self.c.Rr[j - 1]
		Q = multiply((self.c.Lr[j - 1] / (self.c.Lr[j - 1] + multiply(R, self.dt))), h.Q) + (multiply((P0 - Po), self.dt)) / (
				self.c.Lr[j - 1] + multiply(R, self.dt))
		Q = mat_max(Q, 0)
		P = multiply(P0, (1 - multiply(Q, (self.c.Kv[j - 1]))))
		return (V, P, Q)

	def plot_press_Hct(self):
		self.PO2_f = np.zeros((self.tissue_layers, self.c_elem), dtype=np.float64, order='F')
		for i in arange(1, self.tissue_layers).reshape(-1):
			self.PO2_f[i - 1, :] = self.PO2[arange(multiply((i - 1), self.c_elem), multiply(i, self.c_elem) - 1)]

		self.PO2_c = np.zeros((self.tissue_layers, self.c_elem), dtype=np.float64, order='F')
		for i in arange(1, self.tissue_layers).reshape(-1):
			self.PO2_c[i - 1, :] = self.PO2_consmp[arange(multiply((i - 1), self.c_elem), multiply(i, self.c_elem) - 1)]


		self.DD = np.zeros((self.tissue_layers, self.c_elem), dtype=np.float64, order='F')
		for j in arange(1, self.c_elem, 1).reshape(-1):
			for i in arange(1, self.tissue_layers, 1).reshape(-1):
				self.DD[i - 1, j - 1] = self.PO2[multiply((i - 1) - 1, self.c_elem) + j - 1]


		self.DD_c = np.zeros((self.tissue_layers, self.c_elem), dtype=np.float64, order='F')
		for j in arange(1, self.c_elem, 1).reshape(-1):
			for i in arange(1, self.tissue_layers, 1).reshape(-1):
				self.DD_c[i - 1, j - 1] = self.PO2_consmp[multiply((i - 1) - 1, self.c_elem) + j - 1]

		for i in range(1, self.tissue_layers):
			plot(self.PO2_f[i - 1, :], color=self.x[20 * (i - 1), :])
		plt.title('Oxygen pressure in tissue direction')
		plt.xlabel('capillary_direction')
		plt.ylabel('pressure')
		# leg('layer1','L2','L3','L4','L5','L6','L7','L8','L9','L10')

		figure(48)
		for i in range(1, self.tissue_layers):
			plot(self.PO2_c[i - 1, :], color=self.x[20 * (i - 1), :])
		plt.title('Oxygen consumption in tissue direction')
		plt.xlabel('capillary_direction')
		plt.ylabel('pressure')

		figure(49)
		for j in range(1, 10):
			plot(self.DD[0:self.tissue_layers, (j - 1) * 5], color=self.x[20 * (j - 1), :])
		plt.title('Oxygen pressure in capillary direction')
		plt.xlabel('tissue_direction')
		plt.ylabel('pressure')

		figure(50)
		for j in range(1, 10):
			plot(self.DD_c[0:self.tissue_layers, (j - 1) * 5], color=self.x[20 * j - 1, :])
		plt.title('Oxygen consumption in tissue direction')
		plt.xlabel('tissue_direction')
		plt.ylabel('pressure')

		figure(58)
		plot(self.Hct_avg[self.cycle_m - 1] * self.C[0:self.c_elem, self.cycle_m - 1])
		plt.title('Oxygen concentration in capillary elements')

		figure(54)
		plot(self.cap_press[0:self.c_elem])
		plt.title('Oxygen pressure in capillary elements')

		for i in arange(1, self.tissue_layers).reshape(-1):
			for j in arange(1, self.c_elem).reshape(-1):
				self.Px[i - 1, j - 1] = self.PO2[multiply((i - 1) - 1, self.c_elem) + j - 1]

		self.mesh(self.Px, 51, 'axial direction cell', 'radial direction cell', 'PO2', 'Krog cylynder cells PO2 distribution-Last iteration)')

		for i in arange(1, self.tissue_layers).reshape(-1):
			for j in arange(1, self.c_elem).reshape(-1):
				self.Py[i - 1, j - 1] = self.PO2_consmp[multiply((i - 1) - 1, self.c_elem) + j - 1]
		self.mesh(self.Py, 52, 'axial direction cell', 'radial direction cell', 'consumption', 'Krog cylynder cells consumption-Last iteration)')
		figure(53)
		plot(self.Qb)
		plt.title('Qblood')
		return self


	def compute_with_consumption(self, t):

		# -------------------------  consumption_hct  --------------------------------------------

		self.Qb[t - 1] = (self.Qb_tm_avg[self.cycle_m - 1])/1200000000.0
		self.Hct_avg_mod[self.cycle_m - 1] = self.Hct_avg[self.cycle_m - 1]
		part_one = np.asarray([self.Km + x for x in  self.PO2[0:self.t_elem]])
		part_two = np.divide(self.PO2[0:self.t_elem], part_one)
		conju = 1*self.M_elem[0:self.t_elem].conj().T
		part_three = multiply(conju, part_two)
		compare = numpy.append(0, part_three)
		self.PO2_consmp[0:self.t_elem] = np.amax(compare)
		self.c_conct_b = self.Qb[t - 1]*self.dt / self.c_seg_vol
		self.c_conct_f = 1 / (1 + self.c_conct_b)
		self.c_conct_v = self.c_conct_b*self.c_conct_f
		self.c_conct_pres = dot(dot(dot((2 / dot(self.r_capillary, self.Hct_avg_mod[self.cycle_m-1])), self.dt), self.g_cap_tot[self.cycle_m-1]), self.c_conct_f)
		self.C[0, t - 1] = self.Cin

		y = multiply(self.Hct_avg_mod[self.cycle_m - 1], self.C[0:self.c_elem, t - 1])
		for x in range(0, len(y)):
			z = (y[x]-0.16438)/0.070943
			self.cap_press[x] = 7.0624 * z**9 + 52.852 * z**8 + 142.78 * z**7 + 149.52 * z**6 + 2.6376 * z**5 - 81.241 * z**4 - 7.5042 * z**3 + 36.224 * z**2 + 27.711 * z + 37.752
		difference = np.subtract(self.cap_press[1:self.c_elem], self.PO2[1:self.c_elem])
		self.C[1:self.c_elem, t] = multiply(self.c_conct_f, self.C[1:self.c_elem, t-1]) + self.c_conct_v * self.C[0:self.c_elem-1, t] - multiply(self.c_conct_pres, difference)


		sum_array = np.sum(self.C[0:self.c_elem, t-1])
		self.C_avg[t-1] = dot(self.Hct_avg[self.cycle_m-1], sum_array) / self.c_elem

		# -------------------------  Cap_g_interface  --------------------------------------------
		#break up into parts to cut down on run time so computer is not figuring out order of operations
		part_one = dot(4.3, self.Hct_avg[self.cycle_m-1])
		part_two = self.Hct_avg[self.cycle_m-1]** 2
		part_three = dot(23.6, part_two)
		part_four = 1.21 - part_one + part_three
		self.g_cap_tot[self.cycle_m-1] = dot(part_four, 1e-6)

		part_one = dot(self.g_cap_tot[self.cycle_m-1], 2)
		part_two = dot(part_one, np.pi)
		part_three = dot(part_two, self.r_capillary)
		g_cap = dot(part_three, self.axial_res)

		self.R_up_capil[0:self.c_elem] = 1 / (g_cap)

		for i in arange(1, self.c_elem).reshape(-1):
		 	self.R_d_u[i - 1] = self.R_up_capil[i - 1]

		index1 = multiply(2, self.c_elem) - 1
		self.g_d_u[0:self.c_elem] = divide(1.0, (self.R_d_u[0:self.c_elem]))
		self.Y[0, 0] = self.g_d_u[0] + self.g_d_u[self.c_elem] + self.g_L_R[0] + self.g_L_R[self.c_elem - 1]
		self.Y[0, 1] = - (self.g_L_R[0])
		self.Y[0, self.c_elem] = - (self.g_d_u[self.c_elem])
		self.Y[0, self.c_elem - 1] = - (self.g_L_R[self.c_elem - 1])
		self.Y[self.c_elem - 1, self.c_elem - 1] = self.g_d_u[self.c_elem - 1] + self.g_d_u[index1] + self.g_L_R[self.c_elem - 2] + self.g_L_R[self.c_elem - 1]
		self.Y[self.c_elem - 1, self.c_elem - 2] = - (self.g_L_R[self.c_elem - 2])
		self.Y[self.c_elem - 1, multiply(2, self.c_elem) - 1] = - (self.g_d_u[index1])
		self.Y[self.c_elem - 1, 0] = - (self.g_L_R[self.c_elem - 1])

		for i in arange(2, self.c_elem - 1, 1).reshape(-1):
			self.Y[i - 1, i - 1] = self.g_d_u[i - 1] + self.g_d_u[self.c_elem + i - 1] + self.g_L_R[i - 2] + self.g_L_R[i - 1]
			self.Y[i - 1, i - 2] = - (self.g_L_R[i - 2])
			self.Y[i - 1, i] = - (self.g_L_R[i - 1])
			self.Y[i - 1, i + self.c_elem - 1] = - (self.g_d_u[self.c_elem + i - 1])


		self.Y_Cdt[0:self.c_elem, 0:self.c_elem] = multiply((multiply((1 / self.t_cap[0:self.c_elem]), self.dt)), self.Y[0:self.c_elem, 0:self.c_elem])

		# -------------------------  Cap_g_interface  --------------------------------------------

		inv_t_cap = np.divide(1, self.t_cap.conj().T)
		inv_t_cap_dt = multiply(inv_t_cap, self.dt)
		gdu_conj = self.g_d_u[0].conj().T
		multip = multiply(gdu_conj, self.cap_press)
		self.PO2 = (self.PO2 - dot(self.Y_Cdt, self.PO2) - multiply(self.PO2_consmp, inv_t_cap_dt) + multiply(multip, inv_t_cap_dt)).flatten()
		# -------------------------  consumption_hct  --------------------------------------------


		self.PO2_avg[t - 1] = np.sum(self.PO2[0:self.t_elem] / self.t_elem)
		self.PO2_cons_avg[t - 1] = np.sum(self.PO2_consmp[0:self.t_elem] / self.t_elem)
		self.P_EC[t - 1] = self.cap_press[self.c_elem - 1]
		self.PO2_cell_sum[t - 1] = np.sum(self.PO2[0:self.t_elem]) / self.t_elem

		for i in arange(1, self.tissue_layers).reshape(-1):
			self.PO2_capdir_avg[t - 1, i - 1] = (np.sum(self.PO2[(i - 1)*self.c_elem:i*self.c_elem]) / self.c_elem)
		self.P_EC[t - 1] = self.cap_press[self.c_elem - 1]
		return self

	def cyclic_condition(self):
		self.v.h[:, 0] = self.v.h[:, self.N - 1]
		self.Vseg[:, 0] = self.Vseg[:, self.N - 1]
		self.Pseg[:, 0] = self.Pseg[:, self.N - 1]
		self.Qseg[:, 0] = self.Qseg[:, self.N - 1]
		self.Vbnch[:, 0] = self.Vbnch[:, self.N - 1]
		self.Pbnch[:, 0] = self.Pbnch[:, self.N - 1]
		self.Qbnch_inp[:, 0] = self.Qbnch_inp[:, self.N - 1]
		self.Qbnch_out[:, 0] = self.Qbnch_out[:, self.N - 1]
		self.Vvein[0] = self.Vvein[self.N - 1]
		self.Pvein[0] = self.Pvein[self.N - 1]
		self.Qvein[0] = self.Qvein[self.N - 1]
		self.Vpulm[:, 0] = self.Vpulm[:, self.N - 1]
		self.Ppulm[:, 0] = self.Ppulm[:, self.N - 1]
		self.Qpulm[:, 0] = self.Qpulm[:, self.N - 1]
		self.Vtot[0] = self.Vtot[self.N - 1]
		self.Hct[0] = self.Hct[self.N - 1]
		self.Ja_I[0] = self.Ja_I[self.N - 1]
		self.Jv_I[0] = self.Jv_I[self.N - 1]
		self.Cpl[0] = self.Cpl[self.N - 1]
		self.C_I[0] = self.C_I[self.N - 1]
		self.J_bleed[0] = self.J_bleed[self.N - 1]
		self.Vpl[0] = self.Vpl[self.N - 1]
		self.V_I[0] = self.V_I[self.N - 1]
		self.Vrbc[0] = self.Vrbc[self.N - 1]
		self.P_I[0] = self.P_I[self.N - 1]
		self.Pprot_pl[0] = self.Pprot_pl[self.N - 1]
		self.Pprot_I[0] = self.Pprot_I[self.N - 1]
		self.J_res[0] = self.J_res[self.N - 1]
		return self

	def cvsim_heart3(self, t, j, h, Qi, Po):
		self.p.Emax = 1
		h = h
		tn = (t - self.c.delay[j - 1]) / self.c.tmax

		if tn < 0:
			Enor = 0
		else:
			Enor = mat_max(multiply((multiply((multiply(self.c.En[2], tn) + self.c.En[1]), tn) + self.c.En[0]), tn), 0)
		V = h.V + multiply((Qi - h.Q), self.dt)
		P0 = multiply(multiply(multiply(self.p.Emax, self.c.Emax[j - 1]), Enor), (V - self.c.V0[j - 1]))
		Pdia = (multiply(self.c.D[j - 1, 0], np.log(multiply(self.c.D[j - 1, 2], V))) + multiply(self.c.D[j - 1, 1], np.exp(multiply(self.c.D[j - 1, 2], V))))

		if Pdia > P0:
			P0 = Pdia
		else:
			pass
		R = multiply(self.c.Kv[j - 1], P0) + self.c.Rv[j - 1] + self.c.Rr[j - 1]
		Q = mat_max((P0 - Po) / R, 0)
		Q = mat_max(Q, 0)
		P = multiply(P0, (1 - multiply(Q, (self.c.Kv[j - 1]))))
		return (V, P, Q)

	def heart(self, t):
		self.c.Rr[0] = self.Ra.flatten()[0]
		self.c.Lr[0] = self.La.flatten()[0]
		self.v.h[2, t - 1] = VPQ(*self.cvsim_heart2(t, 3, self.v.h[2, t - 2], self.Qpulm[2, t - 2], self.v.h[0, t - 2].P))
		self.v.h[0, t - 1] = VPQ(*self.cvsim_heart(t, 1, self.v.h[0, t - 2], self.v.h[2, t - 2].Q, self.Pseg[0, t - 2]))
		self.v.h[3, t - 1] = VPQ(*self.cvsim_heart2(t, 4, self.v.h[3, t - 2], self.Qvein[t - 2], self.v.h[1, t - 2].P))
		self.v.h[1, t - 1] = VPQ(*self.cvsim_heart3(t, 2, self.v.h[1, t - 2], self.v.h[3, t - 2].Q, self.Ppulm[0, t - 2]))
		return self

	def run_simulation(self):

		# self.prev = datetime.datetime.now()
		for cycle in arange(1, self.start_save).reshape(-1):

			# --------------------------------------------------------
			print('cycle = ' + str(cycle))

			for t in arange(2, self.N).reshape(-1):
				# --------------------------------------------------------
				self.heart(t)
				self.VlV[t - 1] = self.v.h[0, t - 1].V
				self.PlV[t - 1] = self.v.h[0, t - 1].P
				self.QlV[t - 1] = self.v.h[0, t - 1].Q
				self.VrV[t - 1] = self.v.h[1, t - 1].V
				self.PrV[t - 1] = self.v.h[1, t - 1].P
				self.QrV[t - 1] = self.v.h[1, t - 1].Q

				# ------------------------------------------------------------------
				# for kk in arange(1, self.fac_rep).reshape(-1):
				self.Qseg_bnch[0:self.N_seg, t - 2] = concat([self.Qbnch_inp[0, t - 2], self.Qbnch_inp[1, t - 2] + self.Qbnch_inp[2, t - 2], 0, 0, 0,self.Qbnch_inp[3, t - 2], self.Qbnch_inp[4, t - 2], self.Qbnch_inp[5, t - 2],self.Qbnch_inp[6, t - 2]]).T
				self.Vseg[0, t - 1] = self.Vseg[0, t - 2] + multiply((self.v.h[0, t - 1].Q - (self.Qseg[1, t - 2] + self.Qbnch_inp[0, t - 2])),self.dt) / self.fac_rep
				self.Vseg[1:self.N_seg - 1, t - 1] = self.Vseg[1:self.N_seg - 1, t - 2] + multiply((self.Qseg[1:self.N_seg - 1, t - 2] - (self.Qseg[2:self.N_seg,t - 2] + self.Qseg_bnch[1:self.N_seg - 1,t - 2])),self.dt) / self.fac_rep
				self.Vseg[self.N_seg - 1, t - 1] = self.Vseg[self.N_seg - 1, t - 2] + multiply((self.Qseg[self.N_seg - 1, t - 2] - (self.Qbnch_inp[6, t - 2])), self.dt) / self.fac_rep
				self.Pseg[:, t - 1] = multiply(self.c.a3.T, (self.Vseg[:, t - 1] ** 3)) + multiply(self.c.a2.T, (self.Vseg[:, t - 1] ** 2)) + multiply(self.c.a1.T, (self.Vseg[:, t - 1])) + self.c.a0.T
				self.Ra[t - 1, :] = multiply(multiply(((multiply(multiply(np.pi ** 2, self.seg_R0), (self.seg_Len) ** 3)).flatten() / (self.Vseg[:, t - 1].T ** 2)),self.viscos_fac), exp(2.6 * self.Hct[t - 1]))
				self.La[t - 1, :] = multiply(multiply(np.pi, np.power(self.seg_Len.flatten(),2)), self.seg_L0.flatten()) / (self.Vseg[:, t - 1].T)
				self.Q_old_fac[1:self.N_seg, t - 1] = self.La[t - 1, 1:self.N_seg].T / (self.La[t - 1, 1:self.N_seg].T + multiply(self.Ra[t - 1, 1:self.N_seg].T,self.dt) / self.fac_rep)
				self.P_factor[1:self.N_seg, t - 1] = (self.La[t - 1, 1:self.N_seg].T + multiply(self.Ra[t - 1, 1:self.N_seg].T,self.dt) / self.fac_rep)
				self.Qseg[0, t - 1] = (self.v.h[0, t - 2].Q - (self.Qseg[1, t - 2] + self.Qbnch_inp[0, t - 2]))
				self.Qseg[1:self.N_seg, t - 1] = multiply(self.Q_old_fac[1:self.N_seg, t - 1],self.Qseg[1:self.N_seg, t - 2]) + (multiply((self.Pseg[0:self.N_seg - 1, t - 1] - self.Pseg[1:self.N_seg, t - 1]),self.dt) / self.fac_rep) / self.P_factor[1:self.N_seg, t - 1]

				# --------------------------------------------------------------------
				self.Vbnch[:, t - 1] = self.Vbnch[:, t - 2] + multiply((self.Qbnch_inp[:, t - 2] - self.Qbnch_out[:, t - 2]), self.dt)
				self.Qpulm_inp[:, t - 2] = concat([self.v.h[1, t - 2].Q, self.Qpulm[0, t - 2], self.Qpulm[1, t - 2]]).T
				self.Vpulm[:, t - 1] = self.Vpulm[:, t - 2] + multiply((self.Qpulm_inp[:, t - 2] - self.Qpulm[:, t - 2]),self.dt)
				self.Vvein[t - 1] = self.Vtot[t - 1] - (self.v.h[0, t - 1].V + self.v.h[1, t - 1].V + self.v.h[2, t - 1].V + self.v.h[3, t - 1].V + dot(np.ones((1, self.N_seg), order='F', dtype=np.float64).flatten(), self.Vseg[:, t - 1].flatten()) + dot(np.ones((1, self.N_pulm), dtype=np.float64, order='F').flatten(), self.Vpulm[:, t - 1].flatten()) + dot(np.ones((1, self.N_branch), dtype=np.float64, order='F').flatten(), self.Vbnch[:, t - 1].flatten()))
				self.Pvein[t - 1] = multiply(self.c.Esv.T, (self.Vvein[t - 1] - self.c.V0sv.T))
				self.Pbnch[:, t - 1] = multiply(self.c.Eb.T, (self.Vbnch[:, t - 1] - self.c.V0b.T))
				self.Ppulm[:, t - 1] = multiply(self.c.Ep.T, (self.Vpulm[:, t - 1] - self.c.EV0p.T))
				self.Pbnch_tot[t - 1] = ((dot(np.ones((1, self.N_branch), order='F', dtype=np.float64).flatten(), self.Pbnch[:, t - 1].flatten())) / self.N_branch)
				self.Qvein[t - 1] = mat_max((self.Pvein[t - 1] - self.v.h[3, t - 2].P) / self.c.Rsv, 0)
				self.Pbnch_inp[:, t - 1] = concat([self.Pseg[0, t - 1], self.Pseg[1, t - 1], self.Pseg[1, t - 1], self.Pseg[5, t - 1],self.Pseg[6, t - 1],self.Pseg[7, t - 1], self.Pseg[8, t - 1]]).T
				self.c.Rba = multiply(multiply(self.Rbx, self.viscos_fac), np.exp(multiply(2.6, self.Hct[t - 1])))
				self.Qbnch_inp[0:self.N_branch, t - 1] = ((self.Pbnch_inp[0:self.N_branch, t - 1] -self.Pbnch[0:self.N_branch, t - 1]) / self.c.Rba.T)
				self.c.Rbv = multiply(multiply(self.Rby, self.viscos_fac), exp(multiply(2.6, self.Hct[t - 1])))
				self.Qbnch_out[:, t - 1] = ((self.Pbnch[:, t - 1] - self.Pvein[t - 1]) / self.c.Rbv.T)
				self.Ppulm_out[:, t - 2] = concat([self.Ppulm[1, t - 2], self.Ppulm[2, t - 2], self.v.h[2, t - 2].P]).T
				self.c.Rp = multiply(multiply(self.Rbz, self.viscos_fac), exp(multiply(2.6, self.Hct[t - 1])))
				self.Qpulm[0:self.N_pulm, t - 1] = (self.Ppulm[0:self.N_pulm, t - 1] - self.Ppulm_out[0:self.N_pulm, t - 2]) / self.c.Rp.T

				# ---------------------------------------------------------------------
				for pulm_index in arange(1, self.N_pulm).reshape(-1):
					if multiply((self.Qpulm_inp[pulm_index - 1, t - 1] - self.Qpulm[pulm_index - 1, t - 2]), self.dt) > self.Vpulm[pulm_index - 1, t - 1]:
						self.Qpulm[pulm_index - 1, t - 1] = self.Qpulm_inp[pulm_index - 1, t - 1] - (self.Vpulm[pulm_index - 1, t - 1] / self.dt)

				# ---------------------------------------------------------------------
				if (cycle < self.start_bleed or cycle > self.end_bleed):
					self.J_bleed[t - 1] = 0
				else:
					self.J_bleed[t - 1] = (self.Pbnch_tot[t - 1]) / self.R_bleed

				# ---------------------------------------------------------------------
				if ((cycle < self.start_res or cycle > self.end_res) or (self.flg == 1)):
					self.J_res[t - 1] = 0
				else:
					self.J_res[t - 1] = (35 + multiply(25, (self.mres - 1))) / 60

				# ---------------------------------------------------------------------
				if (cycle > self.stable_cycle):
					self.Ja_I[t - 1] = ((self.Pbnch_tot[t - 2] - self.P_I[t - 2]) - (self.Pprot_pl[t - 2] - self.Pprot_I[t - 2])) / self.R_pl_I
					self.Jv_I[t - 1] = ((self.P_I[t - 2] - self.Pvein[t - 2]) + (self.Pprot_pl[t - 2] - self.Pprot_I[t - 2])) / self.R_pl_I
					self.Vpl[t - 1] = self.Vpl[t - 2] - multiply(self.Ja_I[t - 1], self.dt) + multiply(self.Jv_I[t - 1], self.dt) - multiply(multiply(self.J_bleed[t - 2], self.dt), (1 - self.Hct[t - 2])) + multiply(self.J_res[t - 2], self.dt)
					self.Vrbc[t - 1] = self.Vrbc[t - 2] - multiply(multiply(self.Hct[t - 2], self.J_bleed[t - 2]), self.dt)
					self.V_I[t - 1] = self.V_I[t - 2] + multiply(self.Ja_I[t - 1], self.dt) - multiply(self.Jv_I[t - 1], self.dt)
					if self.V_I[t - 1] < 14800 or self.V_I[t - 1] == 14800:
						self.P_I[t - 1] = multiply(0.0025, self.V_I[t - 1]) - 37
					else:
						self.P_I[t - 1] = multiply(0.0001, self.V_I[t - 1]) - 1.48
					self.Vtot[t - 1] = self.Vpl[t - 1] + self.Vrbc[t - 1]
					self.Hct[t - 1] = self.Vrbc[t - 1] / self.Vtot[t - 1]
					self.Qpl_inf[t - 1] = self.Qpl_inf[t - 2] + multiply(multiply(self.J_res[t - 1], self.Res_Prot), self.dt)
					self.Qpl_bld[t - 1] = multiply((multiply(7.3, self.Hct[t - 1]) / self.Hct_norm), (self.Vpl[t - 1] / 100))
					self.Cpl[t - 1] = (self.Qpl_inf[t - 1] + self.Qpl_bld[t - 1]) / (self.Vpl[t - 1] / 100)
					self.C_I[t - 1] = multiply(2.0, self.V_Inorm) / self.V_I[t - 1]
					self.Pprot_pl[t - 1] = multiply(0.2274, self.Cpl[t - 1] ** 2) + multiply(2.1755, self.Cpl[t - 1])
					self.Pprot_I[t - 1] = multiply(0.2274, self.C_I[t - 1] ** 2) + multiply(2.1755, self.C_I[t - 1])
				if cycle > self.start_plot:
					self.compute_with_consumption(t)

				if cycle > self.start_plot:
					self.Pt = self.Pt + 1
					self.monitor_variables1(t)

			# ---------------------------------------------------------------------

			if cycle > self.start_plot:
				self.mnpH = np.max(self.Pseg[0, :].flatten())
				self.mnpL = np.min(self.Pseg[0, :].flatten())
				self.MAP[self.cycle_m - 1] = self.mnpL + (1 / 3)*(self.mnpH - self.mnpL)
				self.CO_min = np.min(self.VlV[0:self.N])
				self.CO_max = np.max(self.VlV[0:self.N])
				self.EDV[self.cycle_m - 1] = self.CO_max
				self.SV[self.cycle_m - 1] = self.CO_max - self.CO_min
				self.Vtot_avg[self.cycle_m - 1] = np.sum(self.Vtot[0:self.N]) / self.N
				self.V_I_avg[self.cycle_m - 1] = np.sum(self.V_I[0:self.N]) / self.N
				self.Vpl_avg[self.cycle_m - 1] = np.sum(self.Vpl[0:self.N]) / self.N
				self.CO[self.cycle_m - 1] = dot(self.SV[self.cycle_m - 1], 80) / 60
				self.Hct_avg[self.cycle_m - 1] = np.sum(self.Hct[0:self.N]) / self.N
				self.Ox_deliv[self.cycle_m - 1] = multiply(multiply(self.Hct_avg[self.cycle_m - 1], self.CO[self.cycle_m - 1]),self.Hct_O2)
				self.Bleed_vol_avg[self.cycle_m - 1] = np.sum(self.Bleed_vol[0:self.N]) / self.N
				self.Res_vol_avg[self.cycle_m - 1] = np.sum(self.Res_vol[0:self.N]) / self.N
				self.colid_infus_vol_avg[self.cycle_m - 1] = np.sum(self.colid_infus_vol[0:self.N]) / self.N
				self.Pbnch_avg[self.cycle_m - 1] = (np.max(self.Pbnch[1, 0:self.N]) + np.min(self.Pbnch[1, 0:self.N])) / 2
				self.Pvein_avg[self.cycle_m - 1] = (np.max(self.Pvein[0:self.N]) + np.min(self.Pvein[0:self.N])) / 2
				self.Ja_I_avg[self.cycle_m - 1] = np.sum(self.Ja_I[0:self.N]) / self.N
				self.Jv_I_avg[self.cycle_m - 1] = np.sum(self.Jv_I[0:self.N]) / self.N
				self.Net_filt[self.cycle_m - 1] = self.Ja_I_avg[self.cycle_m - 1] - self.Jv_I_avg[self.cycle_m - 1]
				self.Qbnch_bl_avg[self.cycle_m - 1] = (np.sum(self.Qbnch_out[1, 0:self.N]) / self.N)*(80 / 60)
				self.Ox_deliv_bleed[self.cycle_m - 1] = multiply(multiply(self.Hct_avg[self.cycle_m - 1], self.Qbnch_bl_avg[self.cycle_m - 1]), self.Hct_O2)
				self.J_bleed_avg[self.cycle_m - 1] = np.sum(self.J_bleed[0: self.N]) / self.N
				self.J_res_avg[self.cycle_m - 1] = np.sum(self.J_res[0:self.N]) / self.N
				self.Pbnch_tot_avg[self.cycle_m - 1] = np.sum(self.Pbnch_tot[0:self.N]) / self.N
				self.Qpl_inf_avg[self.cycle_m - 1] = np.sum(self.Qpl_inf[0:self.N]) / self.N
				self.Qb_tm_avg[self.cycle_m - 1] = np.sum(self.Qbnch_out[1, 0:self.N]) / self.N
				self.C_tm_avg[self.cycle_m - 1] = np.sum(self.C_avg[0:self.N]) / self.N
				self.PO2_tm_space_avg[self.cycle_m - 1] = np.sum(self.PO2_cell_sum[0:self.N]) / self.N
				self.PO2_tm_capdir_avg[self.cycle_m - 1, :] = np.sum(self.PO2_capdir_avg[0:self.N, :]) / self.N
				self.P_EC_avg[self.cycle_m - 1] = np.sum(self.P_EC[0:self.N]) / self.N
				self.cycle_m = self.cycle_m + 1

			# --------------------------------------------------------
			self.cyclic_condition()
			self.VlV[0] = self.VlV[self.N - 1]
			self.PlV[0] = self.PlV[self.N - 1]
			self.QlV[0] = self.QlV[self.N - 1]
			self.PrV[0] = self.PrV[self.N - 1]
			self.VrV[0] = self.VrV[self.N - 1]
			self.QrV[0] = self.QrV[self.N - 1]
			self.PO2_avg[0] = self.PO2_avg[self.N - 1]
			self.PO2_cons_avg[0] = self.PO2_cons_avg[self.N - 1]
			self.P_EC[0] = self.P_EC[self.N - 1]
			self.PO2_cell_sum[0:self.t_elem] = 0
			self.Bleed_vol[0] = self.Bleed_vol[self.N - 1]
			self.Res_vol[0] = self.Res_vol[self.N - 1]
			self.colid_infus_vol[0] = self.colid_infus_vol[self.N - 1]
			self.Pbnch_tot[0] = self.Pbnch_tot[self.N - 1]
			self.Qb[self.N - 1] = self.Qb[self.N - 2]

			if float(cycle) == float(self.start_bleed - 1):
				# --------------------------------------------------------
				yrange_plot = self.Pt - self.frame_disp * self.N - 1
				num_steps = len(self.var1[yrange_plot:self.Pt])
				xrange_plot = np.linspace(0, self.frame_disp * self.N * self.cycle_tic, num_steps)


				figure(8)
				subplot(211)
				plot(xrange_plot, self.var1[yrange_plot:self.Pt], color=self.x[self.col_fac - 1, :])
				plot(xrange_plot, self.var3[yrange_plot:self.Pt], color=self.x[self.col_fac * 2 - 1, :])
				plt.title('Left Ventricle andAortic presssure')
				plt.xlabel('Frame relative time (seconds')
				plt.ylabel('Pressure(mmHg)')
				subplot(212)
				plot(self.VlV[0:self.N], self.PlV[0:self.N])

				figure(80)
				subplot(211)
				plot(xrange_plot, self.var60[yrange_plot:self.Pt], color=self.x[self.col_fac - 1, :])
				plt.title('PO2_avg')
				plt.xlabel('Frame relative time (seconds')
				plt.ylabel('O2_Pressure(mmHg)')
				subplot(212)
				plot(xrange_plot, self.var61[yrange_plot:self.Pt], color=self.x[self.col_fac * 2 - 1, :])
				plt.title('PO2_cons_avg')
				plt.xlabel('Frame relative time (seconds')
				plt.ylabel('O2_Pressure(mmHg)')

				figure(30)
				subplot(211)
				plot(xrange_plot, self.var4[yrange_plot:self.Pt], color=self.x[self.col_fac - 1, :])
				plot(xrange_plot, self.var6[yrange_plot:self.Pt], color=self.x[self.col_fac * 2 - 1, :])
				plt.title('Right Ventricle and pulmunary pressure ')
				plt.xlabel('Frame relative time (seconds')
				plt.ylabel('Pressure(mmHg)')
				subplot(212)
				plot(self.VrV[0:self.N], self.PrV[0:self.N])

				figure(31)
				subplot(211)
				plot(xrange_plot, self.var2[yrange_plot:self.Pt], color=self.x[self.col_fac - 1, :])
				plt.title('Left Ventricle volume  ')
				plt.xlabel('Frame relative time (seconds')
				plt.ylabel('volume(ml)')
				subplot(212)
				plot(xrange_plot, self.var18[yrange_plot:self.Pt], color=self.x[self.col_fac - 1, :])
				plt.title('Left Ventricle fow')
				plt.xlabel('Frame relative time (seconds')
				plt.ylabel('Pressure(ml/sec)')

				figure(32)
				subplot(211)
				plot(xrange_plot, self.var7[yrange_plot:self.Pt], color=self.x[self.col_fac - 1, :])
				plt.title('Right Ventricle volume  ')
				plt.xlabel('Frame relative time (seconds')
				plt.ylabel('volume(ml)')
				subplot(212)
				plot(xrange_plot, self.var19[yrange_plot:self.Pt], color=self.x[self.col_fac - 1, :])
				plt.title('Right Ventricle fow')
				plt.xlabel('Frame relative time (seconds')
				plt.ylabel('Pressure(ml/sec)')

				figure(33)
				for j in range(1, self.N_seg):
					plot(xrange_plot, self.var30[j - 1, yrange_plot:self.Pt], color=self.x[self.col_fac * j - 1, :])
				plt.title('segments pressure')

				figure(34)
				for j in range(1, self.N_branch):
					plot(xrange_plot, self.var40[j - 1, yrange_plot:self.Pt], color=self.x[self.col_fac * j - 1, :])
				plt.title('branches pressure')

				figure(35)
				for j in range(1, self.N_branch):
					subplot(211)
					plot(xrange_plot, self.var8[j - 1, yrange_plot:self.Pt], color=self.x[self.col_fac * j - 1, :])
					subplot(212)
					plot(xrange_plot, self.var5[j - 1, yrange_plot:self.Pt], color=self.x[self.col_fac * j - 1, :])

				self.plot_press_Hct()

			if float(cycle - self.refpx) == float(self.monit_cycles):
				num_steps = len(self.MAP[0:self.cycle_m - 1])
				xrange_plot = np.linspace(self.refx * self.sec_cycle, (self.refx + self.cycle_m - 2) * self.sec_cycle, num_steps)
				xrange_plot2 = np.linspace((self.refx+1)*self.sec_cycle, (self.refx+self.cycle_m-1)*self.sec_cycle, num_steps)
				figure(7)
				plot(xrange_plot, self.MAP[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('MAP(mmHg)')
				plt.title('Mean arterial pressure')

				figure(12)
				subplot(211)
				plot(xrange_plot, self.Pbnch_tot_avg[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('average branch pressure(mmHg)')
				plt.title(' average branches pressure')

				subplot(212)
				plot(xrange_plot, self.Pvein_avg[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('vein pressure(mmHg)')
				plt.title('vein pressure')

				figure(150)
				plot(xrange_plot2, self.PO2_tm_space_avg[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('Qb_tm_space_avg(ml/sec)')
				plt.title('PO2_tm_space_avg')

				figure(151)
				plot(xrange_plot2, (self.Qb_tm_avg[0:self.cycle_m - 1]) / 1.2e9, color=self.x[self.col_fac - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('Qb-tm-avg(cycle)')
				plt.title('Qb-tm-avg')

				figure(152)
				plot(xrange_plot2, (self.P_EC_avg[0:self.cycle_m - 1]), color=self.x[self.col_fac - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('P-EC-avg(cycle)')
				plt.title('P-EC-avg')

				figure(155)
				for j in range(1, self.tissue_layers):
					plot(xrange_plot2, self.PO2_tm_capdir_avg[0:self.cycle_m - 1, j - 1], color=self.x[self.col_fac * j - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('PO2-tm-capdir-avg(cycle)')
				plt.title('PO2-tm-capdir-avg')

				figure(156)
				plot(xrange_plot2, (self.C_tm_avg[0:self.cycle_m - 1]), color=self.x[self.col_fac - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('C-tm-avg(cycle)')
				plt.title('C-tm-avg')

				self.PO2_avg_monit[self.cycle_one - 1] = self.PO2_tm_space_avg[self.cycle_m - 50 - 1]
				self.Hct_avg_monit[self.cycle_one - 1] = self.Hct_avg[self.cycle_m - 50 - 1]
				self.Qb_avg_monit[self.cycle_one - 1] = self.Qb_tm_avg[self.cycle_m - 50 - 1]

				figure(6)
				plot(xrange_plot, self.J_bleed_avg[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('bleeding rate(ml/sec)')
				plt.title('Bleeding rate')

				figure(2)
				plot(xrange_plot, self.Ox_deliv[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('time (seconds)')
				plt.ylabel('O2(mlO2/sec)')
				plt.title('Oxygen delivery')

				figure(10)
				plot(xrange_plot, self.Hct_avg[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('time (seconds)')
				plt.ylabel('HCT')
				plt.title('Hematocrite Blood diluition')

				figure(4)
				subplot(311)
				plot(xrange_plot, self.Vtot_avg[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.title('total Volume ')
				plt.xlabel('time (seconds)')
				plt.ylabel(' volume ml')

				subplot(312)
				plot(xrange_plot, self.Vpl_avg[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.title('Plasma Volume ')
				plt.xlabel('time (seconds)')
				plt.ylabel(' volume ml')

				subplot(313)
				plot(xrange_plot, self.V_I_avg[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.title('Plasma Volume ')
				plt.xlabel('time (seconds)')
				plt.ylabel(' volume ml')

				figure(9)
				plot(xrange_plot, self.J_res_avg[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('time(seconds)')
				plt.ylabel('infusion rate (ml/sec)')
				plt.title('infusion rate')

				figure(101)
				plot(self.Vtot_avg[0:self.cycle_m - 1], self.CO[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('volume ml')
				plt.ylabel('cardiac output ml/sec')
				plt.title('Cardiac output - volume relation')

				figure(3)
				plot(self.Vtot_avg[0:self.cycle_m - 1], self.Ox_deliv[0:self.cycle_m - 1], color=self.x[self.col_fac - 1, :])
				plt.xlabel('volume ml')
				plt.ylabel('O2  ml/sec')
				plt.title('Oxygen delivary - volume relation')

				figure(11)
				plot(self.Vtot_avg[0:self.cycle_m - 1], self.Hct_avg[0:self.cycle_m - 1],color=self.x[self.col_fac - 1, :])
				plt.xlabel('volume ml')
				plt.ylabel('Hematocrite')
				plt.title('Hematocrite- volume relation')

				self.refpx = cycle
				self.ref1 = self.ref1 + 1
				self.refx = cycle - self.start_plot
				self.Pt = 1
				self.cycle_m = 1
				self.cycle_one = self.cycle_one + 1
		plt.show()
		return self


if __name__ == "__main__":
	simulation = Sim()
	simulation = simulation.run_simulation()
