from libc.stdlib cimport malloc, free
from propagation cimport A2prop, Channel

cdef class A2Prop:
	cdef A2prop c_a2prop

	def __cinit__(self):
		self.c_a2prop = A2prop()

	def load(self, str dataset, unsigned int m, unsigned int n, unsigned int seed):
		self.c_a2prop.load(dataset.encode(), m, n, seed)

	def compute(self, unsigned int nchn, chns, np.ndarray feat):
		cdef:
			Channel* c_chns = <Channel*> malloc(nchn * sizeof(Channel))
			float res
		for i in range(nchn):
			c_chns[i].type = chns[i]['type']
				# -1        -2
				# ASE(ADJ), ASE(ADJ2)
            	# 0    1     2     3
            	# ADJ, ADJi, ADJ2, ADJi2,
            	# 4    5     6     7
            	# LAP, LAPi, LAP2, LAPi2
			if c_chns[i].type >= 0:
				c_chns[i].powl = 1 + (chns[i]['type'] // 2) % 2
				c_chns[i].is_idt = (chns[i]['type'] % 2 == 1)
				c_chns[i].is_adj = (chns[i]['type'] < 4)
			else:
				c_chns[i].powl   = -chns[i]['type']
				c_chns[i].is_idt = False
				c_chns[i].is_adj = (chns[i]['type'] > -3)

			c_chns[i].hop = chns[i]['hop']
			c_chns[i].dim = chns[i]['dim']
			c_chns[i].delta = chns[i]['delta']
			c_chns[i].alpha = chns[i]['alpha']
			c_chns[i].rra = chns[i]['rra']
			c_chns[i].rrb = chns[i]['rrb']

		res = self.c_a2prop.compute(nchn, c_chns, Map[MatrixXf](feat))
		free(c_chns)
		return res
