from libc.stdlib cimport malloc, free
from propagation cimport A2prop, Channel

cdef class A2Prop:
	cdef A2prop c_a2prop

	def __cinit__(self):
		self.c_a2prop=A2prop()

	def propagatea(self, dataset, chns, unsigned int m, unsigned int n, unsigned int nchn, unsigned int seed, np.ndarray feat):
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
				c_chns[i].is_i   = (chns[i]['type'] % 2 == 1)
				c_chns[i].is_adj = (chns[i]['type'] < 4)
			else:
				c_chns[i].powl   = -chns[i]['type']
				c_chns[i].is_i   = False
				c_chns[i].is_adj = (chns[i]['type'] > -3)

			c_chns[i].L = chns[i]['L']
			c_chns[i].rmax = chns[i]['rmax']
			c_chns[i].alpha = chns[i]['alpha']
			c_chns[i].rra = chns[i]['rra']
			c_chns[i].rrb = chns[i]['rrb']

		self.c_a2prop.load(dataset.encode(), m, n, seed)
		res = self.c_a2prop.propagatea(nchn, c_chns, Map[MatrixXf](feat))
		free(c_chns)
		return res
