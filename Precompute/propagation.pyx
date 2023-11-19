from libc.stdlib cimport malloc, free
from propagation cimport A2prop, Channel

cdef class A2Prop:
	cdef A2prop c_a2prop

	def __cinit__(self):
		self.c_a2prop=A2prop()

	def propagatea(self, dataset, schemes, unsigned int m, unsigned int n, unsigned int nsch, unsigned int seed, np.ndarray feat):
		cdef:
			Channel* schs = <Channel*> malloc(nsch * sizeof(Channel))
			float res
		for i in range(nsch):
			schs[i].type = schemes[i]['type']
            	# 0    1     2     3
            	# ADJ, ADJi, ADJ2, ADJi2,
            	# 4    5     6     7
            	# LAP, LAPi, LAP2, LAPi2
			schs[i].powl = 1 + (schemes[i]['type'] // 2) % 2
			schs[i].is_i   = (schemes[i]['type'] % 2 == 1)
			schs[i].is_adj = (schemes[i]['type'] < 4)

			schs[i].L = schemes[i]['L']
			schs[i].rmax = schemes[i]['rmax']
			schs[i].alpha = schemes[i]['alpha']
			schs[i].rra = schemes[i]['rra']
			schs[i].rrb = schemes[i]['rrb']

		self.c_a2prop.load(dataset.encode(), m, n, nsch, seed, Map[MatrixXf](feat))
		res = self.c_a2prop.propagatea(schs, Map[MatrixXf](feat))
		free(schs)
		return res
