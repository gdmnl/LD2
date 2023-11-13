from propagation cimport A2prop

cdef class A2Prop:
	cdef A2prop c_a2prop

	def __cinit__(self):
		self.c_a2prop=A2prop()

	def propagate(self,dataset,prop_alg,unsigned int m,unsigned int n,unsigned int seed,int L,rmax,alpha,ra,rb,np.ndarray array3):
		return self.c_a2prop.propagate(dataset.encode(),prop_alg.encode(),m,n,seed,L,rmax,alpha,ra,rb,Map[MatrixXf](array3))