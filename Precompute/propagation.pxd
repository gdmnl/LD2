from eigency.core cimport *
from libcpp.string cimport string

ctypedef unsigned int uint

cdef extern from "prop.cpp":
	pass

cdef extern from "prop.h" namespace "propagation":
	cdef cppclass A2prop:
		A2prop() except+
		# 				dataset,  alg,   m,   n,seed, L,  rmax,alpha,   ra,   rb, feat
		float propagate(string,string,uint,uint,uint,int,float,float,float,float,Map[MatrixXf] &)