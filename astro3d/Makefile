all: Makefile fortran.f90 lic.f90
	f2py -m fmodule -c fortran.f90
	f2py -m lic -c lic.f90

clean:
	-rm *.o *.mod *.cpython*.so 