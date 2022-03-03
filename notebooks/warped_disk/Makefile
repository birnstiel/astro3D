run: warped_disk_test.py
	python $<

%.py : %.ipynb
	jupyter nbconvert --to python $<

clean:
	-rm amr_grid.inp
	-rm dust_density.inp
	-rm dust_temperature.dat
	-rm dustopac.inp
	-rm gas_velocity.inp
	-rm image.out
	-rm lines.inp
	-rm microturbulence.inp
	-rm numberdens_co.inp
	-rm radmc3d.inp
	-rm radmc3d.out
	-rm stars.inp
	-rm wavelength_micron.inp