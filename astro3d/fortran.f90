module fmodule
USE OMP_LIB

    DOUBLE PRECISION :: fill_value=0d0

    contains

subroutine render(image, data, x0, sigma, colors, bg, nx, ny, nz, N)
implicit none
INTEGER, INTENT(IN) :: nx, ny, nz, N
DOUBLE PRECISION, INTENT(IN) :: bg
!f2py DOUBLE PRECISION OPTIONAL, INTENT(IN) :: bg = 0.0
DOUBLE PRECISION, intent(out) :: image(nx, ny, 4)
DOUBLE PRECISION, intent(in) :: data(nx, ny, nz)
DOUBLE PRECISION, intent(in), DIMENSION(N) :: x0, sigma
DOUBLE PRECISION, intent(in), DIMENSION(N, 4) :: colors
DOUBLE PRECISION, DIMENSION(nx, ny) :: slice
DOUBLE PRECISION, DIMENSION(nx, ny, 4) :: rgba
integer :: i

image = bg

do i = 1, nz
    slice = data(:, :, i)
    call transferfunction(slice, x0, sigma, colors, nx, ny, n, rgba)
    image(:, :, 1) = rgba(:, :, 4) * rgba(:, :, 1) + (1 - rgba(:, :, 4)) * image(:, :, 1)
    image(:, :, 2) = rgba(:, :, 4) * rgba(:, :, 2) + (1 - rgba(:, :, 4)) * image(:, :, 2)
    image(:, :, 3) = rgba(:, :, 4) * rgba(:, :, 3) + (1 - rgba(:, :, 4)) * image(:, :, 3)
    image(:, :, 4) = image(:, :, 4) + rgba(:, :, 4)
end do

end subroutine render

subroutine transferfunction(x, x0, sigma, colors, nx, ny, n, rgba)
    implicit none

    INTEGER, INTENT(IN) :: nx, ny, n
    DOUBLE PRECISION, intent(in) :: x(nx, ny)
    DOUBLE PRECISION, intent(in), DIMENSION(n) :: x0, sigma
    DOUBLE PRECISION, intent(in), DIMENSION(n, 4) :: colors
    DOUBLE PRECISION, intent(out), DIMENSION(nx, ny, 4) ::  rgba
    DOUBLE PRECISION, DIMENSION(n) :: dum
    integer :: ix, iy, ic
    
    !vals = colors[..., :, :] * A[..., :, None] * np.exp(-(x[..., None, None] - x0[..., :, None])**2 / (2 * sigma[..., :, None]**2))
    ! all the "SPREAD"s do a broadcasting like np.newaxis in numpy to make the arrays of shape (nx, ny, n, 4)

    !$OMP PARALLEL PRIVATE(dum) SHARED(rgba)
    !$OMP DO
    do ix = 1, nx
        do iy = 1, ny
            dum = exp(-(x(ix, iy) - x0)**2 / (2 * sigma**2))
            do ic = 1, 4
                rgba(ix, iy, ic) = SUM(colors(:, ic) * dum)
            end do
        end do
    end do
    !$OMP END DO
    !$OMP END PARALLEL
	
end subroutine transferfunction

! FLOYD-STEINBERG DITHERING
!
! dithers the input image of shape nx, ny to the same-shape output image.
! input should be double precision with values between 0.0 and 1.0
! output will only have 0.0 or 1.0
subroutine dither(input, output, nx, ny)
    integer, intent(in) :: nx, ny
    double precision, intent(in) :: input(nx, ny)
    double precision, intent(out) :: output(nx, ny)
    
    integer :: ix, iy, new
    double precision :: error, old
    
    
    output = input
    
    do iy=1, ny
        do ix=1, nx
            old = output(ix, iy)
            if (old > 0.5) then
                new = 1.0
            else
                new = 0.0
            endif
            output(ix, iy) = new
            error = old - new
    
            if (ix < nx) then
                output(ix+1, iy) = output(ix+1, iy) + error * 0.4375 ! 7/16 
            endif 
            if (iy < ny) then
                if (ix > 1) then
                    output(ix-1, iy+1) = output(ix-1, iy+1) + error * 0.1875 ! 3/16
                endif
                output(ix, iy+1) = output(ix, iy+1) + error * 0.3125 ! 5/16 
                if (ix < nx) then
                    output(ix+1, iy+1) = output(ix+1, iy+1) + error * 0.0625 ! 1 / 16
                endif
            endif
        end do
    end do
                    
    end subroutine dither



! 3-density FLOYD-STEINBERG DITHERING
!
! dither the input densities of shape nx, ny to the same-shape output image that is dithered, 
! containing just one color in each pixel.
! input should be double precision with values between 0.0 and 1.0
! output will only have 0.0 or 1.0
subroutine dither_colors(input, output, nx, ny, nc)
    integer, intent(in) :: nx, ny
    double precision, intent(in) :: input(nx, ny, nc)
    double precision, intent(out) :: output(nx, ny, nc)
    
    integer :: ix, iy, idx(1)
    double precision :: error(nc), old(nc), new(nc)
    
    output = input
    
    do iy=1, ny
        do ix=1, nx
            old = output(ix, iy, :)
            new = 0.0
            ! we first put all over the threshold to 1.0
            WHERE(old > 0.5) new = 1.0

            ! but there can be only one, so if there are more, we take the largest
            if (sum(new) > 1.0) then
                new = 0.0
                idx = maxloc(old)
                if (old(idx(1)) > 0.5) then
                    new(idx(1)) = 1.0
                endif
            endif

            ! copy over and define error
            output(ix, iy, :) = new
            error = old - new

            ! distribute error to neighbors
            if (ix < nx) then
                output(ix+1, iy, :) = output(ix+1, iy, :) + error * 0.4375 ! 7/16 
            endif 
            if (iy < ny) then
                if (ix > 1) then
                    output(ix-1, iy+1, :) = output(ix-1, iy+1, :) + error * 0.1875 ! 3/16
                endif
                output(ix, iy+1, :) = output(ix, iy+1, :) + error * 0.3125 ! 5/16 
                if (ix < nx) then
                    output(ix+1, iy+1, :) = output(ix+1, iy+1, :) + error * 0.0625 ! 1 / 16
                endif
            endif
        end do
    end do
                        
end subroutine dither_colors
    

! _____________________________________________________________________________
! this routine is for a correlated search within an ordered table:
! it starts with a first guess and then does increasing steps until the borders
! are placed around the intended value
! then a bisection search begins
!
! INPUT:    xx()    = the array containing the ordered values
!            n      = size of xx
!            x      = the search value
!            jlo    = first quess for the position of x in xx
!
! RETURNS:    jlo   = the position left of x in xx
!                   = 0     if x < xx(1)
!                   = n     if x > xx(n)
!
! based on the hunt routine from "Numerical Recipes in Fortran 77"
! _____________________________________________________________________________
subroutine hunt(xx, n, x, jlo)
    integer                 :: n
    doubleprecision, intent(in)     :: x,xx(1:n)
    integer,intent(inout)           :: jlo
    integer                         :: inc,jhi,jm
    logical ascnd

    ascnd = xx(n).gt.xx(1)

    if ((jlo .le. 0) .or. (jlo .gt. n)) then    ! if guess is bad proceed with biscetion
        jlo = 0
        jhi = n+1
    else                            ! if quess is ok: hunt
            inc=1                    ! set the increment to 1
            if(x.ge.xx(jlo).eqv.ascnd) then ! hunt up
                    do
                        jhi = jlo + inc        ! jump with j_hi to j_lo + inc

                        if (jhi.gt.n) then     ! if j_hi out of bounds: set j_hi and stop hunting
                            jhi = n + 1
                            exit
                        else if (x.ge.xx(jhi).eqv.ascnd) then ! if we are still too low
                            jlo = jhi
                            inc = inc+inc     ! increase step
                        else
                            exit            ! exit if j_hi not out of bounds and x less xx(j_hi)
                        endif
                    enddo
            else                            ! hunt down
                    jhi = jlo
                    do
                        jlo = jhi - inc     ! jump with j_lo to j_hi - inc
                        if (jlo.lt.1) then     ! if out of bounds: stop hunting
                            jlo=0
                            exit
                        else if (x.lt.xx(jlo).eqv.ascnd) then ! if we are still too high
                            jhi=jlo
                            inc=inc+inc     ! increase step and retry
                        else
                            exit            ! or exit
                        endif
                    enddo
            endif
    endif

    ! start of bisection
    do
        if (jhi-jlo.eq.1) exit

        jm=(jhi+jlo)/2
        if (x.gt.xx(jm).eqv.ascnd) then
            jlo = jm
        else
            jhi = jm
        endif
    enddo
end subroutine hunt
! =============================================================================

subroutine interpolate(x, y, z, vals, points, nx, ny, nz, np, newvals)
implicit none
INTEGER, INTENT(in) :: nx, ny, nz, np
DOUBLE PRECISION, INTENT(in) :: vals(1:nx, 1:ny, 1:nz)
DOUBLE PRECISION, INTENT(in) :: x(1:nx)
DOUBLE PRECISION, INTENT(in) :: y(1:ny)
DOUBLE PRECISION, INTENT(in) :: z(1:nz)
DOUBLE PRECISION, INTENT(in) :: points(1:np, 1:3) 
DOUBLE PRECISION, INTENT(out) ::  newvals(1:np)
INTEGER :: ix, iy, iz, ip
DOUBLE PRECISION :: xd, yd, zd
DOUBLE PRECISION :: c00, c01, c10, c11, c0, c1


!$OMP PARALLEL PRIVATE(ix, iy, iz, ip, xd, yd, zd, c00,c01,c10,c11,c0,c1) SHARED(newvals)
ix = 1
iy = 1
iz = 1
!$OMP DO
do ip = 1, np

    ! find the left indices and return zero if any one is out of range

    if ( &
         & (points(ip, 1) .lt. x(1)) .or. (points(ip, 1) .gt. x(nx)) .or. &
         & (points(ip, 2) .lt. y(1)) .or. (points(ip, 2) .gt. y(ny)) .or. &
         & (points(ip, 3) .lt. z(1)) .or. (points(ip, 3) .gt. z(nz)) &
         & ) then     
        newvals(ip) = fill_value
        CYCLE
    endif

    call hunt(x, nx, points(ip, 1), ix)
    call hunt(y, ny, points(ip, 2), iy)
    call hunt(z, nz, points(ip, 3), iz)

    if ((ix > nx - 1) .or. (iy > ny - 1) .or. (iz > nz - 1)) then
        newvals(ip) = fill_value
        CYCLE
    endif

    ! this follows bilinear/trilinear interpolation from Wikipedia:

    ! calculate distances
    xd = (points(ip, 1) - x(ix)) / (x(ix + 1) - x(ix))
    yd = (points(ip, 2) - y(iy)) / (y(iy + 1) - y(iy))
    zd = (points(ip, 3) - z(iz)) / (z(iz + 1) - z(iz))

    ! first interpolation
    c00 = vals(ix, iy,   iz)   * (1d0 - xd) + vals(ix+1, iy,   iz)   * xd
    c01 = vals(ix, iy,   iz+1) * (1d0 - xd) + vals(ix+1, iy,   iz+1) * xd
    c10 = vals(ix, iy+1, iz)   * (1d0 - xd) + vals(ix+1, iy+1, iz)   * xd
    c11 = vals(ix, iy+1, iz+1) * (1d0 - xd) + vals(ix+1, iy+1, iz+1) * xd

    c0 = c00 * (1d0 - yd) + c10 * yd
    c1 = c01 * (1d0 - yd) + c11 * yd

    newvals(ip) = c0 * (1d0 - zd) + c1 * zd

end do
!$OMP END DO
!$OMP END PARALLEL


end subroutine interpolate

subroutine test_interp()
implicit none
INTEGER, PARAMETER :: n=2
DOUBLE PRECISION ::  vals(n, n, n), x(n), y(n), z(n)
INTEGER :: ix, iy, iz
DOUBLE PRECISION, DIMENSION(1) :: res
DOUBLE PRECISION, DIMENSION(1, 3) :: points

x = (/0d0, 1d0/)
y = (/0d0, 1d0/)
z = (/0d0, 1d0/)

do  ix = 1, n
    do  iy = 1, n
        do  iz = 1, n
            vals(ix, iy, iz) = iz
        end do
    end do
end do

points(1, :) = (/0.5, 0.5, 0.5/)

call interpolate(x, y, z, vals, points, n, n, n, 1, res)

write(*,*) res

end subroutine test_interp

! =============================================================================

subroutine mark_streamline(mask, x, y, z, r, nx, ny, line, np)
implicit none

integer, intent(in) :: nx, ny, np
double precision, intent(in) :: x(nx), y(ny), z, r, line(np, 3)
logical, intent(out) :: mask(nx, ny)

double precision :: d1_sq, d2(3), el(3), d, length
logical :: point_mask(np)
integer :: ix, iy, ip

! first we check which parts of the streamline are within reach
! of our z-layer
! then we add also the neighboring points of the points within reach
WHERE (ABS(line(:, 3) - z) .le. r)
    point_mask = .true.
ELSEWHERE
    point_mask = .false.
ENDWHERE

! now make all neighbors true as well
point_mask(1) = point_mask(1) .or. point_mask(2)
point_mask(2:np-1) = point_mask(2:np-1) .or. point_mask(1:np-2) .or. point_mask(3:np)
point_mask(np) = point_mask(np) .or. point_mask(np-1)


! now for each pixel, we need to compute all distances to the lines and to the points
mask = .false.

do ip = 1, np
    ! if the point mask of this point is not set, we ignore it
    if (.not.(point_mask(ip))) cycle

    do ix = 1, nx
        do iy = 1, ny

            !if point is already colored: we can skip it
            if (mask(ix, iy)) cycle

            ! compute distance to point
            d2 = (/x(ix), y(iy), z/) - line(ip, :)
            if (NORM2(d2) .le. r) then
                mask(ix, iy) = .true.
                cycle
            endif

            ! next: compute the distance to the next line segment if there is one
            if (ip == np) cycle
            if (.not.point_mask(ip+1)) cycle
            
            ! first the unit vector along the line
            el = line(ip+1,:) - line(ip,:)
            length = norm2(el)
            el = el / length

            ! now the projected distances parallel and perpendicular to the line
            ! check that we are within the length of the line and no less than 
            ! one radius away perpendicular to it
            d = dot_product(d2, el)
            d1_sq = dot_product(d2, d2) - d**2
            if ((d1_sq .le. r**2) .and. (d .ge. 0) .and. (d .le. length)) then
                mask(ix, iy) = .true.
            endif
        enddo
    end do
end do

end subroutine mark_streamline

subroutine compute_view(data, i0, i1, step, image, n_tauone, empty_colors, bg, nx, ny, nz, nempty)
    implicit none
    integer, intent(in) :: i0, i1, step, nx, ny, nz, n_tauone, nempty
    integer, intent(in) :: data(nx, ny, nz, 3)
    integer, intent(in) :: empty_colors(nempty, 3)
    double precision, intent(in) :: bg(3)
    double precision, intent(out) :: image(nx, ny, 3)
    double precision :: slice(nx, ny, 3), factor1, tf

    integer :: i, ix, iy, ie

    do i = 1, 3
        image(:, :, i) = bg(i)
    enddo

    factor1 = exp(-1d0 / n_tauone)

    do i = i0, i1, step
        slice = data(:, :, i, :)
        !$OMP PARALLEL PRIVATE(tf, iy, ie) SHARED(image)
        !$OMP DO
        do ix = 1, nx
            do iy = 1, ny
                ! this array is set to 0 for every transparent pixel, and 1 for the rest
                tf = factor1
                do ie = 1, nempty
                    if (all(slice(ix, iy, :) .eq. empty_colors(ie, :))) then
                        tf = 1.0
                    endif
                enddo

                image(ix, iy, :) = image(ix, iy, :) * tf + (1d0 - tf) * slice(ix, iy, :)
            enddo
        enddo
        !$OMP END DO
        !$OMP END PARALLEL
    enddo
end subroutine compute_view


! this routines takes a 3D image stack (nx, ny, nz, nc). Here, nc is the number
! of integers that describe a color, so 3 for RGB, 4 for RGBA. This routine determines
! the unique colors within the stack.
!
! This function is just a wrapper to get_colors_batched, due to the limitation of f2py
! of not being able to use allocatable arrays. Therefore we need to compute
! the array sizes and pass all those to `get_colors_batched`.
!
! stack:
!     the 3D image stack of shape (nx, ny, nz, nc) where nc = 3 or 4 for RGB or RGBA
! nx, ny, nz, nc:
!     determine the shape of the image stack
! nout:
!     how many colours could be in the output. This should be less than 8 as we have no more printing channels
! output:
!     the returned colors that were found, up to nout of them
! nresults:
!     how many colors are *actually* returned. the array output would have space for more
subroutine get_colors(stack, nx, ny, nz, nc, nout, nresults, output)
    implicit none
    integer, intent(in) :: nc, nx, ny, nz, nout
    integer, intent(in) :: stack(nx, ny, nz, nc)
    integer, intent(out) :: output(nout, nc), nresults
    
    integer :: batchsize = 10!1000000
    integer :: nr, n_steps, n_all

    nr = nx * ny * nz
    n_steps = max(nr / batchsize, 1)
    n_all = nout * n_steps

    call get_colors_batched(stack, nx, ny, nz, nc, nr, batchsize, n_steps, n_all, nout, output, nresults)

end subroutine get_colors

! this routines takes an image stack (nx, ny, nz, 3 or 4) and determines the unique colors
! due to the limitation of f2py of not being able to use allocatable arrays, we need to use a 
! wrapper that computes the array sizes and passes all those to this function
! this wrapper is `get_colors`.
! stack:
!     the 3D image stack of shape (nx, ny, nz, nc) where nc = 3 or 4 for RGB or RGBA
! nx, ny, nz, nc:
!     determine the shape of the image stack
! nr: 
!     numbers of rows in the flattened stack, that is (nx*ny*nz, nc)
! batchsize:
!     how many colors are in one batch
! n_steps:
!   full stack is broken up into n_steps batches, so this is n_steps = max(nr / batchsize, 1)
! n_all: 
!   how large the results from all batches are. Each result is up to "nout", so
!   n_all = nout * n_steps
! nout:
!     how many colours could be in the output. This should be less than 8 as we have no more printing channels
! output:
!     the returned colors that were found, up to nout of them
! nresults:
!     how many colors are *actually* returned. the array output would have space for more
subroutine get_colors_batched(stack, nx, ny, nz, nc, nr, batchsize, n_steps, n_all, nout, output, nresults)
    implicit none
    integer, intent(in) :: nr, n_steps, nc, nx, ny, nz, nout, n_all, batchsize
    integer, intent(in) :: stack(nx, ny, nz, nc)
    integer, intent(out) :: output(nout, nc), nresults
    integer :: flatstack(nr, nc), results(nout, nc)
    integer :: allresults(n_all, nc)
    
    integer :: ir, ix, iy, iz, i_step, i_start, i_end, icol

    ! flatten data

    do iz = 1, nz
        do iy = 1, ny
            do ix = 1, nx
                ir = (iz - 1) * nx * ny + (iy - 1) * nx + ix
                flatstack(ir, :) = stack(ix, iy, iz, :)
            enddo
        enddo
    enddo
    
    ! now we break it up into batches
    nresults = 0

    ! we get the results for each batch and store them in allresults
    do i_step = 1, n_steps
        
        i_start = (i_step - 1) * batchsize + 1
        i_end = i_step * batchsize
        if (i_step == n_steps) i_end = nr

        call get_colors_1D(flatstack(i_start:i_end, :), icol, i_end - i_start + 1, nc, nout, results)
        
        allresults(nresults+1:nresults+icol, :) = results(1:icol, :)
        nresults = nresults + icol

    enddo

    ! and call the method on the results of the batches
    icol = nresults
    call get_colors_1D(allresults(1:nresults, :), nresults, icol, nc, nout, output)

end subroutine get_colors_batched


subroutine get_colors_1D(stack, icol, nr_in, nc, nout, cols)
    implicit none
    integer, intent(in) :: nc, nr_in, nout
    integer, intent(out) :: icol
    integer, intent(in) :: stack(nr_in, nc)
    integer, intent(out) :: cols(nout, nc)

    integer :: old_mask(nr_in), new_mask(nr_in)
    integer :: col(nc)
    integer ir, nr, new_counter, old_counter

    nr = nr_in

    ! initialize mask as all indices to be checked
    do ir = 1, nr
        old_mask(ir) = ir
    enddo


    ! now look up unique colors
    icol = 1
    do while (nr > 0)
        ! pick first color in remaining colors
        col = stack(old_mask(1), :)
        ! store it in list of colors
        cols(icol, :) = col

        ! we loop through all elements in mask:
        ! old_counter is the index within mask
        ! the value in mask is the row in stack
        new_counter = 0
        do old_counter = 1, nr
            ir = old_mask(old_counter)
            ! if the color in stack is not a match, we keep it
            if (.not. (ALL(stack(ir, :) == col))) then
                new_counter = new_counter + 1
                new_mask(new_counter) = ir
            end if
        enddo

        ! iteration update
        old_mask = new_mask
        nr = new_counter
        icol = icol + 1
    enddo
    icol = max(1,icol - 1)

end subroutine get_colors_1D


end module