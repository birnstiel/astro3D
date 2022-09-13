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
    
    integer :: ix, iy, new(nc), idx(1)
    double precision :: error(nc), old(nc)
    
    
    output = input
    
    do iy=1, ny
        do ix=1, nx
            old = output(ix, iy, :)
            new = 0.0
            ! we first put all over the threshold to 1.0
            WHERE(old > 0.5) new = 1.0

            ! but there can be only one, so if there are more, we take the larges
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

ix = 1
iy = 1
iz = 1

!$OMP PARALLEL PRIVATE(ix, iy, iz, xd, yd, zd, c00,c01,c10,c11,c0,c1) SHARED(newvals)
!$OMP DO
do ip = 1, np

    ! find the left indices and return zero if any one is out of range

    if ( &
         & (points(ip, 1) .le. x(1)) .or. (points(ip, 1) .ge. x(nx)) .or. &
         & (points(ip, 2) .le. y(1)) .or. (points(ip, 2) .ge. y(ny)) .or. &
         & (points(ip, 3) .le. z(1)) .or. (points(ip, 3) .ge. z(nz)) &
         & ) then
        newvals(ip) = fill_value
        CYCLE
    endif

    call hunt(x, nx, points(ip, 1), ix)
    call hunt(y, ny, points(ip, 2), iy)
    call hunt(z, nz, points(ip, 3), iz)

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

end module