module lic
    implicit none
    DOUBLE PRECISION :: fill_value=0d0

    contains

    subroutine flic(data, x, y, vel, length, n_steps, output, nx, ny)
        implicit none
        INTEGER, INTENT(IN) :: nx, ny
        !f2py DOUBLE PRECISION OPTIONAL, INTENT(IN) :: length = 1.0
        !f2py INTEGER OPTIONAL, INTENT(IN) :: n_steps = 20
        DOUBLE PRECISION, INTENT(IN) :: data(nx, ny)
        DOUBLE PRECISION, INTENT(IN) :: x(nx), y(ny)
        DOUBLE PRECISION, INTENT(IN) :: vel(nx, ny, 2)
        DOUBLE PRECISION, INTENT(IN) :: length
        DOUBLE PRECISION, INTENT(OUT) :: output(nx, ny)
        INTEGER, INTENT(IN) :: n_steps
        
        DOUBLE PRECISION :: ds(1), integral
        DOUBLE PRECISION :: values(2 * n_steps - 1)
        DOUBLE PRECISION :: path(2 * n_steps - 1, 2)
        INTEGER :: ipath(2 * n_steps - 1, 2)
        INTEGER :: ix, iy, n
        
        do ix = 1, nx
            do iy = 1, ny

                call calc_2D_streamline_bothways(x(ix), y(iy), x, y, vel, data, path, ipath, values, length, n_steps, nx, ny)
                ds = SQRT(SUM( (path(2:nx, :) - path(1:nx-1, :))**2, DIM=2))

                integral = 0d0
                do n = 1, 2 * n_steps - 2
                    integral = integral + ds(n) * 0.5 * (values(n) + values(n+1))
                enddo

                output(ix, iy) = integral / SUM(ds)
            end do
        end do

    end subroutine flic

    ! take a 1st order RK step with given velocity field for length ds
    ! p: the point (x, y) to be updated.
    ! ds: length of the step
    ! x, y: the regular grids in x and y direction
    ! vec : velocity v(x, y)
    ! ix, iy : initial guesses for the interpolation; will be updated
    ! nx, ny: size of x, y dimension
    subroutine rkstep(p, ds, x, y, vel, ix, iy, nx, ny)
        implicit none
        INTEGER, INTENT(IN) :: nx, ny
        DOUBLE PRECISION, INTENT(INOUT) :: p(2)
        DOUBLE PRECISION, INTENT(IN) :: x(nx), y(ny), vel(nx, ny, 2)
        DOUBLE PRECISION, INTENT(IN) :: ds
        INTEGER, INTENT(INOUT) :: ix, iy
        
        DOUBLE PRECISION :: dt, v
        DOUBLE PRECISION :: pdot1(2), pdot0(2)

        call interpolate2d(x, y, vel, p(1), p(2), pdot0, ix, iy, nx, ny, 2)

        v = sqrt(sum(pdot0**2))
        if (v == 0d0) then
            return 
        endif

        dt = ds / v
        
        call interpolate2d(x, y, vel, p(1) + dt / 2d0 * pdot0(1), p(2) + dt / 2d0 * pdot0(2), pdot1, ix, iy, nx, ny, 2)

        v = sqrt(sum(pdot1**2))
        if (v == 0d0) then
            return
        endif

        ! update the position
        dt = ds / v
        p = p + dt * pdot1
    end subroutine rkstep

    !helper function that does the integration from point p0 in velocity field pdot, always forward.
    subroutine calc_2D_streamline_forward(x0, y0, x, y, vel, data, path, ipath, values, length, n_steps, nx, ny)
        implicit none
        INTEGER, INTENT(IN) :: nx, ny
        DOUBLE PRECISION, INTENT(IN) :: x0, y0
        DOUBLE PRECISION, INTENT(IN) :: x(nx), y(ny)
        DOUBLE PRECISION, INTENT(IN) :: vel(nx, ny, 2)
        DOUBLE PRECISION, INTENT(IN) :: data(nx, ny)
        DOUBLE PRECISION, INTENT(IN) :: length
        INTEGER, INTENT(IN) :: n_steps
        !f2py DOUBLE PRECISION OPTIONAL, INTENT(IN) :: length = 1.0
        !f2py INTEGER OPTIONAL, INTENT(IN) :: n_steps = 20
        DOUBLE PRECISION, INTENT(OUT) :: path(n_steps + 1, 2)
        INTEGER, INTENT(OUT) :: ipath(n_steps + 1, 2)
        DOUBLE PRECISION, INTENT(OUT) :: values(n_steps + 1)

        INTEGER :: i, ix, iy
        DOUBLE PRECISION :: ds, val(1)
        DOUBLE PRECISION :: data3(nx, ny, 1)
        DOUBLE PRECISION :: p(2)

        data3(:, :, 1) = data
        ds = length / n_steps

        ! get initial index
        call hunt(x, nx, x0, ix)
        call hunt(y, ny, y0, iy)

        ! go forward one length
        p(1) = x0
        p(2) = y0
        path(1, : ) = p
        ipath(1, 1) = ix
        ipath(1, 2) = iy
        values(1) = data(ix, iy)

        do i = 2, n_steps + 1
            call rkstep(p, ds, x, y, vel, ix, iy, nx, ny)
            path(i, :) = p

             ! call the interpolation once more to get the ix, iy of the final position and the value at the place
            call interpolate2d(x, y, data3, p(1), p(2), val, ix, iy, nx, ny, 1)
            ipath(i, 1) = ix
            ipath(i, 2) = iy
            values(i) = val(1)
        enddo
        
    end subroutine

    !helper function that does the integration from point p0 in velocity field pdot, both forward and backward
    subroutine calc_2D_streamline_bothways(x0, y0, x, y, vel, data, path, ipath, values, length, n_steps, nx, ny)
        implicit none
        INTEGER, INTENT(IN) :: nx, ny
        DOUBLE PRECISION, INTENT(IN) :: x0, y0
        DOUBLE PRECISION, INTENT(IN) :: x(nx), y(ny), vel(nx, ny, 2), data(nx, ny)
        DOUBLE PRECISION, INTENT(IN) :: length
        INTEGER, INTENT(IN) :: n_steps
        !f2py DOUBLE PRECISION OPTIONAL, INTENT(IN) :: length = 1.0
        !f2py INTEGER OPTIONAL, INTENT(IN) :: n_steps = 20
        INTEGER :: i, ix, iy
        DOUBLE PRECISION :: ds, val(1)
        DOUBLE PRECISION :: p(2)
        DOUBLE PRECISION :: data3(nx, ny, 1)
        DOUBLE PRECISION, INTENT(OUT) :: path(2 * n_steps - 1, 2)
        DOUBLE PRECISION, INTENT(OUT) :: values(2 * n_steps - 1)
        INTEGER, INTENT(OUT) :: ipath(2 * n_steps - 1, 2)

        data3(:, :, 1) = data

        path = 0d0
        ds = length / n_steps / 2

        ! get initial index
        call hunt(x, nx, x0, ix)
        call hunt(y, ny, y0, iy)

        ! go FORWARD half a length

        p(1) = x0
        p(2) = y0
        path(n_steps + 1, : ) = p
        ipath(n_steps + 1, 1) = ix
        ipath(n_steps + 1, 2) = iy
        values(n_steps + 1) = data(ix, iy)

        do i = n_steps + 2, 2 * n_steps - 1
            call rkstep(p, ds, x, y, vel, ix, iy, nx, ny)
            path(i, :) = p
            
            ! call the interpolation once more to get the ix, iy of the final position and the value at the place
            call interpolate2d(x, y, data3, p(1), p(2), val, ix, iy, nx, ny, 1)
            ipath(i, 1) = ix
            ipath(i, 2) = iy
            values(i) = val(1)
        enddo

        ! go BACKWARD half a length

        p(1) = x0
        p(2) = y0
        ix = ipath(n_steps + 1, 1)
        iy = ipath(n_steps + 1, 2)
        do i = n_steps + 2, 1, -1
            call rkstep(p, -ds, x, y, vel, ix, iy, nx, ny)
            path(i, :) = p

            ! call the interpolation once more to get the ix, iy of the final position and the value at the place
            call interpolate2d(x, y, data3, p(1), p(2), val, ix, iy, nx, ny, 1)
            ipath(i, 1) = ix
            ipath(i, 2) = iy
            values(i) = val(1)
        enddo
        
    end subroutine

        
    ! bilinear interpolation on regular grids
    ! x, y, interpolating z(x, y) at new values
    ! xn, yn.
    subroutine interpolate2d(x, y, z, xn, yn, zn, ix, iy, nx, ny, nz)
        implicit none
        INTEGER, INTENT(in) :: nx, ny, nz
        DOUBLE PRECISION, INTENT(in) :: x(1:nx)
        DOUBLE PRECISION, INTENT(in) :: y(1:ny)
        DOUBLE PRECISION, INTENT(in) :: z(1:nx, 1:ny, 1:nz)
        DOUBLE PRECISION, INTENT(in) :: xn, yn
        DOUBLE PRECISION, INTENT(out) :: zn(1:nz)
        INTEGER, INTENT(inout) :: ix, iy
        DOUBLE PRECISION :: xd, yd, xnn, ynn
        DOUBLE PRECISION :: z0(nz), z1(nz)
        
        ! check for out of range
        
        !if ( &
        !        & (xn .le. x(1)) .or. (xn .ge. x(nx)) .or. &
        !        & (yn .le. y(1)) .or. (yn .ge. y(ny)) &
        !        & ) then
        !    zn = fill_value
        !    return
        !endif
        
        ! search for the left indices
        
        call hunt(x, nx, xn, ix)
        call hunt(y, ny, yn, iy)

        ix = max(1, min(ix, nx - 1))
        iy = max(1, min(iy, ny - 1))

        xnn = max(x(1), min(x(nx), xn))
        ynn = max(y(1), min(y(ny), yn))
        
        ! bilinear interpolation

        xd = (xnn - x(ix)) / (x(ix + 1) - x(ix))
        yd = (ynn - y(iy)) / (y(iy + 1) - y(iy))
        
        z0 = z(ix, iy,   :) * (1d0 - xd) + z(ix+1, iy,   :) * xd
        z1 = z(ix, iy+1, :) * (1d0 - xd) + z(ix+1, iy+1, :) * xd
        
        zn = z0 * (1d0 - yd) + z1 * yd
        
    end subroutine interpolate2d

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

    subroutine gen_noise(nx, ny, noise, n_size)
        implicit none
        !f2py INTEGER OPTIONAL, INTENT(IN) :: n_size = 1
        INTEGER, INTENT(IN) :: nx, ny, n_size
        DOUBLE PRECISION, INTENT(OUT) :: noise(nx, ny)
        DOUBLE PRECISION :: u
        INTEGER :: x(nx, ny), y(nx, ny), ix, iy, i, n_noise
        noise = 0d0
        n_noise = int(nx * ny / n_size**2 / 4.0)
    
        do ix = 1, nx
            do iy = 1, ny
                x(ix, iy) = ix
                y(ix, iy) = iy
            end do
        end do

        do i = 1, n_noise
            call random_number(u)
            ix = INT(1.0 + FLOOR(nx * u))
            call random_number(u)
            iy = INT(1.0 + FLOOR(ny * u))

            noise = noise + exp(-((x - ix)**2 + (y - iy)**2) / (2.0 * n_size**2))
        enddo

    end subroutine gen_noise

    ! =============================================================================

    subroutine gen_noise_fast(nx, ny, noise, sigma, n_noise)
        implicit none
        !f2py DOUBLE PRECISION, INTENT(IN) :: sigma = 1.0
        !f2py INTEGER, INTENT(IN) :: n_noise = -1
        INTEGER, INTENT(INOUT) :: n_noise
        INTEGER, INTENT(IN) :: nx, ny
        DOUBLE PRECISION, INTENT(IN) :: sigma
        DOUBLE PRECISION, INTENT(OUT) :: noise(nx, ny)
        DOUBLE PRECISION :: u, patch(3 * INT(CEILING(sigma)) + 1, 3 * INT(CEILING(sigma)) + 1)
        INTEGER :: n_size
        INTEGER :: ix, iy, ix0, iy0, ixn, iyn, i, n_patch
        n_size = INT(CEILING(sigma))
        noise = 0d0
        if (n_noise<1) then
            n_noise = int(nx * ny / n_size**2 / 8.0)
        endif

        n_patch = 3 * n_size + 1
    
        do ix = 1, n_patch
            do iy = 1, n_patch
                patch(ix, iy) = exp(-((n_size + 1 - ix)**2 + (n_size + 1 - iy)**2) / (2.0 * n_size**2))
            end do
        end do

        do i = 1, n_noise
            call random_number(u)
            ix0 = INT(1.0 + FLOOR(nx * u))
            call random_number(u)
            iy0 = INT(1.0 + FLOOR(ny * u))

            do ix = 1, n_patch
                do iy = 1, n_patch
                    ixn = ix0 - n_size -1 + ix
                    iyn = iy0 - n_size -1 + iy
                    if ((ixn < 1) .or. (ixn > nx) .or. (iyn < 1) .or. (iyn > nx)) continue
                    noise(ixn, iyn) = noise(ixn, iyn) + patch(ix, iy)
                end do
            end do
        enddo

    end subroutine gen_noise_fast

    ! =============================================================================

end module lic