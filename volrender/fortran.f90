module fmodule

    contains

subroutine render(image, data, x0, Amp, sigma, colors, nx, ny, nz, N)
implicit none
INTEGER, INTENT(IN) :: nx, ny, nz, N
DOUBLE PRECISION, intent(out) :: image(nx, ny, 3)
DOUBLE PRECISION, intent(in) :: data(nx, ny, nz)
DOUBLE PRECISION, intent(in), DIMENSION(N) :: x0, Amp, sigma
DOUBLE PRECISION, intent(in), DIMENSION(N, 4) :: colors
DOUBLE PRECISION, DIMENSION(nx, ny) :: slice
DOUBLE PRECISION, DIMENSION(nx, ny, 4) :: rgba
integer :: i

image = 0.0

do i = 1, nz
    slice = data(:, :, i)
    call transferfunction(slice, x0, Amp, sigma, colors, nx, ny, n, rgba)
    image(:, :, 1) = rgba(:, :, 4) * rgba(:, :, 1) + (1 - rgba(:, :, 4)) * image(:, :, 1)
    image(:, :, 2) = rgba(:, :, 4) * rgba(:, :, 2) + (1 - rgba(:, :, 4)) * image(:, :, 2)
    image(:, :, 3) = rgba(:, :, 4) * rgba(:, :, 3) + (1 - rgba(:, :, 4)) * image(:, :, 3)
end do

end subroutine render

! subroutine transferfunction(x, x0, Amp, sigma, colors, nx, ny, n, rgba)
!     implicit none

!     INTEGER, INTENT(IN) :: nx, ny, n
!     DOUBLE PRECISION, intent(in) :: x(nx, ny)
!     DOUBLE PRECISION, intent(in), DIMENSION(n) :: x0, Amp, sigma
!     DOUBLE PRECISION, intent(in), DIMENSION(n, 4) :: colors
!     DOUBLE PRECISION, intent(out), DIMENSION(nx, ny, 4) ::  rgba
    
!     !vals = colors[..., :, :] * A[..., :, None] * np.exp(-(x[..., None, None] - x0[..., :, None])**2 / (2 * sigma[..., :, None]**2))
!     ! all the "SPREAD"s do a broadcasting like np.newaxis in numpy to make the arrays of shape (nx, ny, n, 4)

!     rgba = SUM( &
!     & SPREAD(SPREAD(COLORS, 1, ny), 1, nx) * SPREAD(SPREAD(SPREAD(Amp, 1, ny), 1, nx), 4, 4) * exp( &
!     & - (SPREAD(SPREAD(SPREAD(x0, 1, ny), 1, nx), 4, 4) - SPREAD(SPREAD(x, 3, n), 4, 4))**2/ &
!     &(2 * SPREAD(SPREAD(SPREAD(sigma, 1, ny), 1, nx), 4, 4)**2) &
!     ), DIM=3)
	
! end subroutine transferfunction

subroutine transferfunction(x, x0, Amp, sigma, colors, nx, ny, n, rgba)
    implicit none

    INTEGER, INTENT(IN) :: nx, ny, n
    DOUBLE PRECISION, intent(in) :: x(nx, ny)
    DOUBLE PRECISION, intent(in), DIMENSION(n) :: x0, Amp, sigma
    DOUBLE PRECISION, intent(in), DIMENSION(n, 4) :: colors
    DOUBLE PRECISION, intent(out), DIMENSION(nx, ny, 4) ::  rgba
    DOUBLE PRECISION, DIMENSION(n) :: dum
    integer :: ix, iy, ic
    
    !vals = colors[..., :, :] * A[..., :, None] * np.exp(-(x[..., None, None] - x0[..., :, None])**2 / (2 * sigma[..., :, None]**2))
    ! all the "SPREAD"s do a broadcasting like np.newaxis in numpy to make the arrays of shape (nx, ny, n, 4)

    do ix = 1, nx
        do iy = 1, ny
            dum = Amp * exp(-(x(ix, iy) - x0)**2 / (2 * sigma**2))
            do ic = 1, 4
                rgba(ix, iy, ic) = SUM(colors(:, ic) * dum)
            end do
        end do
    end do
	
end subroutine transferfunction

end module