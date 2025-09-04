program matmul
  use iso_fortran_env, only: real64
  implicit none
  integer :: i, j, k, n, stat, arglen
  character(len=128) :: arg
  real(real64), allocatable :: A(:,:), B(:,:), C(:,:)
  real(real64) :: sum

  ! Default size; can be overridden by first CLI arg
  n = 512
  if (command_argument_count() >= 1) then
    call get_command_argument(1, arg, length=arglen, status=stat)
    if (stat /= 0) then
      print *, "Error reading argument 1."
      stop 2
    end if
    read(arg(:arglen), *, iostat=stat) n
    if (stat /= 0 .or. n <= 0) then
      print *, "Usage: ./matmul_omp_cli [N]  (N must be a positive integer)"
      stop 3
    end if
  end if

  print *, "Using N =", n

  allocate(A(n,n), B(n,n), C(n,n), stat=stat)
  if (stat /= 0) then
    print *, "Allocation failed (try a smaller N)."
    stop 4
  end if

  ! Initialize
  do j = 1, n
    do i = 1, n
      A(i,j) = real(i + j, real64)
      B(i,j) = real(i - j, real64)
    end do
  end do

  ! Naïve C = A*B with OpenMP
  !$omp parallel do collapse(2) default(none) &
  !$omp& shared(A,B,C,n) private(i,j,k,sum) schedule(static)
  do j = 1, n
    do i = 1, n
      sum = 0.0_real64
      do k = 1, n
        sum = sum + A(i,k) * B(k,j)
      end do
      C(i,j) = sum
    end do
  end do
  !$omp end parallel do

  print *, "C(1,1)=", C(1,1), "  C(n,n)=", C(n,n)
  deallocate(A,B,C)
end program matmul