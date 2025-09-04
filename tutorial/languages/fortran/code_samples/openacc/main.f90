program matmul_acc
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
      print *, "Usage: ./matmul_acc_cli [N]  (N must be a positive integer)"
      stop 3
    end if
  end if

  print *, "Using N =", n

  allocate(A(n,n), B(n,n), C(n,n), stat=stat)
  if (stat /= 0) then
    print *, "Allocation failed (try a smaller N)."
    stop 4
  end if

  !-------------------------------
  ! OpenACC section
  ! Keep arrays on device; initialize A,B on the GPU; compute C=A*B on the GPU;
  ! then copy C back to host for printing.
  !-------------------------------
  !$acc data create(A(1:n,1:n), B(1:n,1:n), C(1:n,1:n))
    ! Initialize A and B on device
    !$acc parallel loop collapse(2)
    do j = 1, n
      do i = 1, n
        A(i,j) = real(i + j, real64)
        B(i,j) = real(i - j, real64)
      end do
    end do

    ! Naïve C = A*B with OpenACC
    !$acc parallel loop collapse(2) private(sum)
    do j = 1, n
      do i = 1, n
        sum = 0.0_real64
        !$acc loop seq
        do k = 1, n
          sum = sum + A(i,k) * B(k,j)
        end do
        C(i,j) = sum
      end do
    end do

    !$acc update host(C(1:n,1:n))
  !$acc end data
  !-------------------------------

  print *, "C(1,1)=", C(1,1), "  C(n,n)=", C(n,n)

  deallocate(A,B,C)
end program matmul_acc
