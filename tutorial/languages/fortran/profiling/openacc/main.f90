program acc_dot
  implicit none
  integer(8) :: n, i
  real(8), allocatable :: A(:), B(:)
  real(8) :: sum, t0, t1

  if (command_argument_count() < 1) then
     print *, "usage: acc_dot n"
     stop
  end if
  call get_command_argument(1, n)

  allocate(A(n), B(n))
  do i = 1, n
     A(i) = dble(i-1)
     B(i) = 1.0d0/dble(i)
  end do

  call cpu_time(t0)
  sum = 0.0d0

!$acc data copyin(A,B) copy(sum)
!$acc parallel loop reduction(+:sum)
  do i = 1, n
     sum = sum + A(i)*B(i)
  end do
!$acc end data

  call cpu_time(t1)
  print '(a, i12, a, f20.12, a, f10.3)', "[OpenACC] n=", n, " sum=", sum, " time=", t1-t0, " s"
end program
