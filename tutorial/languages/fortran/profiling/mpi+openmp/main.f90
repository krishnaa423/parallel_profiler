program ring_mm
  use iso_fortran_env, only : int32, int64, real64
  use mpi_f08
  implicit none

  integer(int32)           :: rank, size, ierr
  integer(int64)           :: N
  integer(int64), allocatable :: row_counts(:), row_offs(:), k_counts(:), k_offs(:)
  integer(int64)           :: m_i, k_i
  real(real64),  allocatable :: A_local(:,:), B_curr(:,:), B_next(:,:), C_local(:,:), A_sub(:,:)
  integer(int32)           :: left, right
  integer(int32)           :: step
  integer(int64)           :: owner, next_owner, ks, ke, recv_rows
  real(real64)             :: t0, t1, dt, t_max, max_err, local_err
  real(real64)             :: gflops
  integer(int32)           :: argn
  character(len=:), allocatable :: arg

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

  ! --- Parse N from command line ---
  call get_command_argument(1, length=argn)
  if (argn <= 0) then
     if (rank == 0) write(*,*) 'Usage: mpirun -np P ./ring_mm N'
     call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  allocate(character(len=argn) :: arg)
  call get_command_argument(1, arg)
  read(arg, *, iostat=ierr) N
  if (ierr /= 0 .or. N < 1_int64) then
     if (rank == 0) write(*,*) 'N must be a positive integer'
     call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if

  ! --- Partition rows among ranks (1D block, nearly equal) ---
  allocate(row_counts(size), row_offs(size))
  call split_sizes(N, size, row_counts, row_offs)

  ! Reuse same partitioning for B's k-panels
  k_counts => row_counts
  k_offs   => row_offs

  m_i = row_counts(rank+1_int32 - 1_int32 + 1_int32)  ! adjust indexing clarity
  m_i = row_counts(rank+1)   ! Fortran arrays are 1-based: rank 0 -> index 1
  k_i = k_counts(rank+1)

  ! --- Allocate local blocks ---
  allocate(A_local(m_i, N))
  allocate(B_curr(k_i, N))
  allocate(C_local(m_i, N))
  A_local = 1.0_real64
  B_curr  = 1.0_real64
  C_local = 0.0_real64

  ! --- Ring neighbors ---
  left  = modulo(rank - 1, size)
  right = modulo(rank + 1, size)

  owner = int(rank, int64)  ! each rank starts owning its own panel (0..size-1)

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t0 = MPI_Wtime()

  ! --- Ring over all B panels ---
  do step = 0, size-1
     ks = k_offs(owner+1) + 1_int64                      ! start col (1-based)
     ke = ks + k_counts(owner+1) - 1_int64               ! end   col (inclusive)

     if (m_i > 0_int64 .and. k_counts(owner+1) > 0_int64) then
        A_sub => A_local(:, ks:ke)                       ! shape (m_i x k_blk)
        ! Local GEMM: C_local += A_sub * B_curr
        C_local = C_local + matmul(A_sub, B_curr)        ! uses intrinsic MATMUL
        nullify(A_sub)
     end if

     if (size > 1) then
        next_owner = modulo(owner - 1_int64, int(size, int64))
        recv_rows  = k_counts(next_owner+1)
        if (allocated(B_next)) deallocate(B_next)
        allocate(B_next(recv_rows, N))

        call MPI_Sendrecv(B_curr,  dest=right, sendtag=step, &
                          recvbuf=B_next, source=left,  recvtag=step, &
                          comm=MPI_COMM_WORLD, status=MPI_STATUS_IGNORE, ierror=ierr)

        if (allocated(B_curr)) deallocate(B_curr)
        call move_alloc(B_next, B_curr)
        owner = next_owner
     end if
  end do

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t1 = MPI_Wtime()
  dt = t1 - t0

  ! --- Verification: A=B=1 => C == N everywhere ---
  if (size(C_local) > 0) then
     local_err = maxval(abs(C_local - real(N, kind=real64)))
  else
     local_err = 0.0_real64
  end if

  call MPI_Allreduce(local_err, max_err, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(dt,       t_max,   1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, ierr)

  if (t_max > 0.0_real64) then
     gflops = (2.0_real64 * real(N,real64) * real(N,real64) * real(N,real64)) / (t_max * 1.0e9_real64)
  else
     gflops = huge(1.0_real64)
  end if

  if (rank == 0) then
     write(*,'(A, I0, A, I0, A, F0.6, A, F0.3, A, ES10.3)') &
       'RESULT algo=ring_mm N=', N, ' P=', size, ' time=', t_max, ' gflops=', gflops, ' max_err=', max_err
  end if

  call MPI_Finalize(ierr)
contains

  subroutine split_sizes(n, p, counts, offs)
    integer(int64), intent(in)  :: n
    integer(int32), intent(in)  :: p
    integer(int64), intent(out) :: counts(p), offs(p)
    integer(int64) :: base, r
    integer(int32) :: i

    base = n / p
    r    = mod(n, int(p, int64))
    do i = 1, p
       if (int(i-1, int64) < r) then
          counts(i) = base + 1_int64
       else
          counts(i) = base
       end if
    end do
    offs(1) = 0_int64
    do i = 2, p
       offs(i) = offs(i-1) + counts(i-1)
    end do
  end subroutine split_sizes

end program ring_mm
