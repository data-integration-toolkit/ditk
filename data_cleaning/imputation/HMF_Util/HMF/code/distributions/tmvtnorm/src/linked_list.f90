module linked_list
  implicit none

  ! type matrix row, holds a pointer to the root element of the linked list
  type matrixrow
    type(node),pointer :: first   ! pointer to first node in linked list
    type(node),pointer :: last    ! pointer to last node in linked list
  end type matrixrow

  ! matrix element for sparse matrix elements H[i,j]=v
  type matrixelem
    integer :: i, j
    double precision :: v
  end type matrixelem

  ! define a linked list of matrix elements
  type node
    type(matrixelem) data        ! data
    type(node),pointer::next     ! pointer to the
                                 ! next element
  end type node

  CONTAINS

  ! insert the new matrix element H[i,j]=v to the linked list of row "i"
  subroutine insert_list_element(row, newelem)
    type(matrixrow) :: row
    type(matrixelem) :: newelem

   if (.not. associated(row%first)) then
     allocate(row%first)
     nullify(row%first%next)
     row%first%data = newelem
     row%last => row%first
     !print *,"added element to linked list i=",newelem%i," j=",newelem%j," v=",newelem%v
   else
    allocate(row%last%next)
    nullify(row%last%next%next)
    row%last%next%data = newelem
    row%last => row%last%next
    !print *,"added element to linked list i=",newelem%i," j=",newelem%j," v=",newelem%v
   endif
  end subroutine

  ! remove all elements of the linked list and free memory
  subroutine free_all(row)
    implicit none
    type(matrixrow) :: row
    type(node), pointer :: tmp

    do
     tmp => row%first
     if (associated(tmp) .eqv. .FALSE.) exit
     row%first => row%first%next
     deallocate(tmp)
    end do
  end subroutine free_all


end module linked_list
