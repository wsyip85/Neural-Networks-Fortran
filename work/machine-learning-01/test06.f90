module utility 
  implicit none 
    contains 
      subroutine count_rows (fin,nrows)
        character(:),allocatable,intent(in)::fin
        integer,intent(out)::nrows
        nrows = 0
        open(10,file=fin) 
        do 
          read(10,*,end=100)
          nrows = nrows + 1
        enddo
        100 close(10)
      end subroutine
      subroutine read_train (fin,X1,X2,Y) 
        character(:),allocatable,intent(in)::fin
        real,dimension(:),allocatable::X1,X2,Y
        integer::n
        n = 1
        open(10,file=fin) 
        do 
          read(10,*,end=100)X1(n),X2(n),Y(n) 
          n = n + 1
        enddo
        100 close (10) 
      end subroutine 
      subroutine mean_stddev_standardize ( X, mu, sigma, XNorm ) 
        real,dimension(:),allocatable,intent(in)::X
        real,intent(out)::mu,sigma
        real,dimension(:),allocatable,intent(out)::XNorm
        mu = sum(X) / size(X) 
        sigma = sqrt( sum((X - mu)**2)/size(X) )
        XNorm = (X - mu) / sigma 
      end subroutine 
      subroutine random_stduniform( u, nrows, ncols ) 
        real,dimension(:,:),allocatable,intent(out)::u
        integer,intent(in)::nrows,ncols
        real,dimension(:,:),allocatable::r
        allocate( u(nrows,ncols), r(nrows,ncols) )
        call random_number(r) 
        u = 1 - r 
      end subroutine
      subroutine random_stdnormal( x, nrows, ncols, mu, sigma ) 
        real,intent(in)::mu,sigma
        real,parameter::pi=3.14159265
        real,dimension(:,:),allocatable,intent(out)::x
        integer,intent(in)::nrows,ncols
        real,dimension(:,:),allocatable::u1,u2
        allocate( u1(nrows,ncols), u2(nrows,ncols), x(nrows,ncols) )
        call random_stduniform( u1, nrows, ncols )
        call random_stduniform( u2, nrows, ncols ) 
        x = sigma * sqrt( -2 * log( u1 ) ) * cos( 2 * pi * u2 ) + mu
      end subroutine 
      subroutine mean_stddev( x, mu, sigma ) 
        real,dimension(:,:),allocatable,intent(in)::x
        real,intent(out)::mu,sigma
        mu = sum(x)/size(x)
        sigma = sqrt( sum( (x - sum(x)/size(x))**2 )/size(x) )
      end subroutine 
      subroutine glorot_init ( w, nrows, ncols, fan_in, fan_out ) 
        real,dimension(:,:),allocatable,intent(out)::w
        integer,intent(in)::nrows,ncols,fan_in,fan_out
        real::mu,sigma
        mu = 0.0
        sigma = sqrt( 2.0 / (fan_in+fan_out) ) 
        call random_stdnormal( w, nrows, ncols, mu, sigma ) 
      end subroutine 
      subroutine grad_htan( x, y )
        real,dimension(:,:),allocatable,intent(in)::x
        real,dimension(:,:),allocatable,intent(out)::y
        y = 1. - tanh( x ) ** 2.
      end subroutine  
      subroutine DenseLayer ( x, w, b, oc, doc_dob, dob_doa, dob_db, doa_dw, doa_dx ) 
        real,dimension(:,:),allocatable,intent(in)::x,w
        real,intent(in)::b
        real,dimension(:,:),allocatable,intent(out)::oc, doc_dob, dob_doa, dob_db, doa_dw, doa_dx
        real,dimension(:,:),allocatable::oa,ob
        real,dimension(:,:),allocatable::xT 
        real,dimension(:,:),allocatable::doa_dxT 
        allocate( oa( size(x,1), size(w,2) ) ) 
        oa = matmul( x, w ) 
        allocate( xT( size(x,2), size(x,1) ) ) 
        xT = transpose( x ) 
        xT = 1.
        allocate( doa_dxT( size(w,1), size(xT,2) ) ) 
        doa_dxT = matmul( w, xT ) 
        allocate( doa_dx( size(doa_dxT,2), size(doa_dxT,1) ) ) 
        doa_dx = transpose( doa_dxT ) 
        allocate( doa_dw( size(x,2), 2 ) )  
        doa_dw(1,1) = x(1,1)
        doa_dw(2,1) = x(1,2) 
        doa_dw(1,2) = x(1,1)
        doa_dw(2,2) = x(1,2)  
        allocate( ob( size(oa,1), size(oa,2) ) ) 
        ob = oa + b
        allocate( dob_doa( size(oa,1), size(oa,2) ) )
        dob_doa = 1. 
        allocate( dob_db( size(oa,1), size(oa,2) ) )
        dob_db = 1. 
        allocate( oc( size(ob,1), size(ob,2) ) ) 
        oc = tanh( ob ) 
        allocate( doc_dob( size(ob,1), size(ob,2) ) )
        call grad_htan( ob, doc_dob )
      end subroutine
      subroutine OutputLayer( x, w, b, ob, dob_db, dob_doa, doa_dw, doa_dx )
        real,dimension(:,:),allocatable,intent(in)::x,w
        real,intent(in)::b
        real,dimension(:,:),allocatable,intent(out)::ob,dob_db,dob_doa,doa_dw,doa_dx
        real,dimension(:,:),allocatable::oa
        allocate( oa( size(x,1), size(w,2) ) ) 
        oa = matmul( x, w ) 
        allocate( doa_dx( size(w,2), size(w,1) ) )
        doa_dx = transpose( w ) 
        allocate( doa_dw( size(x,2), size(x,1) ) ) 
        doa_dw = transpose( x ) 
        allocate( ob( size(oa,1), size(oa,2) ) ) 
        ob = oa + b 
        allocate( dob_doa( size(oa,1), size(oa,2) ) )
        dob_doa = 1.
        allocate( dob_db( size(oa,1), size(oa,2) ) )
        dob_db = 1. 
      end subroutine
      subroutine LossLayer ( o, y, L, dL_do )
        real,dimension(:,:),allocatable,intent(in)::o,y
        real,dimension(:,:),allocatable,intent(out)::L,dL_do
        allocate( L( size(o,1), size(o,2) ), dL_do( size(o,1), size(o,2) ) ) 
        L = ( o - y ) ** 2. 
        dL_do = 2 * ( o - y ) 
      end subroutine  
end module
program main 
  use utility 
  implicit none
    character(:),allocatable::datatrain,datatest,datadir
    integer::nfile
    character::cha*2
    integer::ntrain,ntest
    integer::i
    real,dimension(:),allocatable::Xtrain1, Xtrain2, Ytrain 
    real,dimension(:),allocatable::Xtest1, Xtest2, Ytest 
    real::mu_Xtrain1,sigma_Xtrain1
    real::mu_Xtrain2,sigma_Xtrain2
    real::mu_Ytrain,sigma_Ytrain
    real::mu_Xtest1,sigma_Xtest1
    real::mu_Xtest2,sigma_Xtest2
    real::mu_Ytest,sigma_Ytest
    real,dimension(:),allocatable::Xtrain1Norm, Xtrain2Norm, YtrainNorm
    real,dimension(:),allocatable::Xtest1Norm, Xtest2Norm, YtestNorm
    real,dimension(:,:),allocatable::w1,w2,w3
    real::b1,b2,b3
    real,dimension(:,:),allocatable::x
    integer::nData
    real,dimension(:,:),allocatable::o1c, do1c_do1b, do1b_do1a, do1b_db1, do1a_dw1, do1a_dx
    real,dimension(:,:),allocatable::o2c, do2c_do2b, do2b_do2a, do2b_db2, do2a_dw2, do2a_do1c
    real,dimension(:,:),allocatable::o3b, do3b_db3,  do3b_do3a, do3a_dw3, do3a_do2c
    real,dimension(:,:),allocatable::y
    real,dimension(:,:),allocatable::L, dL_do3b
    real,dimension(:,:,:),allocatable::delta_b3,delta_w3,delta_b2,delta_w2,delta_b1,delta_w1
    real,dimension(:,:),allocatable::tmp12
    integer,parameter::nbatch=10
    integer::batch_pos
    integer::pos,posmax
    real::lr
    integer::count_batch
    integer::epoch
    integer,parameter::Nepoch=5000
    real,dimension(:),allocatable::Loss
    real,dimension(:),allocatable::epoch_b3,epoch_b2,epoch_b1
    real,dimension(:,:,:),allocatable::epoch_w3,epoch_w2,epoch_w1
    integer::idx
    real,dimension(:,:),allocatable::best_w1,best_w2,best_w3
    real::best_b1,best_b2,best_b3
    real,dimension(:,:),allocatable::grad_w1_01,grad_w2_01,grad_w3_01
    real,dimension(:,:),allocatable::grad_w1_02,grad_w2_02,grad_w3_02
    real,dimension(:,:),allocatable::grad_w1_03,grad_w2_03,grad_w3_03
    real,dimension(:,:),allocatable::grad_w1_04,grad_w2_04,grad_w3_04
    real::grad_b1_01,grad_b2_01,grad_b3_01
    real::grad_b1_02,grad_b2_02,grad_b3_02
    real::grad_b1_03,grad_b2_03,grad_b3_03
    real::grad_b1_04,grad_b2_04,grad_b3_04
    real,parameter::momen=0.9
    integer::nmomen
    real,dimension(:),allocatable::Yhat,LTest
    real,dimension(:,:),allocatable::Ltmp
    integer::epoch_count
    allocate( grad_w1_01(2,2), grad_w2_01(2,2), grad_w1_02(2,2), grad_w2_02(2,2), grad_w1_03(2,2), grad_w2_03(2,2), grad_w1_04(2,2), grad_w2_04(2,2) )
    allocate( grad_w3_01(2,1), grad_w3_02(2,1), grad_w3_03(2,1), grad_w3_04(2,1) )
    grad_w1_01 = 0.
    grad_w2_01 = 0.
    grad_w1_02 = 0.
    grad_w2_02 = 0.
    grad_w1_03 = 0.
    grad_w2_03 = 0.
    grad_w1_04 = 0.
    grad_w2_04 = 0.
    grad_w3_01 = 0.
    grad_w3_02 = 0.
    grad_w3_03 = 0.
    grad_w3_04 = 0.
    grad_b1_01 = 0.
    grad_b2_01 = 0.
    grad_b3_01 = 0.
    grad_b1_02 = 0.
    grad_b2_02 = 0.
    grad_b3_02 = 0.
    grad_b1_03 = 0.
    grad_b2_03 = 0.
    grad_b3_03 = 0.
    grad_b1_04 = 0.
    grad_b2_04 = 0.
    grad_b3_04 = 0.
    !write(cha,'(I0.2)')nfile
    call getarg(1,cha) 
    datadir = '../../data/'
    datatrain = datadir // 'train' //cha// '.txt'
    call count_rows( datatrain, ntrain ) 
    allocate( Xtrain1(ntrain), Xtrain2(ntrain), Ytrain(ntrain) ) 
    call read_train (datatrain,Xtrain1,Xtrain2,Ytrain)
    call mean_stddev_standardize ( Xtrain1, mu_Xtrain1, sigma_Xtrain1, Xtrain1Norm ) 
    call mean_stddev_standardize ( Xtrain2, mu_Xtrain2, sigma_Xtrain2, Xtrain2Norm ) 
    call mean_stddev_standardize ( Ytrain, mu_Ytrain, sigma_Ytrain, YtrainNorm ) 
    allocate( w1(2,2), w2(2,2), w3(2,1) ) 
    allocate( tmp12(1,2) ) 
    allocate( x(1,2) )
    allocate( y(1,1) ) 
    allocate( delta_b3(nbatch,1,1) )
    allocate( delta_w3(nbatch,2,1) ) 
    allocate( delta_b2(nbatch,1,1) ) 
    allocate( delta_w2(nbatch,2,2) )
    allocate( delta_b1(nbatch,1,1) )
    allocate( delta_w1(nbatch,2,2) )
    allocate( Loss(Nepoch) ) 
    allocate( epoch_b3(Nepoch) )
    allocate( epoch_b2(Nepoch) )
    allocate( epoch_b1(Nepoch) )
    allocate( epoch_w3(Nepoch,2,1) )
    allocate( epoch_w2(Nepoch,2,2) )
    allocate( epoch_w1(Nepoch,2,2) )
    101 call glorot_init( w1, 2, 2, 2, 2 ) 
    call glorot_init( w2, 2, 2, 2, 2 ) 
    call glorot_init( w3, 2, 1, 2, 1 )
    b1 = 0.
    b2 = 0.
    b3 = 0.
    delta_b3 = 0. 
    delta_w3 = 0.
    posmax = size( Xtrain1Norm, 1 ) / nbatch 
    lr = 0.00001
    do epoch = 1, Nepoch
      epoch_count = 1
      do pos = 1, posmax
        batch_pos = 1
        do count_batch = (nbatch * (pos-1) + 1), nbatch * pos  
          nData = count_batch  
          x(1,1) = Xtrain1Norm(nData)
          x(1,2) = Xtrain2Norm(nData)
          y(1,1) = YtrainNorm(nData) 
          call DenseLayer (  x,  w1, b1, o1c, do1c_do1b, do1b_do1a, do1b_db1, do1a_dw1, do1a_dx )
          call DenseLayer ( o1c, w2, b2, o2c, do2c_do2b, do2b_do2a, do2b_db2, do2a_dw2, do2a_do1c )
          call OutputLayer( o2c, w3, b3, o3b, do3b_db3,  do3b_do3a, do3a_dw3, do3a_do2c )
          call LossLayer ( o3b, y, L, dL_do3b )
          delta_b3(batch_pos,:,:) = sum( dL_do3b * do3b_db3  ) 
          delta_w3(batch_pos,:,:) = sum( dL_do3b * do3b_do3a  ) * do3a_dw3
          delta_b2(batch_pos,:,:) = sum( dL_do3b * do3b_do3a ) * sum( do3a_do2c * do2c_do2b * do2b_db2 ) 
          tmp12 = 0.
          tmp12 = sum( dL_do3b * do3b_do3a ) * do3a_do2c * do2c_do2b * do2b_do2a
          delta_w2(batch_pos,1,1) = tmp12(1,1) * do2a_dw2(1,1)
          delta_w2(batch_pos,1,2) = tmp12(1,2) * do2a_dw2(1,2)
          delta_w2(batch_pos,2,1) = tmp12(1,1) * do2a_dw2(2,1)
          delta_w2(batch_pos,2,2) = tmp12(1,2) * do2a_dw2(2,2)
          delta_b1(batch_pos,:,:) = sum( dL_do3b * do3b_do3a ) * sum( do3a_do2c * do2c_do2b * do2b_do2a * do2a_do1c * do1c_do1b * do1b_db1 ) 
          tmp12 = 0.
          tmp12 = sum( dL_do3b * do3b_do3a ) * do3a_do2c * do2c_do2b * do2b_do2a * do2a_do1c * do1c_do1b * do1b_do1a
          delta_w1(batch_pos,1,1) = tmp12(1,1) * do1a_dw1(1,1)
          delta_w1(batch_pos,1,2) = tmp12(1,2) * do1a_dw1(1,2)
          delta_w1(batch_pos,2,1) = tmp12(1,1) * do1a_dw1(2,1)
          delta_w1(batch_pos,2,2) = tmp12(1,2) * do1a_dw1(2,2)
          batch_pos = batch_pos + 1
          if (epoch_count.eq.1) then
            Ltmp = L 
          else
            Ltmp = Ltmp + L
          endif
          epoch_count = epoch_count + 1
        enddo
        nmomen = 1
        b3 = b3 - sum( lr * sum(delta_b3,1) / size(delta_b3,1) ) - momen * grad_b3_01 - momen ** 2 * grad_b3_02 - momen ** 3 * grad_b3_03 - momen ** 4 * grad_b3_04
        w3 = w3 - ( lr * sum(delta_w3,1) / size(delta_w3,1) ) - momen * grad_w3_01 - momen ** 2 * grad_w3_02 - momen ** 3 * grad_w3_03 - momen ** 4 * grad_w3_04
        b2 = b2 - sum( lr * sum(delta_b2,1) / size(delta_b2,1) ) - momen * grad_b2_01 - momen ** 2 * grad_b2_02 - momen ** 3 * grad_b2_03 - momen ** 4 * grad_b2_04
        w2 = w2 - ( lr * sum(delta_w2,1) / size(delta_w2,1) ) - momen * grad_w2_01 - momen ** 2 * grad_w2_02 - momen ** 3 * grad_w2_03 - momen ** 4 * grad_w2_04
        b1 = b1 - sum( lr * sum(delta_b1,1) / size(delta_b1,1) ) - momen * grad_b1_01 - momen ** 2 * grad_b1_02 - momen ** 3 * grad_b1_03 - momen ** 4 * grad_b1_04
        w1 = w1 - ( lr * sum(delta_w1,1) / size(delta_w1,1) ) - momen * grad_w1_01 - momen ** 2 * grad_w1_02 - momen ** 3 * grad_w1_03 - momen ** 4 * grad_w1_04
        if (nmomen.gt.4) then 
          nmomen = 1
        endif
        if (nmomen.eq.1) then
          grad_b3_01 = sum( lr * sum(delta_b3,1) / size(delta_b3,1) )
          grad_w3_01 = ( lr * sum(delta_w3,1) / size(delta_w3,1) )
          grad_b2_01 = sum( lr * sum(delta_b2,1) / size(delta_b2,1) )
          grad_w2_01 = ( lr * sum(delta_w2,1) / size(delta_w2,1) )
          grad_b1_01 = sum( lr * sum(delta_b1,1) / size(delta_b1,1) )
          grad_w1_01 = ( lr * sum(delta_w1,1) / size(delta_w1,1) )
        endif
        if (nmomen.eq.2) then
          grad_b3_02 = sum( lr * sum(delta_b3,1) / size(delta_b3,1) )
          grad_w3_02 = ( lr * sum(delta_w3,1) / size(delta_w3,1) )
          grad_b2_02 = sum( lr * sum(delta_b2,1) / size(delta_b2,1) )
          grad_w2_02 = ( lr * sum(delta_w2,1) / size(delta_w2,1) )
          grad_b1_02 = sum( lr * sum(delta_b1,1) / size(delta_b1,1) )
          grad_w1_02 = ( lr * sum(delta_w1,1) / size(delta_w1,1) )
        endif
        if (nmomen.eq.3) then
          grad_b3_03 = sum( lr * sum(delta_b3,1) / size(delta_b3,1) )
          grad_w3_03 = ( lr * sum(delta_w3,1) / size(delta_w3,1) )
          grad_b2_03 = sum( lr * sum(delta_b2,1) / size(delta_b2,1) )
          grad_w2_03 = ( lr * sum(delta_w2,1) / size(delta_w2,1) )
          grad_b1_03 = sum( lr * sum(delta_b1,1) / size(delta_b1,1) )
          grad_w1_03 = ( lr * sum(delta_w1,1) / size(delta_w1,1) )
        endif
        if (nmomen.eq.4) then
          grad_b3_04 = sum( lr * sum(delta_b3,1) / size(delta_b3,1) )
          grad_w3_04 = ( lr * sum(delta_w3,1) / size(delta_w3,1) )
          grad_b2_04 = sum( lr * sum(delta_b2,1) / size(delta_b2,1) )
          grad_w2_04 = ( lr * sum(delta_w2,1) / size(delta_w2,1) )
          grad_b1_04 = sum( lr * sum(delta_b1,1) / size(delta_b1,1) )
          grad_w1_04 = ( lr * sum(delta_w1,1) / size(delta_w1,1) )
        endif
        nmomen = nmomen + 1
      enddo
      Loss(epoch) = sqrt( sum(Ltmp)/epoch_count ) 
      epoch_b3(epoch) = b3
      epoch_b2(epoch) = b2
      epoch_b1(epoch) = b1
      epoch_w3(epoch,:,:) = w3
      epoch_w2(epoch,:,:) = w2
      epoch_w1(epoch,:,:) = w1 
      if ( mod(epoch,100).eq. 0 ) then 
        write(*,*)'Root Mean Square Training Error in Epoch ',epoch,' is = ',Loss(epoch)
      endif
    enddo
    idx = minloc( Loss, dim = 1 ) 
    if ( Loss(idx).gt.0.1 ) go to 101
    write(*,*)'The lowest error is in epoch no. ',idx,' at ',Loss(idx)
    best_b3 = epoch_b3(idx)
    best_w3 = epoch_w3(idx,:,:)
    best_b2 = epoch_b2(idx)
    best_w2 = epoch_w2(idx,:,:)
    best_b1 = epoch_b1(idx)
    best_w1 = epoch_w1(idx,:,:)
    datatest = datadir // 'test' //cha// '.txt'
    call count_rows( datatest, ntest ) 
    allocate( Xtest1(ntest), Xtest2(ntest), Ytest(ntest), Yhat(ntest), Ltest(ntest) ) 
    call read_train (datatest,Xtest1,Xtest2,Ytest)
    allocate( Xtest1Norm( ntest ), Xtest2Norm( ntest ), YtestNorm( ntest ) ) 
    do i = 1, ntest 
      Xtest1Norm(i) = ( Xtest1(i) - mu_Xtrain1 ) / sigma_Xtrain1
      Xtest2Norm(i) = ( Xtest2(i) - mu_Xtrain2 ) / sigma_Xtrain2 
      YtestNorm(i) = ( Ytest(i) - mu_Ytrain ) / sigma_Ytrain
    enddo
    open(10,file=trim(cha)//'.data.txt',status='replace')
    do i = 1, ntest 
      x(1,1) = Xtest1Norm(i)
      x(1,2) = Xtest2Norm(i)
      y(1,1) = YtestNorm(i)
      call DenseLayer (  x,  best_w1, best_b1, o1c, do1c_do1b, do1b_do1a, do1b_db1, do1a_dw1, do1a_dx )
      call DenseLayer ( o1c, best_w2, best_b2, o2c, do2c_do2b, do2b_do2a, do2b_db2, do2a_dw2, do2a_do1c )
      call OutputLayer( o2c, best_w3, best_b3, o3b, do3b_db3,  do3b_do3a, do3a_dw3, do3a_do2c )
      Ltest(i) = sum( ( (o3b*sigma_Ytrain+mu_Ytrain) - (Ytest(i)) ) ** 2 )
      write(10,*)(Xtest1(i)),(Xtest2(i)),(Ytest(i)),(o3b*sigma_Ytrain+mu_Ytrain)
    enddo
    close(10)
    open(10,file=trim(cha)//'.rmse.txt',status='replace') 
      write(10,*)'RMSE Test Dataset = ',sqrt( sum(Ltest) / size(Ltest) )
    close(10)
end program 

