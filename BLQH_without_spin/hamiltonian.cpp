#include"hamiltonian.h"
static std::mutex mutex_update;
hamil::hamil() {}

double hamil::Coulomb_interaction(int alpha,int q_x, int q_y) {
    double q = sqrt(q_x * q_x / (lx * lx) + q_y * q_y / (ly * ly)) * 2.0 * M_PI;
    if(alpha ==0)
        return 2.0*M_PI/q*exp(-q*q/2.0)*pow(1.0-exp(-q*q/2.0),nLL*2);
    else
        return 2.0*M_PI/q*exp(-q*q/2.0-q*d)*pow(1.0-exp(-q*q/2.0),nLL*2);
}

void hamil::init_Coulomb_matrix(double theta_x) {
    Ec=-2.0;
    for(int i=0; i<nphi; i++)
        for(int j=0; j<nphi; j++)
            if(!(i==0 &&j==0))
                //Ec+=Integrate_ExpInt((i*i*lx/ly+j*j*ly/lx)*M_PI);
                Ec+=erfc(sqrt(M_PI*(i*i*lx/ly+j*j*ly/lx)));
    Ec/=sqrt(lx*ly);
    // classical Coulomb energy for interwell interaction
    Coulomb_matrix.assign(2 * nphi*nphi, 0);
    for(int alpha = 0; alpha < 2; alpha++)
        // n=j_1, m=_j3
        for(int s = 0; s < nphi; s++)
            for(int q_y = 0; q_y < nphi; q_y++) {
                double V=0;
                for(int q_x = -nphi; q_x <=nphi; q_x++)
                    if(!(q_x==0 && q_y==0))
                        V+=2.0*Coulomb_interaction(alpha,q_x,q_y)*cos(2.0*M_PI*s*q_x/nphi)/(2.0*lx*ly);

                if(alpha==1) {
                    V=0;
                    for(int q_x = -10*nphi/(d+0.01); q_x <10*nphi/(d+0.01); q_x++)
                        if(!(q_x==0 &&q_y==0))
                            V+=2.0*Coulomb_interaction(alpha,q_x,q_y)*cos((2.0*M_PI*s-2*theta_x)*q_x/nphi)/(2.0*lx*ly);
                            //V+=2.0*Coulomb_interaction(alpha,q_x,q_y)*cos(2.0*M_PI*s*q_x/nphi)/(2.0*lx*ly);
                }
                // Coulomb matrix elements in Landau gauge
                Coulomb_matrix[alpha*nphi*nphi+s*nphi+q_y]=V;
            }
    // store FT coefficients
    int kx=sector.K;
    FT.assign(nphi*nphi,0);
    for(int kl=0; kl<nphi; kl++)
        for(int kr=0; kr<nphi; kr++)
            FT[kl*nphi+kr]=complex<double>(cos(2.0*M_PI*(kl-kr)*kx/nphi),sin(2.0*M_PI*(kl-kr)*kx/nphi));
}

inline void hamil::peer_set_hamil(double Delta_SAS,double Delta_V,double theta_x, double theta_y,int id, long nbatch,long nrange) {
    int kx=sector.K;
    unsigned long lbasis,rbasis,rbasis_0,mask,mask_t,occ_t,b;
    int n,m,s,t,nt,mt,sign,signl,signr;
    int ql,qr,Dl,Dr,D;

    long i,j,k,l;
    vector<complex<double> > matrix_elements;

    for(int _i = 0; _i < nrange; _i++) {
        // i for each thread
        i=_i+id*nbatch;
        matrix_elements.assign(nHilbert,0);

        Dl=(kx<0?1:sector.basis_C[i]);
        for(ql=0; ql<Dl; ql++) {
            signl=1;
            lbasis=(ql==0?sector.id[i]:sector.translate(sector.id[i],ql,signl));
            //down-layer
            for(n=0; n<nphi-1; n++)
                for(m = n+1; m < nphi; m++) {
                    mask = (1UL << n) + (1UL << m);
                    // consider the upper-layer two electrons
                    // looking up the corresponding basis in id_up
                    // if there're two electrons on n and m;
                    if((lbasis &mask) == mask ) {
                        // b is the rest electon positions
                        b = lbasis ^ mask;
                        // mt=j3, nt=j4
                        // perform translation along x-direction (q_y), positive q_y
                        for(t = -nphi+1; t < nphi; t++) {
                            if(n + t >=nphi)
                                nt = n + t - nphi;
                            else if (n+t <0)
                                nt = n + t +nphi;
                            else
                                nt = n + t;
                            if(m - t <0)
                                mt = m - t + nphi;
                            else if (m - t >=nphi)
                                mt = m - t -nphi;
                            else
                                mt = m - t;

			    // s=j3-j1
                            s=abs(mt-n);
                            // the translated two electrons indices
                            mask_t = (1UL << nt) + (1UL << mt);
                            // occupation of electons on the translated position
                            occ_t = mask_t & b;
                            rbasis_0=mask_t+b;
                            // if there're no electon on the translated position
                            // which is a valid translation, can be applied
                            if(occ_t == 0 && nt!=mt) {
                                // determine the right side size of the translation
                                Dr=nphi;
                                for(D=1; D<nphi; D++)
                                    if(sector.translate(rbasis_0,D,signr)==rbasis_0) {
                                        Dr=D;
                                        break;
                                    }
                                // if parameter kx<0, do not perform basis translation
                                Dr=(kx<0?1:Dr);
                                for(qr=0; qr<Dr; qr++) {
                                    signr=1;
                                    rbasis=(qr==0?rbasis_0:sector.inv_translate(rbasis_0,qr,signr));
                                    if(sector.basis_set.find(rbasis) != sector.basis_set.end()) {
                                        j = sector.basis_set[rbasis];
                                        sign=sector.get_sign(lbasis,n,m,nt,mt,t)*signl*signr;
					// down-layer y-direction twisted phase
					//complex<double> FT_twisted=FT[ql*nphi+qr]*complex<double>(cos(theta_y*t/nphi),sin(theta_y*t/nphi));
                                        //matrix_elements[j]+=FT_twisted*Coulomb_matrix[s*nphi+abs(t)]/(sqrt(Dl*Dr)*sign);
                                        matrix_elements[j]+=Coulomb_matrix[s*nphi+abs(t)]*sign*FT[ql*nphi+qr]/sqrt(Dl*Dr);
                                    }
                                }
                            }
			}
                    }
                }
            // upper-layer
            for( n=nphi; n<2*nphi-1; n++)
                for( m = n+1; m < 2*nphi; m++) {
                    mask = (1UL << n) + (1UL << m);
                    // consider the lower-layer two electrons
                    // if there're two electrons on n and m;
                    if((lbasis &mask) == mask && n!=m) {
                        // p is the rest electon positions
                        b = lbasis ^ mask;
                        // perform translation in x-direction, negative q_y
                        for(t = -nphi+1; t < nphi; t++) {
                            if(n + t >=2*nphi)
                                nt = n + t - nphi;
                            else if (n+t <nphi)
                                nt = n + t +nphi;
                            else
                                nt = n + t;
                            if(m - t <nphi)
                                mt = m - t + nphi;
                            else if (m - t >=2*nphi)
                                mt = m - t -nphi;
                            else
                                mt = m - t;
                            s=abs(mt-n);
                            // the translated two electrons indices
                            mask_t = (1UL << nt) + (1UL << mt);
                            // occupation of electons on the translated position
                            occ_t = mask_t & b;
                            // if there're no electon on the translated position
                            // which is a valid translation, can be applied
                            rbasis_0=mask_t+b;
                            if(occ_t == 0 && nt!=mt) {
                                // determine the right side size of the translation
                                Dr=nphi;
                                for(D=1; D<nphi; D++)
                                    if(sector.translate(rbasis_0,D,signr)==rbasis_0) {
                                        Dr=D;
                                        break;
                                    }

                                // if parameter kx<0, do not perform basis translation
                                Dr=(kx<0?1:Dr);
                                for(qr=0; qr<Dr; qr++) {
                                    signr=1;
                                    rbasis=(qr==0?rbasis_0:sector.inv_translate(rbasis_0,qr,signr));
                                    //rbasis=sector.inv_translate(_rbasis,ql,_signr);
                                    if(sector.basis_set.find(rbasis) != sector.basis_set.end()) {
                                        j = sector.basis_set[rbasis];
                                        sign=sector.get_sign(lbasis,n,m,nt,mt,t)*signl*signr;
					// upper-layer x-direction twisted phase
					//double V=0;
                			//for(int q_x = -nphi; q_x <=nphi; q_x++)
                    			//   if(!(q_x==0 && abs(t)==0))
                        		//	V+=2.0*Coulomb_interaction(0,q_x,abs(t))*cos((2.0*M_PI*s-theta_x)*q_x/nphi)/(2.0*lx*ly);
                                        //matrix_elements[j]+=FT[ql*nphi+qr]*V/(sqrt(Dl*Dr)*sign);
                                        matrix_elements[j]+=Coulomb_matrix[s*nphi+abs(t)]*sign*FT[ql*nphi+qr]/sqrt(Dl*Dr);
                                    }
                                }
                            }
                        }
                    }
                }

            for(n=0; n<nphi; n++)
                for(m = nphi; m < 2*nphi; m++) {
                    mask = (1UL << n) + (1UL << m);
                    // if there is one electron at site n in upper-layer
                    // and one electron at site m in lower-layer
                    if((lbasis &mask) == mask) {
                        // b is the rest electon positions for upper-layer electrons
                        b = lbasis ^ mask;
                        // perform translation along x-direction
                        for(t = -nphi+1; t < nphi ; t++) {
                            if(n + t>=nphi)
                                nt = n + t - nphi;
                            else if (n+t <0)
                                nt = n + t +nphi;
                            else
                                nt = n + t;
                            if(m - t <nphi)
                                mt = m - t + nphi;
                            else if (m - t >=2*nphi)
                                mt = m - t -nphi;
                            else
                                mt = m - t;
                            s=abs(mt-nphi-n);
                            // the translated electron index
                            mask_t = (1UL << nt)+(1UL <<mt);
                            // occupation of electons on the translated position
                            occ_t = mask_t & b;
                            // if there're no electon on the translated position
                            // which is a valid translation, can be applied
                            // the translated indices
                            rbasis_0=mask_t+b;
                            if(occ_t == 0 ) {
                                // determine the right side size of the translation
                                Dr=nphi;
                                for(D=1; D<nphi; D++)
                                    if(sector.translate(rbasis_0,D,signr)==rbasis_0) {
                                        Dr=D;
                                        break;
                                    }
                                // if parameter kx<0, do not perform basis translation
                                Dr=(kx<0?1:Dr);
                                for(qr=0; qr<Dr; qr++) {
                                    signr=1;
                                    rbasis=(qr==0?rbasis_0:sector.inv_translate(rbasis_0,qr,signr));
                                    if(sector.basis_set.find(rbasis) != sector.basis_set.end()) {
                                        j = sector.basis_set[rbasis];
                                        sign=sector.get_sign(lbasis,n,m,nt,mt,t)*signl*signr;
					complex<double> FT_twisted=FT[ql*nphi+qr]*complex<double>(cos(2*theta_y*t/nphi),sin(2*theta_y*t/nphi));
					matrix_elements[j]+=FT_twisted*Coulomb_matrix[nphi*nphi+s*nphi+abs(t)]/(sqrt(Dl*Dr)*sign);
                                        //matrix_elements[j]+=Coulomb_matrix[nphi*nphi+s*nphi+abs(t)]*sign*FT[ql*nphi+qr]/sqrt(Dl*Dr);
                                    }
                                }
                            }
                        }
                    }
                }
	    // interlayer tunneling 
            if(Delta_SAS>0)
            for(n=0; n<nphi; n++){
	      nt=(n<nphi?n+nphi:n-nphi);
              mask = (1UL << n)+(1UL <<nt);
	      unsigned long Kn=mask & lbasis;
              if(Kn!=mask && Kn!=0){
	      unsigned long Ln=Kn ^ mask; 
	      rbasis_0=lbasis-Kn+Ln;
              // determine the right side size of the translation
              Dr=nphi;
              for(D=1; D<nphi; D++)
                if(sector.translate(rbasis_0,D,signr)==rbasis_0) {
                  Dr=D;
                  break;
                }
                // if parameter kx<0, do not perform basis translation
                Dr=(kx<0?1:Dr);
                for(qr=0; qr<Dr; qr++) {
                  signr=1;
                  rbasis=(qr==0?rbasis_0:sector.inv_translate(rbasis_0,qr,signr));
                    if(sector.basis_set.find(rbasis) != sector.basis_set.end()) {
                    j = sector.basis_set[rbasis];
                    sign=sector.get_sign(lbasis,n,nt)*signl*signr;
                    //cout<<"i:=  "<<bitset<6>(lbasis).to_string()<<"    j:="<<bitset<6>(rbasis).to_string()<<"    sign:="<<sign<<endl;
                    matrix_elements[j]-=0.5*Delta_SAS*sign*FT[ql*nphi+qr]/sqrt(Dl*Dr);
                  }
                }
	      }
            }
        }
	// bias-voltage
        if(Delta_V>0)
         for(n=0; n<2*nphi; n++){
             mask = 1 << n;
	   // if there's an electron on site n
            if((sector.id[i] &mask) == mask){
	   if(n<nphi)
              matrix_elements[i]+=0.5*Delta_V;
	    else
	      matrix_elements[i]-=0.5*Delta_V;
	   }
        }


        matrix_elements[i]+=Ec*(sector.nel_up+sector.nel_down);
        // charging energy
	matrix_elements[i]+=-d*(sector.get_nel_upper(i))*(sector.nel-sector.get_nel_upper(i))/sector.nphi;

        mutex_update.lock();
        for(k=0; k<nHilbert; k++)
	    hamiltonian[i*nHilbert+k]=matrix_elements[k]; 
        mutex_update.unlock();
    }
    matrix_elements.clear();
}

void hamil::set_hamil(double _lx, double _ly, long _nphi, long _nLL,double _d, double Delta_SAS,double Delta_V,double theta_x, double theta_y,int nthread) {
    d = _d;
    lx = _lx;
    ly = _ly;
    nphi = _nphi;
    nLL = _nLL;
    init_Coulomb_matrix(theta_x);
    nHilbert = sector.nbasis;
    hamiltonian.assign(nHilbert*nHilbert,0);
    std::vector<std::thread> threads;
    long  nbatch=nHilbert/nthread;
    long nresidual=nHilbert%nthread;
    for(int id = 0; id < nthread; id++)
        threads.push_back(std::thread(&hamil::peer_set_hamil,this,Delta_SAS,Delta_V,theta_x,theta_y,id,nbatch,nbatch));
    for(auto &th:threads)
        if(th.joinable())
            th.join();
    if(nresidual!=0)
        peer_set_hamil(Delta_SAS,Delta_V,theta_x,theta_y,nthread,nbatch,nresidual);
}

hamil::~hamil() {}

const hamil & hamil::operator =(const hamil & _gs_hconfig) {
    if(this != &_gs_hconfig) {
        nHilbert = _gs_hconfig.nHilbert;
        d = _gs_hconfig.d;
        nphi = _gs_hconfig.nphi;
        hamiltonian.assign(_gs_hconfig.hamiltonian.begin(),_gs_hconfig.hamiltonian.end());
        eigenvalues.assign(_gs_hconfig.eigenvalues.begin(), _gs_hconfig.eigenvalues.end());
        psi_0.assign(_gs_hconfig.psi_0.begin(), _gs_hconfig.psi_0.end());
        psi_n0.assign(_gs_hconfig.psi_n0.begin(), _gs_hconfig.psi_n0.end());
    }
    return *this;
}

double hamil::ground_state_energy() {
    if(psi_0.size() == 0) return 0;
    complex<double> E_gs = 0;
    vector< complex<double> > psi_t;
    psi_t.assign(nHilbert,0);
    for(int i=0; i<nHilbert; i++)
        for(int j=0; j<nHilbert; j++)
            psi_t[i]+=hamiltonian[i*nHilbert+j]*psi_0[j];
    for(int i = 0; i < nHilbert; i++)
        E_gs += conj(psi_t[i]) * psi_0[i];
    psi_t.clear();
    return E_gs.real();
}

double hamil::pseudospin_Sz(){
     double Nel_upper=0;
     unsigned long mask;
     int sign,q,D;
     mask=(1UL<<nphi)-1;
     for(int i=0;i<nHilbert;i++)
       Nel_upper+=sector.popcount_table[mask&sector.id[i]]*std::norm(psi_0[i]);
     
     return (2.0*Nel_upper-sector.nel)/(2.0*sector.nel);
}

double hamil::pseudospin_Sx(){
       unsigned long mask,mask_t,b,occ_t,lbasis,rbasis,rbasis_0;
       long i,j;
       int n,nt,sign,signl,signr;
       int ql,qr,Dl,Dr,D;
       complex<double> Sx_mean=0;
       for(i=0;i<nHilbert;i++){
        Dl=(sector.K<0?1:sector.basis_C[i]);
        for(ql=0; ql<Dl; ql++) {
         signl=1;
         lbasis=(ql==0?sector.id[i]:sector.translate(sector.id[i],ql,signl));
	 //lbasis=sector.id[i];
	 // interlayer tunneling 
         for(n=0; n<nphi; n++){
	      nt=(n<nphi?n+nphi:n-nphi);
              mask = (1UL << n)+(1UL <<nt);
	      unsigned long Kn=mask & lbasis;
              if(Kn!=mask && Kn!=0){
	      unsigned long Ln=Kn ^ mask; 
	      rbasis_0=lbasis-Kn+Ln;
              // determine the right side size of the translation
	      
              Dr=nphi;
              for(D=1; D<nphi; D++)
                if(sector.translate(rbasis_0,D,signr)==rbasis_0) {
                  Dr=D;
                  break;
                }
                // if parameter kx<0, do not perform basis translation
                Dr=(sector.K<0?1:Dr);
                for(qr=0; qr<Dr; qr++) {
                   signr=1;
                   rbasis=(qr==0?rbasis_0:sector.inv_translate(rbasis_0,qr,signr));
                   if(sector.basis_set.find(rbasis) != sector.basis_set.end()) {
                   j = sector.basis_set[rbasis];
                   sign=sector.get_sign(lbasis,n,nt)*signl*signr;
                   //sign=sector.get_sign(lbasis,n,nt);
		   Sx_mean+=conj(psi_0[i])*psi_0[j]*FT[ql*nphi+qr]*double(0.5*sign/sqrt(Dl*Dr));
                  }
                }
	      }
            }
       }
    }
    return abs(Sx_mean)/sector.nel;
}

void hamil::diag() {
    int i;
    complex<double> *h = new complex<double>[nHilbert * nHilbert];
    double *en = new double[nHilbert];
    memset(h, 0, sizeof( complex<double>)*nHilbert * nHilbert);
    for(i = 0; i <nHilbert*nHilbert; i++)
        h[i]=hamiltonian[i];
    diag_zheevd(h, en, nHilbert);
    psi_0.assign(nHilbert, 0);
    psi_1.assign(nHilbert, 0);
    psi_n0.assign(nHilbert, 0);
    eigenvalues.assign(nHilbert, 0);
    for(i = 0; i < nHilbert; i++) {
        eigenvalues[i] = en[i];
        psi_0[i] = h[i];
        psi_1[i] = h[i+nHilbert];
        psi_n0[i] = h[i * nHilbert];
    }
    delete h, en;
}

void hamil::print_hamil(int range) {
    int i, j, count;
    if(range>nHilbert)
        range=nHilbert;
    for(i = 0; i < range; i++) {
        if(i == 0)
            cout <<setw(2)<< "[[";
        else cout <<setw(2)<< " [";
        // count is the No. of nonzero elements in the row
        for(j=0; j<range; j++)
            cout<<setw(5)<<setprecision(2)<<hamiltonian[i*nHilbert+j]<<", ";
        if(i == range - 1)
            cout << ",...]]" << endl;
        else cout << ",...]" << endl;
    }
}
void hamil::print_eigen(int range) {
    if(range>=nHilbert)
        range=nHilbert;
    std::cout << "Eigenvalues:=[ ";
    for(int i = 0; i < range; i++)
        if(i != range - 1)
            std::cout << eigenvalues[i] << ", ";
        else
            std::cout << eigenvalues[i] << " , ...]" << std::endl;
}
