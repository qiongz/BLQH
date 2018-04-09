#include"basis.h"
using namespace std;

basis::basis() {
}

basis::basis(long _nphi,long _nel_up, long _nel_down):nphi(_nphi),nel_up(_nel_up),nel_down(_nel_down) {
  K_up=-1;
  K_down=-1;
}

basis::basis(long _nphi,long _nel_up, long _nel_down, long _K_up,long _K_down):nphi(_nphi),nel_up(_nel_up),nel_down(_nel_down),K_up(_K_up),K_down(_K_down){
}

const basis & basis::operator =(const basis & _basis) {
    if(this !=&_basis) {
        nphi=_basis.nphi;
        nel_up=_basis.nel_up;
        nel_down=_basis.nel_down;
        nbasis_up=_basis.nbasis_up;
        nbasis_down=_basis.nbasis_down;
        basis_up=_basis.basis_up;
        basis_down=_basis.basis_down;
        id_up=_basis.id_up;
        id_down=_basis.id_down;
        K_up=_basis.K_up;
        K_down=_basis.K_down;
    }
    return *this;
}

basis::~basis() {}

long basis::factorial(long N, long m) {
    unsigned long num,denum;
    long i;
    num=1;
    for(i=N-m+1; i<=N; i++)
        num*=i;
    denum=1;
    for(i=1; i<=m; i++)
        denum*=i;
    return num/denum;
}

void basis::init() {
    long i,config_init;
    nbasis_up=factorial(nphi,nel_up);
    nbasis_down=factorial(nphi,nel_down);
    config_init=0;
    for(i=0; i<nel_up; i++)
        config_init+=(1<<i);
    generate_up(config_init);
    std::map<long,long>::iterator it;
    for(it=basis_up.begin(); it!=basis_up.end(); it++){
        if(K_up<0)
           id_up.push_back(it->first);
        else{
          long sum_j=0;
          long c=it->first;
          for(i=0;i<nphi;i++)
             if((c>>i)%2==1)
               sum_j+=i;
           if(sum_j%nphi==K_up)
          id_up.push_back(it->first);
         }
    }
    sort(id_up.begin(),id_up.end());
    basis_up.clear();
    for(i=0; i<nbasis_up; i++)
        basis_up[id_up[i]]=i;
    nbasis_up=basis_up.size();

    config_init=0;
    for(i=0; i<nel_down; i++)
        config_init+=(1<<i);
    generate_down(config_init);
    for(it=basis_down.begin(); it!=basis_down.end(); it++){
        if(K_down<0)
           id_down.push_back(it->first);
        else{
         long sum_j=0;
          long c=it->first;
          for(i=0;i<nphi;i++)
             if((c>>i)%2==1)
               sum_j+=i;
           if(sum_j%nphi==K_down)
          id_down.push_back(it->first);
         }
    }
    sort(id_down.begin(),id_down.end());
    basis_down.clear();
    for(i=0; i<nbasis_down; i++)
        basis_down[id_down[i]]=i;
    nbasis_down=basis_down.size();
}

void basis::init(long _nphi, long _nel_up, long _nel_down){
   nphi=_nphi;
   nel_up=_nel_up;
   nel_down=_nel_down;
   init();
}

long basis::interlayer_hopping(long i,long n,long m) {
    long mask,K,L,b;
    if(m<0) m+=nphi;
    else if (m>=nphi) m-=nphi;

    mask=(1<<n)+(1<<m);
    K=mask&id_up[i];
    L=K^mask;

    if(L!=0 && L!=mask) {
        b=id_up[i]-K+L;
        if(basis_up.find(b)!=basis_up.end())
            return basis_up[b];
        else
            return basis_up[id_up[i]];
    }
    else
        return basis_up[id_up[i]];
}


long basis::onsite_potential(long i,long j,long n) {
    long mask,bu,bd;
    mask=(1<<n);
    bu=id_up[i]&mask;
    bd=id_down[j]&mask;
    if(bu==mask && bd==mask)
        return 1;
    else
        return 0;
}

int basis::get_signu(long i,long n, long m, long nt, long mt){
     long b,k,kl,kr,mask,mask_k,nsign;
     mask=(1<<n)+(1<<m);
      // get the rest electrons
     b=id_up[i]^mask;
     // if there're no crossing between two electrons
     nsign=0;
        kl=nt<n?nt:n;
        kr=nt<n?n:nt;
        for(k=kl+1;k<kr;k++){
          mask_k=(1<<k);
          if((b&mask_k)==mask_k)
             nsign++;
        }
        kl=mt<m?mt:m;
        kr=mt<m?m:mt;
        for(k=kl+1;k<kr;k++){
          mask_k=(1<<k);
          if((b&mask_k)==mask_k)
             nsign++;
        }
        // if there're crossings between two electrons
        if(nt>mt && m>n || mt>nt && m<n)
          nsign++;

     return pow(-1,nsign);
}

int basis::get_signd(long j,long n, long m, long nt, long mt){
     long b,k,kl,kr,mask,mask_k,nsign;
     mask=(1<<n)+(1<<m);
      // get the rest electrons
     b=id_down[j]^mask;
     // if there're no crossing between two electrons
     nsign=0;
        kl=nt<n?nt:n;
        kr=nt<n?n:nt;
        for(k=kl+1;k<kr;k++){
          mask_k=(1<<k);
          if((b&mask_k)==mask_k)
             nsign++;
        }
        kl=mt<m?mt:m;
        kr=mt<m?m:mt;
        for(k=kl+1;k<kr;k++){
          mask_k=(1<<k);
          if((b&mask_k)==mask_k)
             nsign++;
        }
        // if there're crossings between two electrons
        if(nt>mt && m>n || mt>nt && m<n)
          nsign++;

     return pow(-1,nsign);
}

int basis::get_signud(long i,long j,long n, long m, long nt, long mt){
     long b,p,k,kl,kr,mask,mask_k,nsign;
     mask=(1<<n)+(1<<m);
      // get the rest electrons
     b=id_up[i]^mask;
     p=id_up[j]^mask;
     // if there're no crossing between two electrons
     nsign=0;
        kl=nt<n?nt:n;
        kr=nt<n?n:nt;
        for(k=kl+1;k<kr;k++){
          mask_k=(1<<k);
          if((b&mask_k)==mask_k)
             nsign++;
        }
        kl=mt<m?mt:m;
        kr=mt<m?m:mt;
        for(k=kl+1;k<kr;k++){
          mask_k=(1<<k);
          if((p&mask_k)==mask_k)
             nsign++;
        }
        // if there're crossings between two electrons
        if(nt>mt && m>n || mt>nt && m<n)
          nsign++;

     return pow(-1,nsign);
}

long basis::creation(long s,long n)
{
    long mask,bu;
    mask=(1<<n);
    bu=s&mask;
    // there's no electron on site n
    // which means s_n=0
    if(bu!=mask)
        return s+mask;
    // there's already electron on site n
    else
        return s;
}

long basis::annihilation(long s,long n)
{
    long mask,bu;
    mask=(1<<n);
    bu=s&mask;
    if(bu==mask)
        return s-mask;
    else
        return s;
}


void basis::generate_up(long a) {
    long mask,K,L,b,j;
    #if __cplusplus > 199711L
    basis_up.emplace(a,a);
    #else
    basis_up[a]=a;
    #endif
    for(long i=0; i<nphi; i++) {
        j=(i+1>=nphi)?i+1-nphi:i+1;
        mask=(1<<i)+(1<<j);
        K=mask&a;
        L=K^mask;
        if(L!=0 && L!=mask) {
            b=a-K+L;
            if(basis_up.find(b)==basis_up.end())
                generate_up(b);
            if(basis_up.size()==nbasis_up)
                return;
        }
    }
    return;
}

void basis::generate_down(long a) {
    long mask,K,L,b,j;
    #if __cplusplus > 199711L
    basis_down.emplace(a,a);
    #else
    basis_down[a]=a;
    #endif
    for(long i=0; i<nphi-1; i++) {
        j=(i+1>=nphi)?i+1-nphi:i+1;
        mask=(1<<i)+(1<<j);
        K=mask&a;
        L=K^mask;
        if(L!=0 && L!=mask) {
            b=a-K+L;
            if(basis_down.find(b)==basis_down.end())
                generate_down(b);
            if(basis_down.size()==nbasis_down)
                return;
        }
    }
    return;
}

void basis::prlong() {
    std::map<long,long>::iterator it;
    cout<<"---------------------------------------"<<endl;
    cout<<"spin-up electrons:"<<endl;
    cout<<"---------------------------------------"<<endl;
    for(it=basis_up.begin(); it!=basis_up.end(); it++)
        cout<<bitset<20>(it->first).to_string()<<" "<<setw(6)<<it->first<<" "<<it->second<<endl;
    cout<<"---------------------------------------"<<endl;
    cout<<"spin-down electrons:"<<endl;
    cout<<"---------------------------------------"<<endl;
    for(it=basis_down.begin(); it!=basis_down.end(); it++)
        cout<<bitset<20>(it->first).to_string()<<" "<<setw(6)<<it->first<<" "<<it->second<<endl;
    cout<<"---------------------------------------"<<endl;
    cout<<"No. basis for spin-up electrons: "<<setw(6)<<nbasis_up<<endl;
    cout<<"No. basis for spin-down electrons: "<<setw(6)<<nbasis_down<<endl;
    /*
    cout<<"---------------------------------------"<<endl;
    cout<<"Lin's Table:"<<endl;
    cout<<"---------------------------------------"<<endl;
    cout<<"spin-up electrons:"<<endl;
    cout<<"---------------------------------------"<<endl;
    for(auto &x: id_up)
       cout<<x<<endl;
    cout<<"---------------------------------------"<<endl;
    cout<<"spin-down electrons:"<<endl;
    cout<<"---------------------------------------"<<endl;
    for(auto &x: id_down)
       cout<<x<<endl;
    cout<<"---------------------------------------"<<endl;
    */
}
