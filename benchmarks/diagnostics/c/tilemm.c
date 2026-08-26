/* C twin of tilemm.mx: same 20 reps of 128x128x128 int matmul, plain
 * loops (what the tile abstraction races against on CPU). */
#include <stdio.h>
#include <stdlib.h>
int main(void){
    long n=128;
    long *a=malloc(n*n*8),*b=malloc(n*n*8),*c=malloc(n*n*8);
    for(long i=0;i<n*n;i++){a[i]=i%7;b[i]=i%5;c[i]=0;}
    for(int rep=0;rep<20;rep++)
        for(long r=0;r<n;r++)for(long j=0;j<n;j++){
            long acc=0;
            for(long k=0;k<n;k++)acc+=a[r*n+k]*b[k*n+j];
            c[r*n+j]=acc;
        }
    printf("%ld\n%ld\n",c[0],c[n*n-1]);
    return 0;
}
