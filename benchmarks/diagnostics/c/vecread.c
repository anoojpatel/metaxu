#include <stdio.h>
#include <stdlib.h>
int main(void){long n=5000000;long*v=malloc(n*8);for(long i=0;i<n;i++)v[i]=i;
long acc=0;for(int r=0;r<4;r++){for(long j=0;j<n;j++)acc+=v[j];}
printf("%ld\n",acc);return 0;}
