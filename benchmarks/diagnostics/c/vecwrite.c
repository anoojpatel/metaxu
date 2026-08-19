#include <stdio.h>
#include <stdlib.h>
int main(void){long n=5000000;long*v=calloc(n,8);
for(int r=0;r<4;r++){for(long j=0;j<n;j++)v[j]=j;}
printf("%ld\n",v[n-1]);return 0;}
