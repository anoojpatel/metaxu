#include <stdio.h>
typedef struct{long a,b,c;}P;
static P mix(P p){P r={p.b+1,p.c*2,p.a+p.b};return r;}
int main(void){P p={1,2,3};
for(long i=0;i<40000000;i++){p=mix(p);P q={p.a%1000,p.b%1000,p.c%1000};p=q;}
printf("%ld\n",p.a+p.b+p.c);return 0;}
