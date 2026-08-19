#include <stdio.h>
typedef struct{long tag;long k;}Op;
static long eval(Op op,long x){switch(op.tag){case 0:return x+op.k;case 1:return x*op.k;default:return x;}}
int main(void){long acc=1;
for(long i=0;i<20000000;i++){Op op; long m=i%3; if(m==0){op.tag=0;op.k=2;}else if(m==1){op.tag=1;op.k=3;}else{op.tag=2;op.k=0;}
acc=(acc+eval(op,i))%1000003;}
printf("%ld\n",acc);return 0;}
