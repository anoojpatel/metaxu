#include <stdio.h>
static long inc(long v){return v+1;}
static long apply3(long(*f)(long),long x){return f(f(f(x)));}
int main(void){long acc=0;for(long i=0;i<10000000;i++)acc+=apply3(inc,i)%7;printf("%ld\n",acc);return 0;}
