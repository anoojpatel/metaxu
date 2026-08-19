#include <stdio.h>
int main(void){long acc=0;for(long i=0;i<40000000;i++)acc+=(i*i)%1000;printf("%ld\n",acc);return 0;}
