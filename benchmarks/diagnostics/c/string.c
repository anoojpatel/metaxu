#include <stdio.h>
#include <string.h>
int main(void){long n=0;char buf[32];
for(long i=0;i<200000;i++){snprintf(buf,32,"%ld",i);n+=strlen(buf);}
printf("%ld\n",n);return 0;}
