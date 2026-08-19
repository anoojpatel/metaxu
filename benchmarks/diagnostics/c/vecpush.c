#include <stdio.h>
#include <stdlib.h>
typedef struct{long*d;long len,cap;}vec;
static void push(vec*v,long x){if(v->len==v->cap){v->cap=v->cap?v->cap*2:8;v->d=realloc(v->d,v->cap*8);}v->d[v->len++]=x;}
int main(void){long total=0;
for(int r=0;r<4;r++){vec v={0,0,0};for(long i=0;i<5000000;i++)push(&v,i);total+=v.d[4999999];free(v.d);}
printf("%ld\n",total);return 0;}
