#include <stdio.h>
typedef struct { double x, y, vx, vy; } Body;
static Body step(Body b) {
    double r2 = b.x * b.x + b.y * b.y + 0.5;
    double ax = 0.0 - b.x / r2, ay = 0.0 - b.y / r2;
    Body n = { b.x + b.vx * 0.001, b.y + b.vy * 0.001,
               (b.vx + ax * 0.001) * 0.9999995,
               (b.vy + ay * 0.001) * 0.9999995 };
    return n;
}
int main(void) {
    Body b = { 1.0, 0.0, 0.0, 1.0 };
    for (long i = 0; i < 20000000; i++) b = step(b);
    printf("%.17g\n%.17g\n", b.x, b.y);
    return 0;
}
