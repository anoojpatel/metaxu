#include <stdio.h>
int main(void) {
    long total = 0;
    for (double fy = 0.0; fy < 800.0; fy += 1.0) {
        for (double fx = 0.0; fx < 800.0; fx += 1.0) {
            double x0 = fx / 400.0 - 1.5, y0 = fy / 400.0 - 1.0;
            double x = 0.0, y = 0.0;
            long i = 0;
            while (i < 50) {
                if (x * x + y * y > 4.0) { i = 99; }
                else {
                    double xt = x * x - y * y + x0;
                    y = 2.0 * x * y + y0;
                    x = xt;
                    i++;
                }
            }
            if (i < 99) total++;
        }
    }
    printf("%ld\n", total);
    return 0;
}
