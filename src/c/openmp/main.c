#include <stdio.h>
#include <stdlib.h>
#include "array_pow10.h"

int main() {
    long n = ARRAY_SIZE ;
    double *arr = malloc(sizeof(double)*n);
    
    printf("Starting array init. \n");
    for (long i=0; i<n; i++) {
        arr[i] = ARRAY_INIT_VALUE ;
    }
    printf("Done array init.\n");

    printf("Starting pow10.\n");
    array_pow10(arr, n);
    printf("Done pow10.\n");

    printf("Array value is: %f", arr[0]);
    free(arr);

    return 0;
}
