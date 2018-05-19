#include <stdio.h>

void printArray(int *a, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        printf("%2d ", a[i]);
    }
    putchar('\n');

}

//��������
void insert(int *a, int n)
{
    int i, j, temp;

    for (i = 1; i < n; i++) {
        temp = a[i];
        for (j = i; j > 0; j--) {
            if (a[j-1] > temp) {
                a[j] = a[j-1];
            } else {
                break;
            }
        }
        a[j] = temp;
    }
}

int main(void)
{
    int n, i;

    printf("n = ");
    while(scanf("%d", &n)) {

        int a[n];

        //��������
        printf("array:\n");
        for (i = 0; i < n; i++) {
            scanf("%d", &a[i]);
        }

        //��������
        insert(a, n);

        printArray(a, n);
        printf("n = ");
    }

    return 0;
}
