//CountTheNumbers.c
//���ɣ�natural numbers ��Ϊ���������Ѿ�������������Ϊ-1
//����©������
#include <stdio.h>

int main(void)
{
    int n, i, j, k, count;
    scanf("%d", &n);

    int data[n], num[n], times[n];

    for (i = 0; i < n; i++) {
        scanf("%d", &data[i]);
    }

    //����
    for (i = 0, k = 0; i < n; i++) {
        if (data[i] == -1) {
            continue;
        }

        count = 0;
        num[k] = data[i];
        
        for (j = 0; j < n; j++) {
            if (num[k] == data[j]) {
                count++;
                data[j] = -1;
            }
        }

        times[k] = count;
        k++;
    }

    //����
    int temp, min, mini;
    for (i = 0; i < k - 1; i++) {
        min = num[i];
        mini = i;
        for (j = i + 1; j < k; j++) {
            if (min > num[j]) {
                min = num[j];
                mini = j;
            }
        }

        temp = num[i];
        num[i] = num[mini];
        num[mini] = temp;

        temp = times[i];
        times[i] = times[mini];
        times[mini] = temp;
    }

    //���
    for (i = 0; i < k; i++) {
        printf("%d %d\n", num[i], times[i]);
    }

    return 0;
}
