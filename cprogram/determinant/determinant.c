/*
 * ���ܣ���������ʽ��ֵ
 * ���ߣ���
 * */
#include <stdio.h>

void getData(int n, double data[n][n]);
int preProcess(int n, double data[n][n]);
int process(int n, double data[n][n]);
double getResult(int n, double data[n][n]);
void showResult(double produce, int n, double data[n][n]);

int main (void)
{
    int n = 0, flag = 1;
    printf("����������ʽ�Ľ���: ");
    scanf("%d", &n);
    double data[n][n], product = 0;

    getData(n, data);  //get the number of the determinant from keyboard or one file
    flag = preProcess(n, data);  //make sure that the first number of the first line is not zero, or the next process will be interrupt
    if (flag != 0) {
        flag = process(n, data);  //process the data in the determinant, after this you can get a simple determinant
        product = getResult(n, data);  //calculate the result of the determinant
    }
    product *= flag;
    showResult(product, n, data);  //show the processed determinant and the final result

    printf("Press any key to continue...\n");
    getchar();
    getchar();

    return 0;
}

void getData(int n, double data[n][n])
{
    int i, j;
    printf("����������ʽ:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            scanf("%lf", &data[i][j]);
        }
    }
}

void exchange(int n, double array1[n], double array2[n])
{
    int i = 0;
    double temp = 0;

    for (i = 0; i < n; i++) {
        temp = array1[i];
        array1[i] = array2[i];
        array2[i] = temp;
    }
}

int rowSearch(int n, double array[n][n])
{
    int i = 0, j = 0, count = 0;

    for (i = 0; i < n; i++) {
        count = 0;
        for (j = 0; j < n; j++) {
            if (array[i][j] != 0) {
                break;
            }
            count++;
        }
        if (count == n) {
            return i;
        }
    }
    return -1;
}

//ȷ��������Ԫ�ز�Ϊ��
int preProcess(int n, double data[n][n])
{
    int i, temp = 0, flag = 1;

    for (i = 0; i < n; i++) {
        if (data[i][0] != 0) {  //������е�һ��Ԫ�ز�Ϊ�㣬��ס���е�λ��
            temp = i;
            break;
        } else {
            if (i == n - 1) {   //����ѵ���ĩ�м���һ��ȫ��Ԫ�ض�Ϊ�㣬����ʽΪ��
                return 0;
            }
            continue;           //û�����һ�������������
        }
    }
    if (temp != 0) {
        exchange(n, data[temp], data[0]);
        flag *= -1;             //ÿ����һ������ʽ���
    }
    return flag;
}

int process(int n, double data[n][n])
{
    int i, j, k;
    double t;

    for (i = 0; i < n - 1; i++) {
        if(rowSearch(n, data) != -1) {
            return 0;
        }
        for (j = i + 1; j < n; j++) {
            t = data[j][i] / data[i][i];
            for (k = i; k < n; k++) {
                data[j][k] += (-1) * t * data[i][k];
            }
        }
    }
}

double getResult(int n, double data[n][n])
{
    int i;
    double produce = 1;
    for (i = 0; i < n; i++) {
        produce *= data[i][i];
    }
    return produce;
}

void showResult(double product, int n, double data[n][n])
{
    int i, j;
    printf("\n����������ʽ:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%7.3f ", data[i][j]);
        }
        putchar('\n');
    }
    printf("\n����ʽ��ֵ: %.3f \n", product);
}
