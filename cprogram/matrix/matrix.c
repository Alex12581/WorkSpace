/*
 * ���ܣ��Ծ�����и����
 *       1�����������ת��������;���
 *       2������
 *       3������
 * ���ޣ�ֻ�ܴ����־�����ͬ�е����
 * ���ߣ���
 * */

#include <stdio.h>
#include <math.h>

void getData(int row, int column, double arr[row][column]);
int convert(int row, int column, double arr[row][column], int option);
int rank(int row, int column, double arr[row][column]);
void printArray(int row, int column, double arr[row][column]);
void inverse(int row, int column, double arr[row][column], int option);
void copyArray(int row, int column, double origin_arr[row][column], double target_arr[row][column]);

int main(void)
{
    int row = 0, column = 0, option = 0;
    printf("������ ����: ");
    scanf("%d", &row);
    printf("������ ����: ");
    scanf("%d", &column);
    printf("�Ƿ�չ������(1-��, 0-��)");
    scanf("%d", &option);

    double data[row][column];
    printf("\n���������(�Կո��س����): \n");
    getData(row, column, data);

    double cp_data[row][column];
    copyArray(row, column, data, cp_data);

    int rank = convert(row, column, data, option);
    printf("�������Ϊ: %d\n", rank);

    printf("����ξ���Ϊ:\n");
    printArray(row, column, data);

    printf("����:\n");
    if (rank == row) {
        inverse(row, column, cp_data, option);  //��������Ҫ��ԭʼ����
    } else {
        printf("<����>\n");  //�Ƿ�������������
    }

    printf("Press any key to continue...\n");
    getchar();
    getchar();

    return 0;
}

//����ָ������������(�ڲ����ܺ���)
void exchange(int row, int column, double arr[row][column], int origin, int target)
{
    int i = 0;
    double temp = 0;

    if (origin != target) {
        for (i = 0; i < column; i++) {
            temp = arr[origin][i];
            arr[origin][i] = arr[target][i];
            arr[target][i] = temp;
        }
    }
}

//�������������һ�������н���(�ڲ����ܺ���)
void preConvert(int row, int column, double arr[row][column])
{
    int i = 0, target = 0;

    for (i = 0; i < row; i++) {
        if (arr[i][0] != 0) {
            target = i;
        }
    }
    exchange(row, column, arr, 0, target);
}

//����Ƿ��з����У����򻻵�����(�ڲ����ܺ���)
int rowSearch(int row, int column, double arr[row][column])
{
    int i = 0, j = 0, count = 0, count1 = 0;

    while(1) {
        for (i = 0; i < row-count1; i++) {
            count = 0;
            for (j = 0; j < column; j++) {
                if (fabs(arr[i][j]-0) < 0.000001) {
                    count++;
                }
            }
            if (count == column) {
                exchange(row, column, arr, i, row-1-count1);
                count1++;  //������е�����
            }
        }
        break;
    }

    return count1;

}


//��ȡ��������
void getData(int row, int column, double arr[row][column])
{
    int i = 0, j = 0;

    for (i = 0; i < row; i++) {
        for (j = 0; j < column; j++) {
            scanf("%lf", &arr[i][j]);
        }
    }
}

//ת��Ϊ�����
int convert(int row, int column, double arr[row][column], int option)
{
    int i = 0, j = 0, k = 0, count1 = 0;
    double rate = 0;

    //������е�һ��Ԫ��Ϊ�㣬�����Ԥ����
    if (arr[0][0] == 0) {
        preConvert(row, column, arr);
    }

    //��Ϊ�����Ǿ���
    for (i = 0; i < row - 1 - count1; i++) {
        for (j = i + 1; j < row; j++) {
            rate = arr[j][i] / arr[i][i];
            for (k = i; k < column; k++) {
                arr[j][k] -= arr[i][k] * rate;
            }
        }
        count1 = rowSearch(row, column, arr);
    }
    if (option == 1) {
        printf("(��Ϊ�����Ǿ���)\n");
        printArray(row, column, arr);
    }

    //����
    int r = rank(row, column, arr);

    //����Ҳ��Ϊ��
    for (i = r - 1; i > 0; i--) {
        for (j = i - 1; j >= 0; j--) {
            rate = arr[j][i] / arr[i][i];
            for (k = i; k < column; k++) {
                arr[j][k] -= arr[i][k] * rate;
            }
        }
    }
    if (option == 1) {
        printf("(����Ҳ��Ϊ��)\n");
        printArray(row, column, arr);
    }

    //���ݴ�ȫ����Ϊ1
    for (i = 0; i < r; i++) {
        rate = 1 / arr[i][i];
        for (j = i; j < column; j++) {
            arr[i][j] *= rate;
        }
    }
    if (option == 1) {
        printf("(���ݴ�ȫ����Ϊ1)\n");
        printArray(row, column, arr);
    }

    return r;
}


//����
int rank(int row, int column, double arr[row][column])
{
    int i = 0, j = 0, rank = row, count = 0;

    for (i = 0; i < row; i++) {
        count = 0;
        for (j = 0; j < column; j++) {
            if ( fabs(arr[i][j]-0) < 0.000001) {
                count++;
            }
        }
        if (count == column) {
            rank--;
        }
    }

    return rank;
}

//����
void inverse(int row, int column, double arr[row][column], int option)
{
    if (row != column) {
        printf("<����>\n");  //���Ƿ�������
    } else {
        int i = 0, j = 0;
        double extendArr[row][column+row];

        //׼������
        for (i = 0; i < row; i++) {
            for (j = 0; j < column; j++) {
                extendArr[i][j] = arr[i][j];
            }
        }
        for (i = 0; i < row; i++) {
            for (j = column; j < column+row; j++) {
                if ((j - row) == i) {
                    extendArr[i][j] = 1;
                } else {
                    extendArr[i][j] = 0;
                }
            }
        }

        //�����������
        convert(row, (column+row), extendArr, option);

        //��ӡ����
        printf("��Ϊ��\n");
        for (i = 0; i < row; i++) {
            for (j = column; j < column+row; j++) {
                printf("%7.3f ", extendArr[i][j]);
            }
            putchar('\n');
        }

    }
}

//��ӡ����
void printArray(int row, int column, double arr[row][column])
{
    int i, j;

    for (i = 0; i < row; i++) {
        for (j = 0; j < column; j++) {
            printf("%7.3f ", arr[i][j]);
        }
        putchar('\n');
    }
}

//��������
void copyArray(int row, int column, double origin_arr[row][column], double target_arr[row][column])
{
    int i = 0, j = 0;

    for (i = 0; i < row; i++) {
        for (j = 0; j < column; j++) {
            target_arr[i][j] = origin_arr[i][j];
        }
    }
}

