/*
 * ���ܣ��Ծ�����и����
 *       1�����������ת��������;���
 *       2������
 *       3������
 *       4�����Ӧ������Ľ�
 * ���ߣ���
 * */

#include <stdio.h>

void getData(int row, int column, double arr[row][column]);
void convert(int row, int column, double arr[row][column]);
int rank(int row, int column, double arr[row][column]);
void printArray(int row, int column, double arr[row][column]);
void inverse(int row, int column, double arr[row][column]);

int main(void)
{
    int row = 0, column = 0;
    printf("������ ���� �� ���� (�Կո��س����): ");
    scanf("%d%d", &row, &column);

    double data[row][column];
    printf("\n���������(�Կո��س����): \n");
    getData(row, column, data);
    
    printf("�������Ϊ:\n");   //��������Ҫ��ԭʼ���ݣ������治��ı�ԭʼ����
    inverse(row, column, data);//����������ڸı�ԭʼ���ݵĺ���֮ǰ

    convert(row, column, data);

    printf("����ξ���Ϊ:\n");
    printArray(row, column, data);

    printf("�������Ϊ: %d\n", rank(row, column, data));

    return 0;
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

//�������������һ�������н���
void preConvert(int row, int column, double arr[row][column])
{
    int i = 0, j = 0, target = 0;
    double temp = 0;

    for (i = 0; i < row; i++) {
        if (arr[i][0] != 0) {
            target = i;
        }
    }
    for (i = 0; i < column; i++) {
        temp = arr[0][i];
        arr[0][i] = arr[target][i];
        arr[target][i] = temp;
    }
}

//��Ϊ�����
void convert(int row, int column, double arr[row][column])
{
    int i = 0, j = 0, k = 0;
    double rate = 0;

    //������е�һ��Ԫ��Ϊ�㣬�����Ԥ����
    if (arr[0][0] == 0) {
        preConvert(row, column, arr);
    }

    //��Ϊ�����Ǿ���
    for (i = 0; i < row - 1; i++) {
        for (j = i + 1; j < row; j++) {
            rate = arr[j][i] / arr[i][i];
            for (k = i; k < column; k++) {
                arr[j][k] -= arr[i][k] * rate;
            }
        }
    }

    //����Ҳ��Ϊ��
    for (i = row - 1; i > 0; i--) {
        for (j = i - 1; j >= 0; j--) {
            rate = arr[j][i] / arr[i][i];
            for (k = i; k < column; k++) {
                arr[j][k] -= arr[i][k] * rate;
            }
        }
    }

    //���ݴ�ȫ����Ϊ1
    for (i = 0; i < row; i++) {
        rate = 1 / arr[i][i];
        for (j = i; j < column; j++) {
            arr[i][j] *= rate;
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

//����
int rank(int row, int column, double arr[row][column])
{
    int i = 0, rank = 0;

    for (i = 0; i < row; i++) {
        if (arr[i][i] == 1) {
            rank++;
        } else {
            break;
        }
    }

    return rank;
}

//����
void inverse(int row, int column, double arr[row][column])
{
    if (row != column) {
        printf("����\n");
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
        convert(row, (column+row), extendArr);

        //��ӡ����
        for (i = 0; i < row; i++) {
            for (j = column; j < column+row; j++) {
                printf("%7.3f ", extendArr[i][j]);
            }
            putchar('\n');
        }

    }
}
