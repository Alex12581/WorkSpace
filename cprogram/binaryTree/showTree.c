#define _CRT_SECURE_NO_WARNINGS  //ʹ����scanf��strcp�ȱ�VS�ж�Ϊ����ȫ�ĺ������Դ˾佫��ȫ�������к���

#include <stdio.h>
#include <math.h>
#include "showTree.h"
#include "binaryTree.h"
#include "queue.h"
#include "queueQueue.h"

static void traverseTree(const Tree *tree, int *floor, QueueQueue *queueQueue);
static void printBlocks(FILE *fp, char opt, int num);
static void printLines(FILE *fp, char opt, int num);


/*
  ���ݽṹ˵����
  �˳���ʹ����������У�ÿ���������Ӧһ���ڶ���Queue���洢��ͬһ���е����ݣ�
  �����ڶ��е�head�ڵ��������QueueQueue����洢��
*/
void showTree(const Tree *tree, FILE *fp)
{
	int floor = 0;

	if (fp == NULL) {
		fp = fopen("./tree.txt", "w");
	}  //���δָ���ļ������Ĭ�����������ǰִ��·��

	//������
	QueueQueue queueQueue;
	InitializeQueueQueue(&queueQueue);

	//�ݹ����������
    traverseTree(tree, &floor, &queueQueue);

	while (queueQueue.total != 0)
	{  //��������У����ڵ��������
		Queue *queue = RemoveQueue(&queueQueue).queue;

		while (queue->total != 0)
		{  //�����ڶ��У����ڵ��������
			Item item = Remove(queue);

			//������Ҫ���ٿ�հ�(�̺���)��num=2^(floor-1)��numΪ�հ׿�(�̺���)������floorΪ������Ҳ�ǽڵ���
			int num = (int)pow(2, queueQueue.total-1);
			num = (num < 0) ? 0 : num;

			printBlocks(fp, 'l', num);  //��ӡ����ǰ�Ŀհף�'l'��ʾ"left"
			printLines(fp, 'l', num);  //��ӡ����ǰ�Ķ̺���
			fprintf(fp, "%d", item.value);  //��ӡ��ǰ���нڵ������
			printLines(fp, 'r', num);  //��ӡ���ݺ�Ķ̺��ߣ�'r'��ʾ"right"
			printBlocks(fp, 'r', num);  //��ӡ���ݺ�Ŀհ�
		}
		fprintf(fp, "\n");  //����
	}

	if (fp != stdout) {
		fclose(fp);  //�ر��ļ����
		if (fp == NULL) {
			printf("tree.txt�����ļ��ѱ��浽����ǰִ��·��\n");
		}
		else {
			printf("�����ѱ��浽ָ���ļ�\n");
		}
	}
}

//�ݹ������������treeΪ �Ե�ǰ�ڵ�Ϊ���ڵ� ��������floor��¼��ǰ���ڲ�����queueQueueΪ�����
static void traverseTree(const Tree *tree, int *floor, QueueQueue *queueQueue)
{
	//������ӡ��ʾ��������
    if (TreeIsEmpty(tree)) {
            fprintf(stderr, "Empty tree, no data.\n");
            return;
    }

	//��ʱ���ڵ㣬�洢�Ե�ǰ�ڵ�Ϊ���ڵ������
	Tree tempTree;
    InitializeTree(&tempTree);
    tempTree = *tree;

	Trnode *p = tempTree.root;  //�洢��ǰ�ڵ㣬Ҳ����ʱ�����ĸ��ڵ�

	//������ʼֵΪ0��������>=����г��ȵ�ʱ��˵�����ǵ�һ�ν�����һ�㣬����ҪΪ��һ���½�һ���ڵ�
	if (*floor >= GetLengthQueue(queueQueue)) {
		AppendQueue(queueQueue, *NewDataQueue(NewQueue()));
	}

	//��ȡ�뵱ǰ���Ӧ���ڶ��У�����ǰ���ڵ�����׷�ӵ��ڶ���ĩβ
	Queue *queue = GetItemQueue(queueQueue, *floor).queue;
	Append(queue, p->item);

	//�ݹ����
    if (TreeHasEmpty(&tempTree)) {  //��������ĩ�ң�������
		(*floor)--;
        return;
    }
    else {
        if (p->left != NULL) {  //��߲�Ϊ�գ�������
            tempTree.root = p->left;
			(*floor)++;
            traverseTree(&tempTree, floor, queueQueue);
        }
        if (NULL != p->right) {  //�ұ߲�Ϊ�գ�������
            tempTree.root = p->right;
			(*floor)++;
            traverseTree(&tempTree, floor, queueQueue);
        }
    }
	(*floor)--;  //���߶��߹��ˣ�������
    return;
}

//��ӡ�հ׿�ĺ������˷������ö��⿪��
static void printBlocks(FILE *fp, char opt, int num)
{
	num *= BLOCK_LENGTH;  //�궨��ÿ��հ׵ĳ��ȣ�Ĭ��Ϊ1

	if (opt == 'l') {  //��ӡ������ߵĿհ�
		while (num > 0)
		{
			fprintf(fp, " ");
			num--;
		}
	}
	else if (opt == 'r') {  //��ӡ�����ұߵĿհ�
		while (num > 1)
		{
			fprintf(fp, " ");
			num--;
		}
	}
}

static void printLines(FILE *fp, char opt, int num)
{
	num *= BLOCK_LENGTH;  //�궨��ÿ��̺��ߵĳ��ȣ�Ĭ��Ϊ1

	if (opt == 'l') {  //��ӡ������ߵĶ̺���
		if (num > 0) fprintf(fp, "|");
		while (num > 1)
		{
			fprintf(fp, "-");
			num--;
		}
	}
	else if (opt == 'r') {  //��ӡ�����ұߵĶ̺���
		while (num > 1)
		{
			fprintf(fp, "-");
			num--;
		}
		fprintf(fp, "|");
	}
}