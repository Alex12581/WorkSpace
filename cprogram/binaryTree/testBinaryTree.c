#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "binaryTree.h"
#include "testBinaryTree.h"
#include "showTree.h"

static int getInt(int *tar, char *tip);
static bool confirm();
static void saveTree(FILE *pdata, const Tree *tree);

int main(void)
{
    bool quit = false;
    int option;

    Tree tree;
    InitializeTree(&tree);

    readTreeFromFile(&tree);

    printf("��ӭ��������������ϵͳ");
    help();

    do {
        getInt(&option, "���ѡ��");

        switch (option) {
            case HELP: help(); break;
            case SHOW: printItems(&tree); break;
            case ADD: addItems(&tree);  break;
            case SEARCH: searchItem(&tree); break;
            case DELETE: deleteItem(&tree); break;
            case QUIT: quit = quitConfirm(); break;
			case PRINT_TO_FILE: showTree(&tree, NULL); break;
			case PRINT_TO_SCREEN: showTree(&tree, stdout); break;
            default: printf("��Чѡ��\n"); break;
        }
    } while(!quit);
    DeleteAllTrnode(&tree);
    printf("���˳�\n");

    return 0;
}

//��ӡ����
void printItems(const Tree *tree)
{
    printf("\nname\t\t\t value\t\t Trnode\t\t\t left\t\t\t right\n\n");
    printTrnode(tree);
    printf("��%d����¼\n\n", TrnodesCount(tree));
}
//��������
void addItems(Tree *tree)
{
    if (TreeIsFull(tree)) {
        printf("�ռ�����, �޷���������\n");
        return;
    }

    char name[NAME_LENGTH];
    int value = 0;
    Item item;

    printf("����: ����name, value. �ո����, # �س� ����\n");

    scanf("%s", name);
    while (0 != strcmp(name, "#")) {
        scanf("%d", &value);

        strcpy(item.name, name);
        item.value = value;

        if (false == AddTrnode(tree, item)) {
            break;
        }
        
        scanf("%s", name);
    }
    saveTreeToFile(tree);
}
//��������
void searchItem(const Tree *tree)
{
    char name[NAME_LENGTH];
    Trnode *pnode = NULL, *plast = NULL;

    printf("����: ����name:");
    scanf("%s", name);
    
    Item item;
    strcpy(item.name, name);
    item.value = -1;

    if (SearchTrnode(tree, item, &pnode, &plast)) {
        printf("\nname\t\t\t value\t\t left\t\t\t right\n\n");
        printf("%-20s\t %-10d\t %-15p\t %-15p\n",(pnode->item).name, (pnode->item).value, pnode->left, pnode->right);
    } else {
        printf("�޽��\n");
    }
}
//ɾ������
void deleteItem(Tree *tree)
{
    char name[NAME_LENGTH];

    printf("ɾ��: ����name:");
    scanf("%s", name);
    
    Item item;
    strcpy(item.name, name);
    item.value = -1;

    DeleteTrnode(tree, item);

    saveTreeToFile(tree);
}

//��ӡ��ʾ
void help()
{
    printf("����ѡ��: -1:�˳� 0:��ʾ���� 1:�������� 2:��ʾ���� 3:�������� 4:ɾ������ 5:ͼ�λ���ӡ���ݵ��ն� 6:ͼ�λ���ӡ���ݵ��ļ�\n");
}
//�˳�
bool quitConfirm ()
{
    printf("ȷ���˳�?[y/n]");
    return confirm();
}


//���ļ���ȡ���� data.txt
void readTreeFromFile(Tree *tree)
{
    FILE *pdata = fopen("./data.txt", "a+");
    fseek(pdata, 0L, SEEK_SET);

    char name[NAME_LENGTH];
    int value, statu = -1;

    while ((statu = fscanf(pdata, "%s", name)) != EOF) {
        statu = fscanf(pdata, "%d", &value);

        Item item;
        strcpy(item.name, name);
        item.value = value;

        AddTrnode(tree, item);
    }

    fclose(pdata);
    printf("���ݶ�ȡ���, ��%d����¼\n", TrnodesCount(tree));
}

//�������ݵ��ļ� data.txt
static void saveTree(FILE *pdata, const Tree *tree)
{

    if (TreeIsEmpty(tree)) {
        // fprintf(stderr, "Empty tree, no data.\n");
        return;
    }

	//������ʱ����Ϊ�������Ŀ���
	Tree tempTree;
	InitializeTree(&tempTree);
	tempTree = *tree;

	//��¼��ʱ���ĸ��ڵ�
	Trnode *p = tempTree.root;

	//�ݹ����
	if (TreeHasEmpty(&tempTree)) {  //��·����,������һ��
        fprintf(pdata, "%s %d ",(p->item).name, (p->item).value);
		return ;
	} else {
        //��������ڵݹ�ǰ����, ��ӡ��˳��Ҫ�͵��������˳����ͬ, �������Ľṹ�����ı�
        fprintf(pdata, "%s %d ",(p->item).name, (p->item).value);
		if (p->left != NULL) {  //�����·,������
			tempTree.root = p->left;
		    saveTree(pdata, &tempTree);
		}
		if (NULL != p->right) {  //�ұ���·,������
			tempTree.root = p->right;
			saveTree(pdata, &tempTree);
		}
	}  
	return ;
}
void saveTreeToFile(const Tree *tree)
{
    FILE *pdata = fopen("./data.txt", "w");

    saveTree(pdata, tree);

    fclose(pdata);
}



/*����*/
//���һ������������Ŀ�������ַ����ʾ�ַ����������һ��scanf�ķ���ֵ����
static int getInt(int *tar, char *tip)
{
    int statu;

    printf("������%s��", tip);
    while ((statu = scanf("%d", tar)) != 1) { //��÷���Ҫ�������Ϊֹ
        printf("��Ǹ��%c�Ƿ����˴���Ҫһ��������\n", getchar());
        while (getchar() != '\n');  //����ʣ�������ַ�
        printf("������%s��", tip);
    }
    while (getchar() != '\n');  //����ʣ�������ַ�

    return statu;
}
//ȷ���Ƿ�ִ�в������û�����y����-1�����򷵻�1
static bool confirm()
{
    char choice;

    scanf("%c", &choice);
    if (choice == 'y' || choice == 'Y') {
        return true;   
    } else {
        return false;
    }
}

/*
int main(void)
{
    Tree tree;
    int i;

    Item items[5];
//    items[5]{"AAA", 0, "bbb", 1, "Abb", 2, "bAa", 3, "AbA", 4};
    strcpy(items[0].name, "AAA");
    strcpy(items[1].name, "bbb");
    strcpy(items[2].name, "Abb");
    strcpy(items[3].name, "bAa");
    strcpy(items[4].name, "AbA");

    items[0].value = 0;
    items[1].value = 1;
    items[2].value = 2;
    items[3].value = 3;
    items[4].value = 4;

    InitializeTree(&tree);
    for (i = 0; i < 5; i++) {
        AddTrnode(&tree, &items[i]);
    }

    printf("TreeIsEmpty: %d\n", TreeIsEmpty(&tree));
    printf("TreeHasEmpty: %d\n", TreeHasEmpty(&tree));
    printf("TreeIsFull: %d\n", TreeIsFull(&tree));
    printf("TreeHasFull: %d\n", TreeHasFull(&tree));


    printf("TrnodesCount: %d\n", TrnodesCount(&tree));

    printItems(&tree);

//    int floor = 1, maxrow = 1;
//    countRow(&tree, &floor, &maxrow);
//    printf("maxrow: %d\n", maxrow);

//    showTree(&tree);

    return 0;
}
*/


