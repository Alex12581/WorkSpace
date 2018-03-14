#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<windows.h>
#include<time.h>
#include<iostream>
HANDLE hSerial; //�����ڵľ��
HANDLE th;      //�����̵߳ľ��
DCB dcb;        //�洢���ڵ�����
HWND hwnd;      //�����ڵľ��
DWORD error;    //���ͺͽ��յ�ʱ������
COMSTAT status;  //���ͺͽ��յ�ʱ������
void Serialreceive();
DWORD WINAPI ThreadProc1(LPVOID lpPraraneter)  //�̺߳�������ͣ�ض�
{
	while (true)
	{
		Serialreceive();
	}
	return  1;
}


void ShutDown()  //�ػ�
{
	HANDLE hToken;
	TOKEN_PRIVILEGES tkp;
	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
		return;
	LookupPrivilegeValue(NULL, SE_SHUTDOWN_NAME, &tkp.Privileges[0].Luid);
	tkp.PrivilegeCount = 1;
	tkp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
	AdjustTokenPrivileges(hToken, FALSE, &tkp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
	ExitWindowsEx(EWX_SHUTDOWN | EWX_FORCE, 0);
}


int SerialBegin()  //�򿪴���
{
	hSerial = CreateFile("com4", GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);  //��
																								   //���豸�������в�ѯ���ںŲ����Ĳ��������ںŲ��ܳ���10������������10����Ĵ��ںţ��������аٶ�
	if (hSerial == INVALID_HANDLE_VALUE)  //�����ʧ��
	{
		printf("error");
		return 0;
	}
	return 1;

}


void Serialsetting()
{
	SetupComm(hSerial, 1024, 1024);  //����TX,RX�Ļ�������С��Ϊ1024
	GetCommState(hSerial, &dcb);    //�õ�hSerial�еĳ�ʼ���ݸ�dcb
	dcb.BaudRate = 9600;           //������
	dcb.ByteSize = 8;              //�ֽ�
	dcb.Parity = NOPARITY;         //��żУ��
	dcb.StopBits = ONESTOPBIT;     //ֹͣλ
	SetCommState(hSerial, &dcb);   //��dcb�е����ݸ�hSerial
	PurgeComm(hSerial, PURGE_TXCLEAR | PURGE_RXCLEAR);  //���Tx��Rx�Ļ�����
	printf("�򿪳ɹ�!\n");
}


int Serialwrite(char *write)  //д��
{
	DWORD bytesend = 0;//Ҫд�������
	while (write[bytesend] != '\0')  //�õ��ַ���*write�Ĵ�С
	{
		bytesend++;
	}
	if (!WriteFile(hSerial, write, bytesend, &bytesend, NULL))  //���д�벻�ɹ�
	{
		ClearCommError(hSerial, &error, &status);  //�������
		return 0;
	}
	else { printf("д��ɹ�\n"); return 1; }
}


void Serialreceive()
{
	char str[50];
	DWORD wcount = 5;  //Ҫ��������ֽ�
	memset(str, '\0', 50);  //str��ʼ��
	BOOL read;
	ClearCommError(hSerial, &error, &status);
	if (status.cbInQue > 0)  //�����ȡ��Чֵ����0
	{
		wcount = status.cbInQue;
		read = ReadFile(hSerial, str, wcount, &wcount, NULL); //��ȡ
		printf("�յ�%s\n", str);
		/*��������Ҫ��Ӵ���ĵط�*/
        printf("\n***Attention! Your computer will shut down in 5 secends!!!***\n");
		/*��������Ҫ��Ӵ���ĵط�*/
		ClearCommError(hSerial, &error, &status);  //��մ���
		PurgeComm(hSerial, PURGE_TXABORT | PURGE_RXABORT | PURGE_RXCLEAR | PURGE_TXCLEAR);  //���
    }
}


int main()
{
	SerialBegin();
	Serialsetting();
	DWORD tid;
	th = CreateThread(NULL, 0, ThreadProc1, 0, CREATE_SUSPENDED, &tid);   //�����߳�
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);    //�����߳����ȼ�
	SetThreadAffinityMask(th, 1);                                     //�Լ��ٶ�
	ResumeThread(th);                                                //�߳̿�ʼ����
	while (1)
	{
		char a[50] = { 0 };
		std::cin >> a;   //c++�﷨,��ʵ����scanf();
		Serialwrite(a);
        Serialreceive();
	}
}
