package familyAccount;
import java.util.Scanner;

public class FamilyAccount
{
    public static void main(String[] args) throws java.io.IOException{
        int option;
        double money;
        String document;
        boolean loopFlag = true;

        Scanner scanner = new Scanner(System.in);
        Account account = new Account(10000);

        Menu.menu();
        while(loopFlag == true) {
            System.out.print("��ѡ��(1-4)��");
            option = scanner.nextInt();
            switch(option) {
                case 1: System.out.println(account.getDetails()); break;
                case 2: 
                        System.out.print("���������");
                        money = scanner.nextDouble();
                        System.out.print("��������˵����");
                        document = scanner.next();
                        account.incomeProcess(money, document); break;
                case 3: 
                        System.out.print("����֧����");
                        money = scanner.nextDouble();
                        System.out.print("����֧��˵����");
                        document = scanner.next();
                        account.spendingProcess(money, document); break;
                case 4: 
                        System.out.print("ȷ���Ƿ��˳�(Y/N)��");
                        if (System.in.read() == (int)'Y') {
                            loopFlag = false;
                        }
                        break;
                default: System.out.println("ѡ����������ԡ�"); break;
            }
        }
        System.out.println("���˳���");
    }
}
