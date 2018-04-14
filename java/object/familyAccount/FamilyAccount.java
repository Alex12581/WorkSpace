package familyAccount;
import java.util.Scanner;

public class FamilyAccount
{
    Scanner scanner = new Scanner(System.in);
    Account account = new Account(10000);

    public static void main(String[] args) {
        FamilyAccount unit = new FamilyAccount();
        unit.mainMenu();
    }

    public void mainMenu() {
        int option;
        boolean loopFlag = true;

        while(loopFlag == true) {
            menu();
            if (scanner.hasNextInt()) {
                option = scanner.nextInt();
            } else {
                scanner.next();
                option = -1;
            }
            switch(option) {
                case 1: System.out.println(account.getDetails()); break;
                case 2: income(); break;
                case 3: spending(); break;
                case 4: loopFlag = quit(); break;
                default: System.out.println("��Ҫһ��ѡ��(1-4)�������ԡ�"); break;
            }
        }
        System.out.println("���˳���");
    }

    public void menu() {
        System.out.println("-----------------��ͥ��֧�������-----------------");
        System.out.println();
        System.out.println("                   1 ��֧��ϸ");
        System.out.println("                   2 �Ǽ�����");
        System.out.println("                   3 �Ǽ�֧��");
        System.out.println("                   4 ��    ��");
        System.out.println();
        System.out.print("        ��ѡ��(1-4)��");
    }

    public void income() {
        double money = 0;
        String document;

        System.out.print("���������");
        money = readDouble(false);  //����ֵΪ�˷����ڲ�һ�����ܵĿ���
        System.out.print("��������˵����");
        document = readString(); 
        account.incomeProcess(money, document);
    }

    public void spending() {
        double money;
        String document;

        System.out.print("����֧����");
        money = readDouble(true);
        System.out.print("����֧��˵����");
        document = readString();
        account.spendingProcess(money, document);
    }

    public boolean quit() {
        String temp;

        System.out.print("ȷ���Ƿ��˳�(Y/N)��");
        temp = scanner.next();
        if (temp.equals("Y") || temp.equals("y")) {
            return false;
        } else {
            return true;
        }
    }

    public double readDouble(boolean optionFlag) {
        double money = 0;
        while(scanner.hasNext()) {
            try {
                money = Double.parseDouble( scanner.next() );
                if (money < 0) {
                    System.out.print("����Ϊ���������ԣ�");
                }
                else if (optionFlag && money > account.getBalance()) {
                //���optionFlagΪ�棬��ִ�д˴���飬�ж�����Ľ���Ƿ�������
                    System.out.print("����Ϊ�����˻��������ԣ�");
                }
                else {
                    break;
                }
            } catch(NumberFormatException e) {
                System.out.print("���ӦΪ���֣�������:");
            }
        }
        return money;
    }

    public String readString() {
        String document = "";
        while(scanner.hasNext()) {
            document = scanner.next();
            if (document.length() < 1 || document.length() > 8) {
                System.out.print("�����ֶι��������������룺");
            } else {
                break;
            }
        }
        return document;
    }
}
