package CustomerView;
import java.util.Scanner;

/*����*/

public class CustomerView
{
    CustomerList customers = new CustomerList(10);
    Scanner scanner = new Scanner(System.in);

    //main������ȡ���������һ��ʵ��������ʵ���ϵ�������enterMainMenu()
    public static void main(String[] args) {
        CustomerView init = new CustomerView();
        init.enterMainMenu();
    }

    //ʵ���ϵ�������
    public void enterMainMenu() {
        int option;
        boolean loopFlag = false;

        do {
            System.out.print("\n\n---------------------�ͻ���Ϣ�������--------------------- \n\n" +
            "                      1 �� �� �� ��\n" +
            "                      2 �� �� �� ��\n" +
            "                      3 ɾ �� �� ��\n" +
            "                      4 �� �� �� ��\n" +
            "                      5 ��       ��\n\n" +
            "                      ��ѡ��(1-5)��");
            option = CMUtility.readInt(true);
            switch (option) {
                case 1: addNewCustomer(); break;    //���ӿͻ�
                case 2: modifyCustomer(); break;    //�޸Ŀͻ�
                case 3: deleteCustomer(); break;    //ɾ���ͻ�
                case 4: listAllCustomers(); break;  //��ӡ�ͻ��б�
                case 5: System.out.print("ȷ���Ƿ��˳�(Y/N) : ");
                        loopFlag = CMUtility.forSure();  //����û�ѡ��ΪY/y����true�����򷵻�false
                        break;
                default: System.out.println("û�д�ѡ������ԡ�"); break;
            }
        } while (!loopFlag);  //loopFlagΪ �� ʱѭ������
        System.out.println("���˳���");
    }

    //���ӿͻ�
    private void addNewCustomer() {
        Customer customer = new Customer();

        System.out.println("\n-------------------------��ӿͻ�------------------------- ");
        
        System.out.print("������");
        customer.setName(scanner.next());
        System.out.print("�Ա�");
        customer.setGender(scanner.next());
        System.out.print("���䣺");
        customer.setAge(CMUtility.readInt(true));
        System.out.print("�绰��");
        customer.setPhone(scanner.next());
        System.out.print("���䣺");
        customer.setEmail(scanner.next());
        scanner.nextLine();  //���ջس��������Ժ����������Ӱ�죬������ÿ�������û��󣬶Ը��û��ĵ�һ���޸�ʱ��

        if (customers.addCustomer(customer)) {
            System.out.println("-------------------------������-------------------------");
        } else {
            System.out.println("�ռ��������޷���ӡ�");
        }
    }

    //�޸Ŀͻ�
    private void modifyCustomer() {
        Customer customer = new Customer();
        int index;

        System.out.println("\n-------------------------�޸Ŀͻ�------------------------- ");
        System.out.print("��ѡ����޸Ŀͻ����(-1�˳�)��");
        if (( index = CMUtility.readInt(false) - 1 ) != -2) {    //���û�����-1��������indexΪ-2ʱ�����޸�
            String temp;
            Customer originalCustomer;
            if (customers.getCustomer(index) != null && customers.getCustomer(index).getName() != null) {
            //ֻ�е����صĲ��ǿն���ʱ�Ž����޸Ĳ�����
            //���ؿն���˵�������Ч����else����ӡ��ʾ
                originalCustomer = customers.getCustomer(index);            

                //�޸�����
                System.out.print("����(" + originalCustomer.getName() + "):");
                temp = scanner.nextLine();
                if (temp.equals("")) {
                    customer.setName(originalCustomer.getName());
                } else {
                    customer.setName(temp);
                }
                //�޸��Ա�
                System.out.print("�Ա�(" + originalCustomer.getGender() + "):");
                temp = scanner.nextLine();
                if (temp.equals("")) {
                    customer.setGender(originalCustomer.getGender());
                } else {
                    customer.setGender(temp);
                }
                //�޸�����
                System.out.print("����(" + originalCustomer.getAge() + "):");
                temp = scanner.nextLine();
                int age;
                if (temp.equals("")) {
                    customer.setAge(originalCustomer.getAge());
                    /*�˲����޸�����Ĵ��뻹û����ã��ԵÆ��¡�*/
                } else {
                    try {
                        age = Integer.parseInt(temp);
                        if (age > 0) {
                            customer.setAge(age);
                        } else {
                            System.out.println("��⵽����������δ�޸ġ�");
                            customer.setAge(originalCustomer.getAge());
                        }
                    } catch(NumberFormatException e) {
                        System.out.println("��⵽�������ַ�������δ�޸ġ�");
                        customer.setAge(originalCustomer.getAge());
                    }
                }
                //�޸ĺ���
                System.out.print("�绰����(" + originalCustomer.getPhone() + "):");
                temp = scanner.nextLine();
                if (temp.equals("")) {
                    customer.setPhone(originalCustomer.getPhone());
                } else {
                    customer.setPhone(temp);
                }
                //�޸�����
                System.out.print("��������(" + originalCustomer.getEmail() + "):");
                temp = scanner.nextLine();
                if (temp.equals("")) {
                    customer.setEmail(originalCustomer.getEmail());
                } else {
                    customer.setEmail(temp);
                }

            //�����Чʱ��ӡ��ʾ
            } else {
                System.out.println("�����Ч��������");
            }

            if (customers.replaceCustomer(index, customer)) {
                System.out.println("-------------------------�޸����------------------------- ");
            }
        }
    }

    //ɾ���ͻ�
    private void deleteCustomer() {
        int index;

        System.out.println("\n-------------------------ɾ���ͻ�-------------------------");
        System.out.print("��ѡ���ɾ���ͻ����(-1�˳�)��");
        if (( index = CMUtility.readInt(false) - 1) != -2) {      //���û�����-1��������indexΪ-2ʱ����ɾ��
            String temp;
            System.out.print("ȷ���Ƿ�ɾ��(Y/N)��");
            if (CMUtility.forSure()) {
                if (customers.deleteCustomer(index)) {
                    System.out.println("-------------------------ɾ�����-------------------------");
                } else {
                    System.out.println("�ÿͻ������ڣ�������Ч");
                }
            } else {
                System.out.println("δɾ��");
            }
        }
    }

    //��ӡ�ͻ��б�
    private void listAllCustomers() {
        Customer[] allCustomers = new Customer[10];
        allCustomers = customers.getAllCustomers();

        System.out.println("\n-------------------------�ͻ��б�-------------------------\n");
        System.out.println("���\t����\t�Ա�\t����\t�绰\t\t����");
        for (int i = 0; i < 10; i++) {
            if (allCustomers[i] != null && allCustomers[i].getName() != null) {
                System.out.println((i + 1) + "\t" + allCustomers[i].getName() + "\t"+ allCustomers[i].getGender() +
                    "\t" + allCustomers[i].getAge() + "\t" + allCustomers[i].getPhone() + "\t" + allCustomers[i].getEmail());
            }
        }
        System.out.println("\n-----------------------�ͻ��б����-----------------------");
    }
}
