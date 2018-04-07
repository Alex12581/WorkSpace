package CustomerView;
import java.util.Scanner;

public class CustomerView
{
    CustomerList customers = new CustomerList(10);
    Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        CustomerView init = new CustomerView();
        init.enterMainMenu();
    }

    public void enterMainMenu() {
        int option;
        
        do {
            System.out.print("-----------------�ͻ���Ϣ�������----------------- \n\n" +
            "                  1 �� �� �� ��\n" +
            "                  2 �� �� �� ��\n" +
            "                  3 ɾ �� �� ��\n" +
            "                  4 �� �� �� ��\n" +
            "                  5 ��       ��\n\n" +
            "                  ��ѡ��(1-5)��");
            if (scanner.hasNextInt()) {
                option = scanner.nextInt();
            } else {
                scanner.nextLine();
                option = 0;
            }
            switch (option) {
                case 1: addNewCustomer(); break;
                case 2: modifyCustomer(); break;
                case 3: deleteCustomer(); break;
                case 4: listAllCustomers(); break;
                case 5: 
                        System.out.print("ȷ���Ƿ��˳�(Y/N) : ");
                        String temp = scanner.next();
                        if (temp.equals("Y") || temp.equals("y")) {
                            System.out.println("���˳���"); 
                            break;
                        } else {
                            option = 0;
                            break;
                        }
                default: System.out.println("ѡ����������ԡ�"); break;
            }
        } while (option != 5);
    }

    private void addNewCustomer() {
        Customer customer = new Customer();

        System.out.println("---------------------��ӿͻ�--------------------- ");
        
        System.out.print("������");
        customer.setName(scanner.next());
        System.out.print("�Ա�");
        customer.setGender(scanner.next());
        System.out.print("���䣺");
        if (scanner.hasNextInt()) {
            customer.setAge(scanner.nextInt());
        } else {
            scanner.next();
        }
        System.out.print("�绰��");
        customer.setPhone(scanner.next());
        System.out.print("���䣺");
        customer.setEmail(scanner.next());

        if (customers.addCustomer(customer)) {
            System.out.println("---------------------������---------------------");
        } else {
            System.out.println("�ռ��������޷���ӡ�");
        }
    }

    private void modifyCustomer() {
        Customer customer = new Customer();
        int index;

        System.out.println("---------------------�޸Ŀͻ�--------------------- ");
        System.out.print("��ѡ����޸Ŀͻ����(-1�˳�)��");
        if (( index = scanner.nextInt() - 1 ) != -2) {
            scanner.nextLine();

            String temp;
            Customer originalCustomer = customers.getCustomer(index);
            
            System.out.print("����(" + originalCustomer.getName() + "):");
            temp = scanner.nextLine();
            if (temp.equals("")) {
                customer.setName(originalCustomer.getName());
            } else {
                customer.setName(temp);
            }

            System.out.print("�Ա�(" + originalCustomer.getGender() + "):");
            temp = scanner.nextLine();
            if (temp.equals("")) {
                customer.setGender(originalCustomer.getGender());
            } else {
                customer.setGender(temp);
            }

            System.out.print("����(" + originalCustomer.getAge() + "):");
            temp = scanner.nextLine();
            if (temp.equals("")) {
                customer.setAge(originalCustomer.getAge());
            } else {
                try {
                    customer.setAge(Integer.parseInt(temp));
                } catch (NumberFormatException e){
                    System.out.println("(�����쳣����������������δ�޸ġ�)");
                    customer.setAge(originalCustomer.getAge());
                }
            }

            System.out.print("�绰����(" + originalCustomer.getPhone() + "):");
            temp = scanner.nextLine();
            if (temp.equals("")) {
                customer.setPhone(originalCustomer.getPhone());
            } else {
                customer.setPhone(temp);
            }

            System.out.print("��������(" + originalCustomer.getEmail() + "):");
            temp = scanner.nextLine();
            if (temp.equals("")) {
                customer.setEmail(originalCustomer.getEmail());
            } else {
                customer.setEmail(temp);
            }

            if (customers.replaceCustomer(index, customer)) {
                System.out.println("---------------------�޸����--------------------- ");
            }
        }
    }

    private void deleteCustomer() {
        int index;

        System.out.println("---------------------ɾ���ͻ�---------------------");
        System.out.print("��ѡ���ɾ���ͻ����(-1�˳�)��");
        if (( index = scanner.nextInt() - 1) != -2) {
            String temp;
            System.out.print("ȷ���Ƿ�ɾ��(Y/N)��");
            temp = scanner.next();
            if (temp.equals("Y") || temp.equals("y")) {
                if (customers.deleteCustomer(index)) {
                    System.out.println("---------------------ɾ�����---------------------");
                }
            }
        }
    }

    private void listAllCustomers() {
        Customer[] allCustomers = new Customer[10];
        allCustomers = customers.getAllCustomers();

        System.out.println("---------------------------�ͻ��б�-------------------------");
        System.out.println("���\t����\t�Ա�\t����\t�绰\t\t����");
        for (int i = 0; i < 10; i++) {
            if (allCustomers[i] != null && allCustomers[i].getName() != "") {
                System.out.println((i + 1) + "\t" + allCustomers[i].getName() + "\t"+ allCustomers[i].getGender() +
                    "\t" + allCustomers[i].getAge() + "\t" + allCustomers[i].getPhone() + "\t" + allCustomers[i].getEmail());
            }
        }
        System.out.println("---------------------------�ͻ��б����---------------------");
    }
}

//����������Ϊһ����������������޷����������⡣
