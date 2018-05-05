package FamilyAccount;

public class Utility
{
    private double balance;
    private StringBuffer details = new StringBuffer("��֧\t�˻����\t��֧���\t˵  ��\n");

    public Utility() {
        this.balance = 0;
    }
    public Utility(double balance) {
        this.balance = balance;
    }

    public double getBalance() {
        return balance;
    }

    public StringBuffer getDetails() {
        return details;
    }

    //��������
    public void incomeProcess(double income, String document) {
        balance += income;
        details = details.append("����\t" + balance + "\t\t" + income + "\t\t" + document + "\n");
    }
    //����֧��
    public void spendingProcess(double spending, String document) {
        balance -= spending;
        details = details.append("֧��\t" + balance + "\t\t" + spending + "\t\t" + document + "\n");
    }
}
