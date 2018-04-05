public class Account
{
    private double balance;
    private String details = "��֧\t�˻����\t��֧���\t˵  ��\n";

    public Account() {
        this.balance = 0;
    }
    public Account(double balance) {
        this.balance = balance;
    }

    public String getDetails() {
        return details;
    }

    public void incomeProcess(double income, String document) {
        balance += income;
        details = getDetails() + "����\t" + balance + "\t\t" + income + "\t\t" + document + "\n";
    }

    public void spendingProcess(double spending, String document) {
        balance -= spending;
        details = getDetails() + "֧��\t" + balance + "\t\t" + spending + "\t\t" + document + "\n";
    }
}
