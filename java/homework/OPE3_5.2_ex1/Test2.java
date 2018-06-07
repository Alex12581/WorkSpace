/*
1. ��дPerson�࣬�������������䡢���ص����ԣ���
����Ӧ�ķ��ʷ�����
2. ��д������Test1����main�����д���������ͬ��Per
son���󣬽��������������л����ļ��У�
3. ʹ�����������࣬���ļ��з����л�����Person��
�󣬴�ӡ�������֤���л�����ȷ
*/

import java.io.*;

public class Test2
{
    public static void main(String[] args) {

        Person[] persons = {
        new Person("1", 1, 1), 
        new Person("2", 2, 2), 
        new Person("3", 3, 3) 
        };
        Person[] persons1 = new Person[3];

        File file = new File("persons.txt");

        try {
            ObjectOutputStream out = new ObjectOutputStream( new FileOutputStream(file) );
            out.writeObject(persons);
            out.close();
            
            ObjectInputStream in = new ObjectInputStream( new FileInputStream(file) );
            persons1 = (Person[])in.readObject();
            in.close();

            for (Person p : persons1) {
                System.out.println(p.toString());
            }

        } catch(Exception e) {
            System.out.println("Something error...");
        }
    }
}
