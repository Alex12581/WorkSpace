package CustomerView;
import java.util.Scanner;

/*�����࣬�淶�û�����*/

public class CMUtility
{
    //ȷ����ù淶����������
    public static int readInt(boolean optionFlag) {
        Scanner scanner = new Scanner(System.in);
        
        int input = 0;
        while(scanner.hasNext()) {
            try {
                input = Integer.parseInt( scanner.next() );
                if (optionFlag && input < 0) {
                    //�����ܿ��ش򿪣���optionFlagΪtrueʱ���˴����ִ�У�����ִ��
                    System.out.print("���벻��Ϊ���������ԣ�");
                }
                else {
                    break;
                }
            } catch(NumberFormatException e) {
                System.out.print("����ӦΪ���֣�������:");
            }
        }
        //scanner.nextLine();
        return input;
    }

    //ȷ���û�ѡ��Y/N
    public static boolean forSure() {
        Scanner scanner = new Scanner(System.in);

        String temp = scanner.next();
        if (temp.equals("Y") || temp.equals("y")) {
            return true;
        } else {
            return false;
        }
    }
}
