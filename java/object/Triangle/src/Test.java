//������
public class Test {

	public static void main(String[] args) {

		//����������
		Point p1, p2, p3;
		p1 = new Point(1, 1);
		p2 = new Point(2, 2);
		p3 = new Point(3, 3);
		
		try {
			Triangle triangle1 = new Triangle(p1, p2, p3);
//			if (triangle1.check()) {
//				System.out.println("���Թ���������");
//			}
			if (triangle1.check2()) {
				System.out.println("���Թ���������");
			}
		} 
		catch(MyFigureException e) {
			System.out.println("�����Զ����쳣");
			System.out.println(e);
		}
		catch(IllegalArgumentException e) {
			System.out.println("�����㲻�ܹ���������");
		} 
		finally {
			System.out.println("���������˳�");
		}
	}

}
