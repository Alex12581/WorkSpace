//5��������

public class Triangle extends ClosedFigure {
	
	public Point p1, p2, p3;
	
	public Triangle(Point p1, Point p2, Point p3) throws IllegalArgumentException{
		
		//��
		this.p1 = p1;
		this.p2 = p2;
		this.p3 = p3;
		
//		//��
//		Line line1 = new Line(p1, p2);
//		Line line2 = new Line(p2, p3);
//		Line line3 = new Line(p1, p3);
//		
//		//����
//		double length1 = line1.length();
//		double length2 = line2.length();
//		double length3 = line3.length();
//		
//		//�жϣ�����֮�ʹ��ڵ����ߣ�����֮��С�ڵ�����
//		boolean flag1 = (length1 + length2 > length3 && Math.abs(length1 - length2) < length3);
//		boolean flag2 = (length2 + length3 > length1 && Math.abs(length2 - length3) < length1);
//		boolean flag3 = (length1 + length3 > length2 && Math.abs(length1 - length3) < length2);
//		
//		if (!(flag1 && flag2 && flag3)) {
//			throw new IllegalArgumentException();
//		}
	}
	
	public boolean check() {
		
		//��
		Line line1 = new Line(p1, p2);
		Line line2 = new Line(p2, p3);
		Line line3 = new Line(p1, p3);
		
		//����
		double length1 = line1.length();
		double length2 = line2.length();
		double length3 = line3.length();
		
		//�жϣ�����֮�ʹ��ڵ����ߣ�����֮��С�ڵ�����
		boolean flag1 = (length1 + length2 > length3 && Math.abs(length1 - length2) < length3);
		boolean flag2 = (length2 + length3 > length1 && Math.abs(length2 - length3) < length1);
		boolean flag3 = (length1 + length3 > length2 && Math.abs(length1 - length3) < length2);
		
		if (!(flag1 && flag2 && flag3)) {
			throw new IllegalArgumentException();
		} else {
			return true;
		}
	}
	
	public boolean check2() throws MyFigureException {
		
		//��
		Line line1 = new Line(p1, p2);
		Line line2 = new Line(p2, p3);
		Line line3 = new Line(p1, p3);
		
		//����
		double length1 = line1.length();
		double length2 = line2.length();
		double length3 = line3.length();
		
		//�жϣ�����֮�ʹ��ڵ����ߣ�����֮��С�ڵ�����
		boolean flag1 = (length1 + length2 > length3 && Math.abs(length1 - length2) < length3);
		boolean flag2 = (length2 + length3 > length1 && Math.abs(length2 - length3) < length1);
		boolean flag3 = (length1 + length3 > length2 && Math.abs(length1 - length3) < length2);
		
		if (!(flag1 && flag2 && flag3)) {
			throw new MyFigureException("�����㲻�ܹ���������");
		} else {
			return true;
		}
	}
}
