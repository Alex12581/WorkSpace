public class Test {
	
	public static void main(String[] args) {
        Frock frock = new Shirt(180, "��ɫ", 200);
        System.out.println(frock.calcArea());

        Clothing clothing = new Shirt(190, "��ɫ", 200);
        System.out.println(clothing.calcArea());
	}
	
}
