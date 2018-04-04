public class Test
{
    public static void main(String [] args)
    {
        int max;  //to store the index of the largest weight 
        Pet temp;

        Pet [] animal = {
            new Cat("С��", 2, 5, "��"),
            new Cat("С��", 3, 6, "��"),
            new Dog("����", 2, 10, "̩��"),
            new Dog("С��", 3, 4, "������"),
            new Bird("С��", 1, 1, true),
            new Bird("��", 2, 2, false)
        };

        for (int i = 0; i < animal.length - 1; i++) {
            max = i;
            for (int j = i + 1; j < animal.length; j++) {
                if (animal[j].getWeight() > animal[max].getWeight()) {
                    max = j;
                }
            }
            temp = animal[i];
            animal[i] = animal[max];
            animal[max] = temp;
        }

        for (int i = 0; i < animal.length; i++) {
            System.out.println(animal[i].toString());
        }
    }
}
