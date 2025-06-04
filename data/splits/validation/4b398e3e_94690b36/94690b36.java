import java.util.Scanner;

public class VupsenPupsenand0 {

    public static void main(String args[]) {
        Scanner scan = new Scanner(System.in);
        int t = scan.nextInt();
        while (t-- > 0) {
            int n = scan.nextInt();
            int[] a = new int[n];
            for (int i = 0; i < n; i++) {
                a[i] = scan.nextInt();
            }
            if (n % 2 == 0) {
                for (int i = 0; i < n; i += 2) {
                    System.out.print(a[i + 1] + " " + -a[i] + " ");
                }
                System.out.println();
            } else {
                for (int i = 0; i < n - 3; i += 2) {
                    System.out.print(a[i + 1] + " " + -a[i] + " ");
                }
                int x = a[n - 3];
                int y = a[n - 2];
                int z = a[n - 1];
                if (x + y != 0) {
                    System.out.println(-z + " " + (-z) + " " + (x + y));
                } else if (y + z != 0) {
                    System.out.println((y + z) + " " + (-x) + " " + (-x));
                } else {
                    System.out.println(-y + " " + (x + z) + " " + (-y));
                }
            }
        }
    }

}
