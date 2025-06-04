import java.util.Scanner;

public class D {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        int lines = s.nextInt();

        s.nextLine();

        for (int i = 0; i < lines; i += 1) {
            solve(s.nextInt(), s);
        }
    }


    public static void solve(int n, Scanner s) {
        int a[] = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = s.nextInt();
        }

        int b[] = new int[n];
        for (int i = 0; i + 1 < n; i += 2) {
            b[i + 1] = a[i];
            b[i] = -a[i + 1];
        }

        while (b[n - 2] == 0 || b[n - 1] == 0) {
            b[n - 1] += a[n - 2];
            b[n - 2] -= a[n - 1];
        }
        
        StringBuilder erg = new StringBuilder();
        for (int i = 0; i < n; i++) {
            erg.append(b[i]).append(" ");
        }

        System.out.println(erg);
    }
}
