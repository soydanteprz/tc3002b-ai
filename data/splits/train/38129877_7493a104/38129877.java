import java.util.Scanner;

public class C {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int t = scanner.nextInt();
        while (t-- > 0) {
            int n = scanner.nextInt();
            int[] h = new int[n];
            for (int i = 0; i < n; i++) {
                h[i] = scanner.nextInt();
            }
            System.out.println(binSearchSolution(h));
        }
    }

    private static int binSearchSolution(int[] a) {
        int l = 0;
        int r = (int)1e9 + 1;
        while (l < r - 1) {
            int m = l + (r - l) / 2;
            if (canAchieve(a, m)) {
                l = m;
            } else {
                r = m;
            }
        }
        return l;
    }

    private static boolean canAchieve(int[] a, int min) {
        int back1 = a[a.length - 2];
        int current = a[a.length - 1];
        int nextIndex = a.length - 3;
        while (nextIndex >= 0) {
            int back2 = a[nextIndex];
            if (current < min) {
                return false;
            }
            int canMove = Math.min(current - min, a[nextIndex + 2]);
            int d = canMove / 3;
            current = back1 + d;
            back1 = back2 + 2 * d;
            nextIndex--;
        }
        if (current < min || back1 < min) {
            return false;
        }
        return true;
    }
}
