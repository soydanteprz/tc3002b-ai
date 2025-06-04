import java.util.Scanner;

public class D {
    public static void main(String[] args) {
        Scanner scn = new Scanner(System.in);
        int t = scn.nextInt();
        StringBuilder sb = new StringBuilder();
        while (t-- > 0) {
            int n = scn.nextInt();
            int[] arr = new int[n];
            int[] ans = new int[n];
            for (int i = 0; i < n; i++) {
                arr[i] = scn.nextInt();
            }
            if (n % 2 != 0) {
                if (arr[n - 2] + arr[n - 3] == 0) {
                    ans[n - 3] = -2 * arr[n - 1];
                    ans[n - 2] = -arr[n - 1];
                    ans[n - 1] = arr[n - 2] + arr[n - 3] * 2;
                } else {
                    ans[n - 3] = -arr[n - 1];
                    ans[n - 2] = -arr[n - 1];
                    ans[n - 1] = arr[n - 2] + arr[n - 3];
                }
                n -= 3;
            }
            for (int i = 0; i < n; i += 2) {
                int div = 1;
                if (arr[i] % 2 == 0 && arr[i + 1] % 2 == 0)
                    div = 2;
                ans[i] = arr[i + 1] / div;
                ans[i + 1] = -arr[i] / div;
            }
            for (int v : ans) {
                sb.append(v);
                sb.append(" ");
            }
            sb.append("\n");
        }
        System.out.println(sb);
    }
}
