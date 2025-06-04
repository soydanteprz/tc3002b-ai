import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
public class D {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        List<Integer> a1 = new ArrayList<>();
        List<Integer> a0 = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int now = sc.nextInt();
            if (now == 1) {
                a1.add(i + 1);
            } else {
                a0.add(i + 1);
            }
        }
        int[][] dp = new int[a1.size() + 1][a0.size() + 1];
        for (int i = 1; i < dp.length; i++) {
            dp[i][i] = dp[i - 1][i - 1] + Math.abs(a1.get(i - 1) - a0.get(i - 1));
            for (int j = i + 1; j < dp[i].length; j++) {
                dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j - 1] + Math.abs(a1.get(i - 1) - a0.get(j - 1)));
            }
        }
        System.out.println(dp[a1.size()][a0.size()]);
    }
}
