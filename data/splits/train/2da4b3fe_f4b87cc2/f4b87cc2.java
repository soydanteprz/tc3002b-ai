import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class D {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);
        int t = 1;
        for (int i = 0; i < t; i++) {
            solve(sc, pw);
        }
        pw.close();
    }
    static void solve(Scanner in, PrintWriter out){
        int n = in.nextInt();
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = in.nextInt();
        }
        int[] pre = new int[n + 1];
        List<Integer> z = new ArrayList<>();
        List<Integer> o = new ArrayList<>();
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            if (arr[i] == 0) {
                cnt++;
                z.add(i);
            }else{
                o.add(i);
            }
        }
        int[][] dp = new int[cnt + 1][(n - cnt) + 1];
        int one = n - cnt;
        Arrays.fill(dp[0], 100000000);
        dp[0][0] = 0;
//        System.out.println(one);
        int min = 100000000;
        for (int i = 1; i <= cnt; i++) {// the position of zero,
            Arrays.fill(dp[i], 100000000);
            dp[i][0] = 0;
            for (int j = 1; j <= Math.min(one, i); j++) {// the position of one
                if (cnt - i < one - j){
                    dp[i][j] = 100000000;
                }else{
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + Math.abs(o.get(j - 1) - z.get(i - 1)), dp[i - 1][j]);
                }
                if (j == one){
                    min = Math.min(min, dp[i][j]);
                }
            }
        }
        min = Math.min(min, dp[0][n - cnt]);
        out.println(min);
    }



    

    

    // method to return LCM of two numbers
    

    

    //    reverse division for 2
    
    

    
    
}
