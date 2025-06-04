import java.io.*;
import java.util.*;

public class ArmChairs {
    

    public static int solution(int n, int[] arr) {

        ArrayList<Integer> one = new ArrayList<Integer>();
        ArrayList<Integer> zero = new ArrayList<Integer>();

        for (int i = 0; i < n; i++) {
            if (arr[i] == 1) {
                one.add(i);
            } else {
                zero.add(i);
            }
        }

        int[][] dp = new int[one.size() + 1][zero.size() + 1];

        for (int i = 1; i <= one.size(); i++) {
            dp[i][i] = dp[i - 1][i - 1] + Math.abs(one.get(i - 1) - zero.get(i - 1));
            for (int j = i + 1; j <= zero.size(); j++) {
                dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j - 1] + Math.abs(one.get(i - 1) - zero.get(j - 1)));
            }
        }

        return dp[one.size()][zero.size()];
    }


    public static void main(String[] args) throws IOException{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter log = new BufferedWriter(new OutputStreamWriter(System.out));

        int n = Integer.parseInt(br.readLine());

        String[] s = br.readLine().split(" ");
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = Integer.parseInt(s[i]);
        } 

        log.write(Integer.toString(solution(n, arr)) + "\n");
        log.flush();
        
    }

}
