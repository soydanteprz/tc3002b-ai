import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.*;

public class ArmChairs {
    static final Random random = new Random();

    

    

    

    public static void main(String[] args) {
        FastReader fs = new FastReader();
        int t = 1;

        for (int z = 0; z < t; z++) {
            int n = fs.nextInt();
            List<Integer> empty = new ArrayList<>();
            List<Integer> chairs = new ArrayList<>();
            for(int i = 0; i < n; i++) {
                int status = fs.nextInt();
                if(status == 1) chairs.add(i+1);
                else empty.add(i+1);
            }
            int[][] dp = new int[empty.size() + 1][chairs.size() + 1];
            dp[0][0] = 0;
            for(int i = 1; i <= chairs.size(); i++) dp[0][i] = (int)3e+8;
            for(int i = 1; i <= empty.size(); i++) {
                for(int j = 1; j <= chairs.size(); j++) {
                    // Shift jth person to ith chair
                    dp[i][j] = dp[i-1][j-1] + Math.abs(empty.get(i-1) - chairs.get(j-1));
                    dp[i][j] = Math.min(dp[i][j], dp[i-1][j]);
                }
                //System.out.println(i + " " + Arrays.toString(dp[i]));
            }
            //System.out.println(empty.size() + " " + chairs.size());
            System.out.println(dp[empty.size()][chairs.size()]);
        }
    }
}
