import java.io.File;
import java.io.IOException;
import java.util.*;

public class C {
    public static void main(String[] args){
        Scanner in = new Scanner(System.in);
        int t = in.nextInt();
        while (t-- > 0){
            int n = in.nextInt();
            int[][] cnt = new int[n][5];
            int[] len = new int[n];
            for (int i = 0; i < n; i++){
                String s = in.next();
                len[i] = s.length();
                for (char c : s.toCharArray()) cnt[i][c-'a']++;
            }
            int max = 0;
            for (int i = 0; i < 5; i++){
                int[] diff = new int[n];
                for (int j = 0; j < n; j++) diff[j] = cnt[j][i] - (len[j] - cnt[j][i]);
                Arrays.sort(diff);
                int j = n-2, sum = diff[n-1];
                while (j>=0 && sum > 0){
                    max = Math.max(max, n - 1 - j);
                    sum += diff[j]; j--;
                }
                if (sum > 0) max = Math.max(max, n);
            }
            System.out.println(max);
        }
    }
}
