import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 */
public class Solution {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Main solver = new Main();
        boolean multipleTC = true;
        int testCount = multipleTC ? Integer.parseInt(in.next()) : 1;
        for (int i = 1; i <= testCount; i++)
            solver.solve(in, out, i);
        out.close();
    }
    static class Main {
        PrintWriter out;
        InputReader in;

        public void solve(InputReader in, PrintWriter out, int test) {
            this.out = out;
            this.in = in;
            int n = ni();
            String[] arr = new String[n];
            int[][] freq = new int[n][5];
            int[][] rem = new int[n][5];
            for(int i = 0; i < n; i++){
                arr[i] = n();
                for(int j = 0; j < arr[i].length(); j++)
                    freq[i][arr[i].charAt(j) - 'a']++;
                for(int j = 0; j < 5; j++)
                    rem[i][j] = arr[i].length() - freq[i][j];
            }
            int ans = 0;
            for(int i = 0; i < 5; i++){
                int[] vals = new int[n];
                for(int j = 0; j < n; j++)
                    vals[j] = freq[j][i] - rem[j][i];
                Arrays.sort(vals);
                int sum = 0, x = 0;
                for(int j = n - 1; j >= 0; j--){
                    if(sum + vals[j] > 0){
                        x++;
                        sum += vals[j];
                    } else {
                        break;
                    }
                }
                if(x > ans) {
                    ans = x;
                }
            }
            System.out.println(ans);
        }

        


        String n(){
            return in.next();
        }

        int ni() {
            return in.nextInt();
        }

        


        

    }

    
}
