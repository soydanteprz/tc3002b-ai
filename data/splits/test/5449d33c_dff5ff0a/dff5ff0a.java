import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Arrays;

public class Main {

    private static void run() throws IOException {
        int n = in.nextInt();
        char[][] a = new char[n][];
        for (int i = 0; i < n; i++) {
            a[i] = in.next().toCharArray();
        }
        int ans = Integer.MIN_VALUE;
        for (char now = 'a'; now <= 'e'; now++) {
            ans = Math.max(ans, check(a, now));
        }
        out.println(ans);
    }

    private static int check(char[][] a, char target) {
        int[] count = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            for (char c : a[i]) {
                if (c == target) {
                    count[i]++;
                } else {
                    count[i]--;
                }
            }
        }
        Arrays.sort(count);
        int ans = 0;
        int sum = 0;
        for (int i = a.length - 1; i >= 0; i--) {
            if (count[i] <= 0) break;

            ans++;
            sum += count[i];
        }

        if (ans == 0) return 0;

        for (int i = a.length - 1; i >= 0; i--) {
            if (count[i] > 0) continue;

            if (sum > -count[i]) {
                sum += count[i];
                ans++;
            }
        }
        return ans;
    }

    public static void main(String[] args) throws IOException {
        in = new Reader();
        out = new PrintWriter(new OutputStreamWriter(System.out));

        int t = in.nextInt();
        for (int i = 0; i < t; i++) {
            run();
        }

        out.flush();
        in.close();
        out.close();
    }

    

    static final long mod = 1000000007;

    

    

    @SuppressWarnings("FieldCanBeLocal")
    private static Reader in;
    private static PrintWriter out;

    

    

    

    

    
}
